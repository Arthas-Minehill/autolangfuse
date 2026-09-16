from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from aieval_runner.agent.app_server.protocol import latest_final_agent_message, mcp_tool_events
from aieval_runner.core.json_utils import MAX_SAFE_INTEGER, json_default
from aieval_runner.integrations.langfuse import get_langfuse_sdk
from aieval_runner.storage.local import read_mapping_records
from aieval_runner.core.models import EvalCase, EvalRunContext, RunnerConfig


MAX_FIELD_CHARS = 20_000


def _clip(value: Any, *, max_chars: int = MAX_FIELD_CHARS) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and abs(value) > MAX_SAFE_INTEGER:
        return str(value)
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + f"\n...[truncated {len(value) - max_chars} chars]"
    if isinstance(value, list):
        return [_clip(item, max_chars=max_chars) for item in value]
    if isinstance(value, dict):
        return {str(key): _clip(item, max_chars=max_chars) for key, item in value.items()}
    return value


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=json_default, separators=(",", ":")) + "\n")


def _tool_name(item: Dict[str, Any]) -> str:
    server = str(item.get("server") or "").strip()
    tool = str(item.get("tool") or "").strip()
    if server and tool:
        return f"{server}.{tool}"
    return tool or server or "mcpToolCall"


def _status_level(item: Dict[str, Any]) -> Optional[str]:
    if item.get("error") or str(item.get("status") or "").lower() in {"failed", "error"}:
        return "ERROR"
    return None


def _tool_output(item: Dict[str, Any]) -> Any:
    if item.get("error") is not None:
        return item.get("error")
    return item.get("result")


def _completed_tool_items(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for event in mcp_tool_events(events):
        if event.get("method") != "item/completed":
            continue
        params = event.get("params")
        item = params.get("item") if isinstance(params, dict) else None
        if isinstance(item, dict):
            items.append(item)
    return items


def _existing_turn_mapping(mapping_path: Path, *, turn_key: str) -> Optional[Dict[str, Any]]:
    matches = [record for record in read_mapping_records(mapping_path) if record.get("turn_key") == turn_key]
    return matches[-1] if matches else None


def create_direct_event_mapping(
    *,
    case: EvalCase,
    ctx: EvalRunContext,
    config: RunnerConfig,
    mapping_path: Path,
    events: List[Dict[str, Any]],
    thread_id: str,
    turn_id: str,
    session_id: str,
    user_prompt: str,
    event_path: Path,
) -> Dict[str, Any]:
    turn_key = f"{session_id}:{thread_id}:{turn_id}"
    existing = _existing_turn_mapping(mapping_path, turn_key=turn_key)
    if existing:
        return existing

    lf = get_langfuse_sdk(config)
    for method_name in ("create_trace_id", "start_observation"):
        if not hasattr(lf, method_name):
            raise RuntimeError(f"Langfuse SDK missing required method: {method_name}; please install langfuse>=4.7.1")

    trace_id = lf.create_trace_id(seed=f"{ctx.eval_run_id}:{case.case_id}:{turn_id}")
    answer = latest_final_agent_message(events)

    root = lf.start_observation(
        trace_context={"trace_id": trace_id},
        name="Codex Turn",
        as_type="agent",
        input=_clip(user_prompt),
        output=_clip(answer),
        metadata={
            "aieval.eval_run_id": ctx.eval_run_id,
            "aieval.case_id": case.case_id,
            "aieval.dataset_name": ctx.dataset_name,
            "aieval.dataset_item_id": case.dataset_item_id,
            "codex.thread_id": thread_id,
            "codex.turn_id": turn_id,
            "codex.session_id": session_id,
            "codex.runner": "codex_app_server_direct_event",
            "codex.event_path": str(event_path),
        },
    )

    tool_observation_ids: List[Dict[str, Any]] = []
    for item in _completed_tool_items(events):
        tool = root.start_observation(
            name=_tool_name(item),
            as_type="tool",
            input=_clip(item.get("arguments")),
            output=_clip(_tool_output(item)),
            level=_status_level(item),
            status_message=str(item.get("error") or "") or None,
            metadata={
                "codex.call_id": item.get("id"),
                "codex.server": item.get("server"),
                "codex.tool": item.get("tool"),
                "codex.status": item.get("status"),
                "codex.duration_ms": item.get("durationMs"),
            },
        )
        tool.end()
        tool_observation_ids.append(
            {
                "id": str(tool.id),
                "name": _tool_name(item),
                "parent_observation_id": str(root.id),
                "call_id": item.get("id"),
            }
        )

    root.end()
    lf.flush()

    record = {
        "eval_run_id": ctx.eval_run_id,
        "case_id": case.case_id,
        "dataset_name": ctx.dataset_name,
        "dataset_item_id": case.dataset_item_id,
        "trace_id": str(trace_id),
        "root_observation_id": str(root.id),
        "tool_observation_ids": tool_observation_ids,
        "turn_key": turn_key,
        "thread_id": thread_id,
        "turn_id": turn_id,
        "session_id": session_id,
        "mapping_source": "codex_app_server_direct_event_fallback",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _append_jsonl(mapping_path, record)
    return record
