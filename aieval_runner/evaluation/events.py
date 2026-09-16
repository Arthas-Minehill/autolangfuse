from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


JsonObject = Dict[str, Any]

MAX_EVENT_FILE_BYTES = 64 * 1024 * 1024
MAX_EVENT_LINE_CHARS = 16 * 1024 * 1024
MAX_EVENT_COUNT = 10_000
MAX_TOOL_FIELD_CHARS = 1_000
MAX_STATUS_CHARS = 200
MAX_RESULT_ROWS = 100
MAX_RESULT_DEPTH = 8
MAX_RESULT_STRING_CHARS = 20_000

_TRACKED_EXECUTION_TYPES = {
    "commandExecution",
    "dynamicToolCall",
    "mcpToolCall",
}
_TOOL_NAMESPACE_SEPARATOR_RE = re.compile(r"(?:__|[./])")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class EventSummary:
    tool_calls: List[JsonObject] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    started_execution_ids: Set[str] = field(default_factory=set)

    @property
    def error(self) -> Optional[str]:
        reasons = list(self.errors)
        if self.started_execution_ids:
            reasons.append(
                f"incomplete_execution_count={len(self.started_execution_ids)}"
            )
        return "; ".join(reasons) if reasons else None


def _normalized_identifier(value: Any) -> str:
    return _NON_ALNUM_RE.sub("", str(value or "").strip().lower())


def expected_tool_leaf(value: Any) -> str:
    parts = [
        part
        for part in _TOOL_NAMESPACE_SEPARATOR_RE.split(str(value or "").strip())
        if part
    ]
    return _normalized_identifier(parts[-1] if parts else value)


def canonical_tool_name(call: JsonObject) -> str:
    server = str(call.get("server") or "").strip()
    tool = str(call.get("tool") or "").strip()
    if server and tool:
        return f"{server}.{tool}"
    return tool or server or "unknown_tool"


def tool_name_matches(expected: Any, call: JsonObject) -> bool:
    expected_leaf = expected_tool_leaf(expected)
    actual_leaf = _normalized_identifier(call.get("tool"))
    return bool(expected_leaf and actual_leaf and expected_leaf == actual_leaf)


def _compact_value(value: Any, *, depth: int = 0) -> Any:
    """保留评估所需的 JSON，同时限制单次 Metabase 返回的体积。"""
    if depth >= MAX_RESULT_DEPTH:
        return "...[truncated-depth]"
    if isinstance(value, str):
        if len(value) > MAX_RESULT_STRING_CHARS:
            return value[:MAX_RESULT_STRING_CHARS] + "...[truncated]"
        return value
    if isinstance(value, list):
        values = [_compact_value(item, depth=depth + 1) for item in value[:MAX_RESULT_ROWS]]
        if len(value) > MAX_RESULT_ROWS:
            values.append(f"...[truncated {len(value) - MAX_RESULT_ROWS} items]")
        return values
    if isinstance(value, dict):
        return {
            str(key): _compact_value(item, depth=depth + 1)
            for key, item in list(value.items())[:200]
        }
    return value


def _compact_tool_call(item: JsonObject) -> JsonObject:
    compacted = {
        "id": str(item.get("id") or "")[:MAX_TOOL_FIELD_CHARS],
        "server": str(item.get("server") or "")[:MAX_TOOL_FIELD_CHARS],
        "tool": str(item.get("tool") or "")[:MAX_TOOL_FIELD_CHARS],
        "status": str(item.get("status") or "")[:MAX_STATUS_CHARS],
        "arguments": _compact_value(item.get("arguments")),
        "result": _compact_value(item.get("result")),
        "error": _compact_value(item.get("error")),
        "duration_ms": item.get("durationMs"),
        "_completed_event": True,
    }
    return compacted


def load_event_summary(event_path: Path) -> EventSummary:
    if event_path.stat().st_size > MAX_EVENT_FILE_BYTES:
        raise ValueError("事件文件超过允许大小")
    summary = EventSummary()
    with event_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if line_number > MAX_EVENT_COUNT:
                raise ValueError("事件数量超过允许上限")
            if len(raw_line) > MAX_EVENT_LINE_CHARS:
                raise ValueError(f"事件文件第 {line_number} 行超过允许大小")
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
            except (json.JSONDecodeError, RecursionError) as exc:
                raise ValueError(f"事件文件第 {line_number} 行不是合法 JSON") from exc
            if not isinstance(event, dict):
                continue
            method = event.get("method")
            if method in {"error", "turn/error"}:
                summary.errors.append("top_level_error")
                continue
            params = event.get("params")
            item = params.get("item") if isinstance(params, dict) else None
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            item_id = str(item.get("id") or "")
            if (
                method == "item/started"
                and item_type in _TRACKED_EXECUTION_TYPES
                and item_id
            ):
                summary.started_execution_ids.add(item_id)
                continue
            if method != "item/completed":
                continue
            if item_type in _TRACKED_EXECUTION_TYPES and item_id:
                summary.started_execution_ids.discard(item_id)
            if item_type == "mcpToolCall":
                summary.tool_calls.append(_compact_tool_call(item))
    return summary


def load_completed_tool_calls(event_path: Path) -> List[JsonObject]:
    return load_event_summary(event_path).tool_calls
