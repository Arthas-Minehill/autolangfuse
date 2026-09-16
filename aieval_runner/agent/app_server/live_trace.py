from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from aieval_runner.agent.app_server.protocol import message_matches_turn
from aieval_runner.agent.app_server.trace_mapping import _append_jsonl, _clip
from aieval_runner.integrations.langfuse import get_langfuse_sdk
from aieval_runner.storage.local import read_mapping_records
from aieval_runner.core.models import EvalCase, EvalRunContext, RunnerConfig


@dataclass
class _GenerationState:
    observation: Any
    reasoning: List[str] = field(default_factory=list)
    content: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class _ToolState:
    observation: Any
    item_id: str
    item_type: str
    name: str
    parent_observation_id: str
    metadata: Dict[str, Any]
    synthetic_start: bool
    orphan: bool


_USAGE_FIELDS = {
    "inputTokens": "input",
    "cachedInputTokens": "cache_read_input_tokens",
    "outputTokens": "output",
    "reasoningOutputTokens": "reasoning_tokens",
    "totalTokens": "total",
}
_TERMINAL_TURN_METHODS = {
    "turn/completed",
    "turn/failed",
    "turn/cancelled",
    "turn/error",
}
_COST_SOURCE = "langfuse_model_pricing"
_COST_UNAVAILABLE_MISSING_MODEL = "missing_model"


def _usage_details(value: Any) -> Dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: Dict[str, int] = {}
    for source, target in _USAGE_FIELDS.items():
        count = value.get(source)
        if isinstance(count, int) and not isinstance(count, bool):
            result[target] = count
    return result


def _cost_tracking_metadata(model: str) -> Dict[str, str]:
    metadata = {"codex.cost_source": _COST_SOURCE}
    if not str(model or "").strip():
        metadata["codex.cost_unavailable_reason"] = _COST_UNAVAILABLE_MISSING_MODEL
    return metadata


def _normalized_usage(value: Any) -> Dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        key: count
        for key, count in value.items()
        if isinstance(key, str)
        and isinstance(count, int)
        and not isinstance(count, bool)
    }


def _text_parts(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            parts.extend(_text_parts(item))
        return parts
    if isinstance(value, dict):
        for key in ("text", "content", "summary"):
            if key in value:
                return _text_parts(value.get(key))
    return []


def _tool_name(item: Dict[str, Any]) -> str:
    item_type = item.get("type")
    if item_type == "mcpToolCall":
        server = str(item.get("server") or "").strip()
        tool = str(item.get("tool") or "").strip()
        if server and tool:
            return f"{server}.{tool}"
        return tool or server or "mcpToolCall"
    if item_type == "commandExecution":
        return "exec_command"
    name = item.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    tool = item.get("tool")
    if isinstance(tool, str) and tool.strip():
        return tool.strip()
    if isinstance(tool, dict):
        nested_name = tool.get("name")
        if isinstance(nested_name, str) and nested_name.strip():
            return nested_name.strip()
    return "dynamic_tool"


def _tool_input(item: Dict[str, Any]) -> Any:
    if item.get("type") == "commandExecution":
        return {
            "command": item.get("command"),
            "cwd": item.get("cwd"),
            "command_actions": item.get("commandActions"),
        }
    if item.get("input") is not None:
        return item.get("input")
    return item.get("arguments")


def _tool_output(item: Dict[str, Any]) -> Any:
    if item.get("error") is not None:
        return item.get("error")
    if item.get("type") == "commandExecution":
        if item.get("aggregatedOutput") is not None:
            return item.get("aggregatedOutput")
        return item.get("output")
    if item.get("result") is not None:
        return item.get("result")
    return item.get("output")


def _error_text(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    if isinstance(value, dict) and value.get("message") not in (None, ""):
        return str(value["message"])
    if isinstance(value, str):
        return value
    return json.dumps(_clip(value), ensure_ascii=False, separators=(",", ":"))


def _tool_level(item: Dict[str, Any]) -> Optional[str]:
    status = str(item.get("status") or "").lower()
    if item.get("error") not in (None, "") or status in {"failed", "error"}:
        return "ERROR"
    if status == "cancelled":
        return "WARNING"
    return None


def _tool_status_message(item: Dict[str, Any]) -> Optional[str]:
    error_text = _error_text(item.get("error"))
    if error_text:
        return error_text
    status = str(item.get("status") or "").lower()
    if status not in {"failed", "error", "cancelled"}:
        return None
    suffix = ""
    exit_code = item.get("exitCode")
    if exit_code is not None and not isinstance(exit_code, bool):
        suffix = f" (exitCode={exit_code})"
    return f"tool status: {status}{suffix}"


def _tool_metadata(
    item: Dict[str, Any],
    *,
    synthetic_start: bool,
    orphan: bool,
) -> Dict[str, Any]:
    return {
        "codex.call_id": item.get("id"),
        "codex.item_type": item.get("type"),
        "codex.server": item.get("server"),
        "codex.tool": item.get("tool"),
        "codex.status": item.get("status"),
        "codex.duration_ms": item.get("durationMs"),
        "codex.exit_code": item.get("exitCode"),
        "codex.error": item.get("error"),
        "codex.synthetic_start": synthetic_start,
        "codex.orphan": orphan,
    }


class AppServerLiveTrace:
    def __init__(
        self,
        *,
        case: EvalCase,
        ctx: EvalRunContext,
        config: RunnerConfig,
        mapping_path: Path,
        thread_id: str,
        turn_id: str,
        session_id: str,
        user_prompt: str,
        event_path: Path,
        model: str,
        langfuse_trace_id: Optional[str] = None,
        langfuse_parent_observation_id: Optional[str] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._case = case
        self._ctx = ctx
        self._config = config
        self._mapping_path = mapping_path
        self._thread_id = thread_id
        self._turn_id = turn_id
        self._session_id = session_id
        self._user_prompt = user_prompt
        self._event_path = event_path
        self._model = model

        self._lf = get_langfuse_sdk(config)
        self._external_trace_context = bool(langfuse_trace_id)
        self._external_parent_observation_id = langfuse_parent_observation_id
        if langfuse_trace_id:
            self._trace_id = str(langfuse_trace_id)
        else:
            self._trace_id = self._lf.create_trace_id(
                seed=f"{ctx.eval_run_id}:{case.case_id}:{turn_id}"
            )
        self._root_metadata: Dict[str, Any] = {
            "aieval.eval_run_id": ctx.eval_run_id,
            "aieval.case_id": case.case_id,
            "aieval.dataset_name": ctx.dataset_name,
            "aieval.dataset_item_id": case.dataset_item_id,
            "aieval.response_profile": ctx.response_profile,
            "aieval.response_instructions_sha256": ctx.response_instructions_sha256,
            "codex.thread_id": thread_id,
            "codex.turn_id": turn_id,
            "codex.session_id": session_id,
            "codex.runner": "codex_app_server_live_event",
            "codex.event_path": str(event_path),
            "codex.model": model,
            **_cost_tracking_metadata(model),
        }
        if self._external_trace_context:
            self._root_metadata["codex.external_trace_context"] = True
        if self._external_parent_observation_id:
            self._root_metadata["codex.external_parent_observation_id"] = str(
                self._external_parent_observation_id
            )
        trace_context = {"trace_id": self._trace_id}
        if self._external_parent_observation_id:
            trace_context["parent_span_id"] = str(self._external_parent_observation_id)
        self._root = self._lf.start_observation(
            trace_context=trace_context,
            name="Codex Turn",
            as_type="agent",
            input=_clip(user_prompt),
            metadata=_clip(self._root_metadata),
        )
        self._active_generation: Optional[_GenerationState] = None
        self._generation_count = 0
        self._pending_generation_input: List[Dict[str, Any]] = []
        self._token_usage_total: Optional[Dict[str, Any]] = None
        self._last_accepted_usage_total: Optional[Dict[str, int]] = None
        self._final_answer = ""
        self._explicit_final_answer_seen = False
        self._active_tools: Dict[str, _ToolState] = {}
        self._generation_observation_ids: List[str] = []
        self._tool_observation_ids: List[Dict[str, Any]] = []
        self._completed_tool_ids: set[str] = set()
        self._turn_started_at: Any = None
        self._turn_completed_at: Any = None
        self._turn_duration_ms: Any = None
        self._turn_status: Optional[str] = None
        self._turn_error: Any = None
        self._root_level: Optional[str] = None
        self._root_status_message: Optional[str] = None
        self._usage_incomplete = False
        self._record: Optional[Dict[str, Any]] = None
        self._mapping_written = False
        self._failure: Optional[Exception] = None
        self._finish_error: Optional[Exception] = None
        self._root_finished = False
        self._flush_finished = False
        self._terminal = False
        self._terminal_method: Optional[str] = None

    def handle_event(self, message: Dict[str, Any]) -> None:
        if not message_matches_turn(
            message,
            thread_id=self._thread_id,
            turn_id=self._turn_id,
        ):
            return
        with self._lock:
            if self._record is not None or self._failure is not None:
                return
            method = message.get("method")
            if self._terminal:
                if method not in _TERMINAL_TURN_METHODS:
                    return
                if method != self._terminal_method:
                    return
            try:
                self._dispatch_event(message)
            except Exception as exc:
                if self._failure is None:
                    self._failure = exc
                raise

    def _dispatch_event(self, message: Dict[str, Any]) -> None:
        method = message.get("method")
        params = message.get("params")
        if not isinstance(params, dict):
            return
        if method == "turn/started" or method in _TERMINAL_TURN_METHODS:
            if method in _TERMINAL_TURN_METHODS:
                self._terminal = True
                self._terminal_method = str(method)
            self._handle_turn(method, params)
            return
        if method == "thread/tokenUsage/updated":
            self._handle_usage(params.get("tokenUsage"))
            return
        item = params.get("item")
        if not isinstance(item, dict):
            return
        item_type = item.get("type")
        if item_type == "reasoning" and method in {"item/started", "item/completed"}:
            generation = self._ensure_generation()
            if method == "item/completed":
                for text in _text_parts(item.get("summary")) + _text_parts(
                    item.get("content")
                ):
                    if text not in generation.reasoning:
                        generation.reasoning.append(text)
            return
        if item_type == "agentMessage" and method in {
            "item/started",
            "item/completed",
        }:
            generation = self._ensure_generation()
            if method == "item/completed":
                text = item.get("text")
                if isinstance(text, str) and text:
                    content = {
                        "phase": str(item.get("phase") or "assistant"),
                        "text": text,
                    }
                    if content not in generation.content:
                        generation.content.append(content)
                    if item.get("phase") == "final_answer":
                        self._final_answer = text
                        self._explicit_final_answer_seen = True
                    elif not self._explicit_final_answer_seen:
                        self._final_answer = text
            return
        if item_type in {
            "mcpToolCall",
            "commandExecution",
            "dynamicToolCall",
        }:
            if method == "item/started":
                self._start_tool(item, synthetic_start=False)
            elif method == "item/completed":
                self._complete_tool(item)

    def finish(self) -> Dict[str, Any]:
        with self._lock:
            if self._finish_error is not None:
                if self._record is not None and not self._mapping_written:
                    self._persist_mapping()
                raise self._finish_error
            if self._record is not None:
                if not self._mapping_written:
                    self._persist_mapping()
                if self._failure is not None:
                    raise self._failure
                return self._record

            cleanup_errors: List[Exception] = []
            for tool in list(self._active_tools.values()):
                self._active_tools.pop(tool.item_id, None)
                tool.metadata["codex.incomplete"] = True
                try:
                    tool.observation.update(
                        metadata=_clip(tool.metadata),
                        level="WARNING",
                        status_message="tool did not complete before trace finish",
                    )
                except Exception as exc:
                    cleanup_errors.append(exc)
                try:
                    tool.observation.end()
                except Exception as exc:
                    cleanup_errors.append(exc)
            self._active_tools.clear()
            if self._active_generation is not None:
                self._usage_incomplete = True
                cleanup_errors.extend(
                    self._end_generation_without_usage()
                )
            if self._usage_incomplete:
                self._root_metadata["codex.usage_incomplete"] = True
            if self._turn_started_at is not None:
                self._root_metadata["codex.turn_started_at"] = self._turn_started_at
            if self._turn_completed_at is not None:
                self._root_metadata["codex.turn_completed_at"] = self._turn_completed_at
            if self._turn_duration_ms is not None:
                self._root_metadata["codex.turn_duration_ms"] = self._turn_duration_ms
            if not self._root_finished:
                try:
                    self._root.update(
                        output=_clip(self._final_answer),
                        metadata=_clip(self._root_metadata),
                        level=self._root_level,
                        status_message=self._root_status_message,
                    )
                except Exception as exc:
                    cleanup_errors.append(exc)
                self._root_finished = True
                try:
                    self._root.end()
                except Exception as exc:
                    cleanup_errors.append(exc)
            if not self._flush_finished:
                self._flush_finished = True
                try:
                    self._lf.flush()
                except Exception as exc:
                    cleanup_errors.append(exc)
            self._record = self._build_record()
            try:
                self._persist_mapping()
            except Exception as exc:
                if self._failure is None and not cleanup_errors:
                    raise
                cleanup_errors.append(exc)

            failures = self._unique_failures(
                [self._failure, *cleanup_errors]
            )
            if failures:
                if len(failures) == 1:
                    self._finish_error = failures[0]
                else:
                    self._finish_error = ExceptionGroup(
                        "app-server live trace finish failed",
                        failures,
                    )
                raise self._finish_error
            return self._record

    def _build_record(self) -> Dict[str, Any]:
        record = {
                "eval_run_id": self._ctx.eval_run_id,
                "case_id": self._case.case_id,
                "dataset_name": self._ctx.dataset_name,
                "dataset_item_id": self._case.dataset_item_id,
                "trace_id": str(self._trace_id),
                "root_observation_id": str(self._root.id),
                "tool_observation_ids": self._tool_observation_ids,
                "generation_observation_ids": self._generation_observation_ids,
                "token_usage_total": self._token_usage_total,
                "turn_duration_ms": self._turn_duration_ms,
                "turn_key": (
                    f"{self._session_id}:{self._thread_id}:{self._turn_id}"
                ),
                "thread_id": self._thread_id,
                "turn_id": self._turn_id,
                "session_id": self._session_id,
                "mapping_source": "codex_app_server_live_event",
                "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if self._external_trace_context:
            record["external_trace_context"] = True
        if self._external_parent_observation_id:
            record["external_parent_observation_id"] = str(
                self._external_parent_observation_id
            )
        if self._turn_started_at is not None:
            record["turn_started_at"] = self._turn_started_at
        if self._turn_completed_at is not None:
            record["turn_completed_at"] = self._turn_completed_at
        if self._turn_status is not None:
            record["turn_status"] = self._turn_status
        if self._turn_error is not None:
            record["turn_error"] = self._turn_error
        return record

    @staticmethod
    def _unique_failures(
        failures: List[Optional[Exception]],
    ) -> List[Exception]:
        unique: List[Exception] = []
        seen: set[int] = set()
        for failure in failures:
            if failure is None or id(failure) in seen:
                continue
            seen.add(id(failure))
            unique.append(failure)
        return unique

    def _persist_mapping(self) -> None:
        if self._record is None or self._mapping_written:
            return
        for existing in reversed(read_mapping_records(self._mapping_path)):
            if (
                existing.get("turn_key") == self._record.get("turn_key")
                and str(existing.get("trace_id")) == str(self._record.get("trace_id"))
                and str(existing.get("root_observation_id"))
                == str(self._record.get("root_observation_id"))
            ):
                self._record = existing
                self._mapping_written = True
                return
        _append_jsonl(self._mapping_path, self._record)
        self._mapping_written = True

    def _ensure_generation(self) -> _GenerationState:
        if self._active_generation is not None:
            return self._active_generation
        self._generation_count += 1
        generation_input: Any
        if self._generation_count == 1:
            generation_input = self._user_prompt
        else:
            generation_input = list(self._pending_generation_input)
            self._pending_generation_input.clear()
        metadata: Dict[str, Any] = {
            "codex.step_index": self._generation_count,
            **_cost_tracking_metadata(self._model),
        }
        observation = self._root.start_observation(
            name="Codex Generation",
            as_type="generation",
            input=_clip(generation_input),
            model=self._model,
            metadata=metadata,
        )
        self._generation_observation_ids.append(str(observation.id))
        self._active_generation = _GenerationState(observation=observation)
        return self._active_generation

    def _handle_usage(self, token_usage: Any) -> None:
        if not isinstance(token_usage, dict):
            return
        total = token_usage.get("total")
        if isinstance(total, dict):
            normalized_total = _normalized_usage(total)
            if self._usage_total_is_stale(normalized_total):
                return
            if normalized_total:
                self._last_accepted_usage_total = normalized_total
            self._token_usage_total = _clip(total)
            self._root_metadata["codex.token_usage_total"] = self._token_usage_total
            self._root.update(metadata=_clip(self._root_metadata))
        if self._active_generation is None:
            return
        usage = _usage_details(token_usage.get("last"))
        if not usage:
            return
        generation = self._active_generation
        generation.observation.update(
            output=_clip(
                {
                    "content": generation.content,
                    "reasoning": generation.reasoning,
                    "tool_calls": generation.tool_calls,
                }
            ),
            usage_details=usage,
        )
        self._active_generation = None
        generation.observation.end()

    def _usage_total_is_stale(self, current: Dict[str, int]) -> bool:
        previous = self._last_accepted_usage_total
        if not current or previous is None:
            return False
        if current == previous:
            return True
        current_total = current.get("totalTokens")
        previous_total = previous.get("totalTokens")
        return (
            current_total is not None
            and previous_total is not None
            and current_total <= previous_total
        )

    def _handle_turn(self, method: Any, params: Dict[str, Any]) -> None:
        turn = params.get("turn")
        source = turn if isinstance(turn, dict) else params
        started_at = source.get("startedAt")
        completed_at = source.get("completedAt")
        duration_ms = source.get("durationMs")
        if started_at is None:
            started_at = params.get("startedAt")
        if completed_at is None:
            completed_at = params.get("completedAt")
        if duration_ms is None:
            duration_ms = params.get("durationMs")
        if started_at is not None:
            self._turn_started_at = _clip(started_at)
            self._root_metadata["codex.turn_started_at"] = self._turn_started_at
        if completed_at is not None:
            self._turn_completed_at = _clip(completed_at)
            self._root_metadata["codex.turn_completed_at"] = self._turn_completed_at
        if duration_ms is not None:
            self._turn_duration_ms = _clip(duration_ms)
            self._root_metadata["codex.turn_duration_ms"] = self._turn_duration_ms
        status = str(source.get("status") or "").lower()
        error = source.get("error")
        if error is None:
            error = params.get("error")
        if status:
            self._turn_status = status
            self._root_metadata["codex.turn_status"] = status
        if error is not None:
            self._turn_error = _clip(error)
            self._root_metadata["codex.turn_error"] = self._turn_error
        if method in {"turn/failed", "turn/error"} or status in {"failed", "error"}:
            self._root_level = "ERROR"
            self._root_status_message = _error_text(error) or "turn failed"
        elif method == "turn/cancelled" or status == "cancelled":
            self._root_level = "WARNING"
            self._root_status_message = _error_text(error) or "turn cancelled"
        self._root.update(metadata=_clip(self._root_metadata))

    def _start_tool(
        self,
        item: Dict[str, Any],
        *,
        synthetic_start: bool,
        orphan: bool = False,
    ) -> _ToolState:
        item_id = str(
            item.get("id")
            or f"{item.get('type') or 'tool'}-{len(self._tool_observation_ids) + 1}"
        )
        existing = self._active_tools.get(item_id)
        if existing is not None:
            return existing
        generation = self._ensure_generation()
        name = _tool_name(item)
        metadata = _tool_metadata(
            item,
            synthetic_start=synthetic_start,
            orphan=orphan,
        )
        observation = generation.observation.start_observation(
            name=name,
            as_type="tool",
            input=_clip(_tool_input(item)),
            metadata=_clip(metadata),
        )
        state = _ToolState(
            observation=observation,
            item_id=item_id,
            item_type=str(item.get("type") or ""),
            name=name,
            parent_observation_id=str(generation.observation.id),
            metadata=metadata,
            synthetic_start=synthetic_start,
            orphan=orphan,
        )
        self._active_tools[item_id] = state
        generation.tool_calls.append(
            _clip(
                {
                    "id": item.get("id"),
                    "type": item.get("type"),
                    "name": name,
                    "arguments": _tool_input(item),
                }
            )
        )
        self._tool_observation_ids.append(
            {
                "id": str(observation.id),
                "trace_id": str(self._trace_id),
                "parent_observation_id": state.parent_observation_id,
                "name": name,
                "type": state.item_type,
                "call_id": item.get("id"),
            }
        )
        return state

    def _complete_tool(self, item: Dict[str, Any]) -> None:
        item_id = str(item.get("id") or "")
        if item_id and item_id in self._completed_tool_ids:
            return
        tool = self._active_tools.get(item_id) if item_id else None
        if tool is None and not item_id:
            matches = [
                active
                for active in self._active_tools.values()
                if active.item_type == str(item.get("type") or "")
                and active.name == _tool_name(item)
            ]
            if len(matches) == 1:
                tool = matches[0]
                item_id = tool.item_id
        if tool is None:
            tool = self._start_tool(
                item,
                synthetic_start=True,
                orphan=True,
            )
            item_id = tool.item_id
        tool.metadata.update(
            _tool_metadata(
                item,
                synthetic_start=tool.synthetic_start,
                orphan=tool.orphan,
            )
        )
        level = _tool_level(item)
        status_message = _tool_status_message(item)
        output = _clip(_tool_output(item))
        tool.observation.update(
            input=_clip(_tool_input(item)),
            output=output,
            metadata=_clip(tool.metadata),
            level=level,
            status_message=status_message,
        )
        self._active_tools.pop(item_id, None)
        if item_id:
            self._completed_tool_ids.add(item_id)
        tool.observation.end()
        self._pending_generation_input.append(
            _clip(
                {
                    "type": "tool_result",
                    "tool_call_id": item.get("id"),
                    "name": tool.name,
                    "status": item.get("status"),
                    "output": output,
                    "error": item.get("error"),
                }
            )
        )

    def _end_generation_without_usage(self) -> List[Exception]:
        generation = self._active_generation
        if generation is None:
            return []
        self._active_generation = None
        errors: List[Exception] = []
        try:
            generation.observation.update(
                output=_clip(
                    {
                        "content": generation.content,
                        "reasoning": generation.reasoning,
                        "tool_calls": generation.tool_calls,
                    }
                )
            )
        except Exception as exc:
            errors.append(exc)
        try:
            generation.observation.end()
        except Exception as exc:
            errors.append(exc)
        return errors
