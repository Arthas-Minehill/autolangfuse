from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
import urllib.error
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aieval_runner.agent.app_server.backend import run_codex_app_server, shutdown_codex_app_server  # noqa: E402
from aieval_runner.runner.config import load_config  # noqa: E402
from aieval_runner.integrations.langfuse import LangfuseClient, api_from_config  # noqa: E402
from aieval_runner.storage.local import read_mapping_records  # noqa: E402
from aieval_runner.core.models import EvalCase, EvalRunContext  # noqa: E402


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _path_from_raw(value: Any) -> Optional[Path]:
    raw = str(value).strip() if value is not None else ""
    return Path(raw) if raw else None


def _write_summary_with_fallback(
    *,
    primary_path: Path,
    fallback_path: Path,
    data: Dict[str, Any],
) -> Path:
    errors: list[Dict[str, str]] = []
    for path in (primary_path, fallback_path):
        try:
            data["summaryPath"] = str(path)
            write_json(path, data)
            return path
        except Exception as exc:
            errors.append(
                {
                    "path": str(path),
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            data["summaryWriteErrors"] = errors
    raise RuntimeError(f"canary summary 写入失败: {errors[-1]['message']}")


def read_jsonl(path: Optional[Path]) -> list[Dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    rows: list[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _nonnegative_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number


_USAGE_TOKEN_TOTAL_KEYS = ("totalTokens", "total_tokens", "total")
_USAGE_TOKEN_INPUT_KEYS = (
    "inputTokens",
    "input_tokens",
    "promptTokens",
    "prompt_tokens",
    "input",
)
_USAGE_TOKEN_OUTPUT_KEYS = (
    "outputTokens",
    "output_tokens",
    "completionTokens",
    "completion_tokens",
    "output",
)
_TOP_LEVEL_TOKEN_TOTAL_KEYS = ("totalTokens", "total_tokens")
_TOP_LEVEL_TOKEN_INPUT_KEYS = (
    "inputTokens",
    "input_tokens",
    "promptTokens",
    "prompt_tokens",
)
_TOP_LEVEL_TOKEN_OUTPUT_KEYS = (
    "outputTokens",
    "output_tokens",
    "completionTokens",
    "completion_tokens",
)


def _token_count(
    container: Any,
    *,
    total_keys: tuple[str, ...] = _USAGE_TOKEN_TOTAL_KEYS,
    input_keys: tuple[str, ...] = _USAGE_TOKEN_INPUT_KEYS,
    output_keys: tuple[str, ...] = _USAGE_TOKEN_OUTPUT_KEYS,
) -> tuple[Optional[int], bool]:
    if not isinstance(container, dict):
        return None, True

    token_keys = total_keys + input_keys + output_keys
    present_keys = [key for key in token_keys if key in container]
    if not present_keys:
        return None, True
    if any(
        isinstance(container[key], bool)
        or not isinstance(container[key], int)
        or container[key] < 0
        for key in present_keys
    ):
        return None, False

    for key in total_keys:
        if key in container:
            return container[key], True

    input_tokens = next(
        (container[key] for key in input_keys if key in container),
        0,
    )
    output_tokens = next(
        (container[key] for key in output_keys if key in container),
        0,
    )
    return input_tokens + output_tokens, True


def _observation_total_tokens(
    observation: Dict[str, Any],
) -> tuple[Optional[int], bool]:
    selected_total: Optional[int] = None
    for key in ("usageDetails", "usage_details", "usage"):
        if key not in observation:
            continue
        candidate = observation[key]
        if not isinstance(candidate, dict):
            return None, False
        total, valid = _token_count(candidate)
        if not valid:
            return None, False
        if selected_total is None and total is not None:
            selected_total = total

    total, valid = _token_count(
        observation,
        total_keys=_TOP_LEVEL_TOKEN_TOTAL_KEYS,
        input_keys=_TOP_LEVEL_TOKEN_INPUT_KEYS,
        output_keys=_TOP_LEVEL_TOKEN_OUTPUT_KEYS,
    )
    if not valid:
        return None, False
    if selected_total is None and total is not None:
        selected_total = total
    return selected_total, True


def _parse_iso_datetime(
    value: Any,
    *,
    require_original_utc: bool = False,
) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
        offset = parsed.utcoffset()
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None or offset is None:
        return None
    if require_original_utc and offset != timedelta(0):
        return None
    return parsed.astimezone(timezone.utc)


def _observation_latency_seconds(observation: Dict[str, Any]) -> float:
    start = _parse_iso_datetime(
        observation.get("startTime") or observation.get("start_time")
    )
    end = _parse_iso_datetime(
        observation.get("endTime") or observation.get("end_time")
    )
    if start is not None and end is not None:
        return max(0.0, (end - start).total_seconds())

    for key in ("latency", "latencySeconds", "latency_seconds"):
        latency = _nonnegative_number(observation.get(key))
        if latency is not None:
            return latency
    for key in ("latencyMs", "latency_ms"):
        latency_ms = _nonnegative_number(observation.get(key))
        if latency_ms is not None:
            return latency_ms / 1000.0
    return 0.0


def _observation_type(observation: Dict[str, Any]) -> str:
    return str(observation.get("type") or "").upper()


def _observation_parent_id(observation: Dict[str, Any]) -> str:
    return str(
        observation.get("parentObservationId")
        or observation.get("parent_observation_id")
        or ""
    )


def _mapping_tool_records(
    mapping: Dict[str, Any],
) -> tuple[list[Dict[str, Any]], bool]:
    value = mapping.get("tool_observation_ids")
    if not isinstance(value, list):
        return [], False
    return (
        [item for item in value if isinstance(item, dict)],
        all(isinstance(item, dict) for item in value),
    )


def _mapping_generation_ids(
    mapping: Dict[str, Any],
) -> tuple[list[str], bool]:
    value = mapping.get("generation_observation_ids")
    if not isinstance(value, list):
        return [], False
    generation_ids = [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]
    valid = (
        len(generation_ids) == len(value)
        and len(set(generation_ids)) == len(generation_ids)
    )
    return generation_ids, valid


LOCAL_MAPPING_REQUIRED_CHECKS = frozenset(
    {
        "event_records_present",
        "event_received_at_present",
        "event_received_at_utc",
        "turn_terminal_event_seen",
        "token_usage_event_seen",
        "mapping_source_live_event",
        "trace_id_non_empty",
        "root_observation_id_non_empty",
        "generation_ids_valid",
        "generation_observation_ids_non_empty",
        "token_usage_total_valid",
        "token_usage_total_positive",
        "turn_duration_exceeds_minimum",
        "mapped_tool_records_valid",
        "mapped_tool_count_matched",
        "mapped_tool_parent_ids_matched",
        "sidecar_trace_owner_matched",
    }
)
LOCAL_REQUIRED_CHECKS = LOCAL_MAPPING_REQUIRED_CHECKS | frozenset(
    {
        "event_file_created",
        "mapping_file_created",
        "sidecar_file_created",
        "mapping_record_matched",
        "sidecar_record_matched",
        "mapping_turn_key_matched",
    }
)
REMOTE_REQUIRED_CHECKS = frozenset(
    {
        "trace_id_matched",
        "observations_is_array",
        "root_agent_observation_matched",
        "generation_ids_valid",
        "all_generation_observations_matched",
        "generation_observation_matched",
        "generation_usage_fields_valid",
        "generation_usage_total_positive",
        "agent_observation_latency_exceeds_one_second",
        "mapped_tool_records_valid",
        "mapped_tool_count_matched",
        "mapped_tool_ids_non_empty",
        "mapped_tool_ids_unique",
        "remote_mapped_tool_count_matched",
        "mapped_tool_parent_ids_matched",
    }
)


def _checks_passed(
    checks: Dict[str, bool],
    required: frozenset[str],
) -> bool:
    return bool(required) and all(checks.get(key) is True for key in required)


def _result_code_after_shutdown(
    *,
    result_code: int,
    main_error: Optional[Exception],
    shutdown_error: Optional[Exception],
) -> int:
    if shutdown_error is not None and main_error is None and result_code == 0:
        return 3
    return result_code


def _local_mapping_checks(
    *,
    mapping: Dict[str, Any],
    sidecar: Dict[str, Any],
    events: list[Dict[str, Any]],
    expected_tool_count: int,
    min_turn_duration_ms: int,
) -> tuple[Dict[str, bool], Dict[str, Any]]:
    generation_ids, generation_ids_valid = _mapping_generation_ids(mapping)
    generation_id_set = set(generation_ids)
    mapped_tools, mapped_tool_records_valid = _mapping_tool_records(mapping)
    token_usage_total = mapping.get("token_usage_total")
    total_tokens, token_usage_total_valid = _token_count(token_usage_total)
    turn_duration_ms = _number(mapping.get("turn_duration_ms"))
    received_at_values = [event.get("_received_at") for event in events]
    checks = {
        "event_records_present": bool(events),
        "event_received_at_present": bool(events)
        and all(isinstance(value, str) and value for value in received_at_values),
        "event_received_at_utc": bool(events)
        and all(
            _parse_iso_datetime(value, require_original_utc=True) is not None
            for value in received_at_values
        ),
        "turn_terminal_event_seen": any(
            event.get("method") in {"turn/completed", "turn/failed", "turn/cancelled", "turn/error"}
            for event in events
        ),
        "token_usage_event_seen": any(
            event.get("method") == "thread/tokenUsage/updated" for event in events
        ),
        "mapping_source_live_event": mapping.get("mapping_source") == "codex_app_server_live_event",
        "trace_id_non_empty": bool(mapping.get("trace_id")),
        "root_observation_id_non_empty": bool(mapping.get("root_observation_id")),
        "generation_ids_valid": generation_ids_valid,
        "generation_observation_ids_non_empty": generation_ids_valid
        and bool(generation_ids),
        "token_usage_total_valid": token_usage_total_valid,
        "token_usage_total_positive": token_usage_total_valid
        and total_tokens is not None
        and total_tokens > 0,
        "turn_duration_exceeds_minimum": (
            turn_duration_ms is not None and turn_duration_ms > min_turn_duration_ms
        ),
        "mapped_tool_records_valid": mapped_tool_records_valid,
        "mapped_tool_count_matched": mapped_tool_records_valid
        and len(mapped_tools) >= max(0, expected_tool_count),
        "mapped_tool_parent_ids_matched": (
            mapped_tool_records_valid
            and len(mapped_tools) >= max(0, expected_tool_count)
            and all(
                str(tool.get("parent_observation_id") or "") in generation_id_set
                for tool in mapped_tools
            )
        ),
        "sidecar_trace_owner_matched": sidecar.get("trace_owner")
        == "aieval_app_server_live_event",
    }
    details = {
        "eventRecordCount": len(events),
        "generationObservationIds": generation_ids,
        "mappedToolObservationCount": len(mapped_tools),
        "tokenUsageTotalTokens": total_tokens,
        "turnDurationMs": turn_duration_ms,
        "traceOwner": sidecar.get("trace_owner"),
        "hookStartedSeen": any(event.get("method") == "hook/started" for event in events),
        "hookCompletedSeen": any(event.get("method") == "hook/completed" for event in events),
    }
    return checks, details


def _unwrap_trace_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = payload.get("data")
    if isinstance(data, dict) and data.get("id"):
        return data
    return payload


def _remote_trace_checks(
    *,
    payload: Dict[str, Any],
    mapping: Dict[str, Any],
    expected_tool_count: int,
) -> tuple[Dict[str, bool], Dict[str, Any]]:
    trace = _unwrap_trace_response(payload)
    observations_value = trace.get("observations")
    observations = observations_value if isinstance(observations_value, list) else []
    observation_rows = [item for item in observations if isinstance(item, dict)]
    by_id = {
        str(item.get("id")): item
        for item in observation_rows
        if item.get("id") not in (None, "")
    }

    trace_id = str(mapping.get("trace_id") or "")
    root_observation_id = str(mapping.get("root_observation_id") or "")
    generation_ids, generation_ids_valid = _mapping_generation_ids(mapping)
    mapped_tools, mapped_tool_records_valid = _mapping_tool_records(mapping)
    required_tool_count = max(0, expected_tool_count)

    root_agent = by_id.get(root_observation_id)
    matched_generations = [
        by_id[generation_id]
        for generation_id in generation_ids
        if generation_id in by_id
        and _observation_type(by_id[generation_id]) == "GENERATION"
    ]
    all_generation_observations_matched = (
        generation_ids_valid
        and bool(generation_ids)
        and len(matched_generations) == len(generation_ids)
    )
    confirmed_generation_ids = {
        str(generation.get("id")) for generation in matched_generations
    }
    generation_usage = {
        str(generation.get("id")): _observation_total_tokens(generation)
        for generation in matched_generations
    }
    generation_tokens = {
        generation_id: result[0]
        for generation_id, result in generation_usage.items()
    }
    generation_usage_fields_valid = all(
        result[1] for result in generation_usage.values()
    )
    positive_generation_tokens = [
        total
        for total in generation_tokens.values()
        if total is not None and total > 0
    ]

    agent_latency_seconds = (
        _observation_latency_seconds(root_agent)
        if isinstance(root_agent, dict)
        else 0.0
    )
    trace_latency_seconds = _nonnegative_number(trace.get("latency")) or 0.0
    mapped_tool_ids = [
        str(tool.get("id") or "").strip()
        for tool in mapped_tools
    ]
    remote_mapped_tools = [
        by_id[tool_id]
        for tool_id in mapped_tool_ids
        if tool_id in by_id and _observation_type(by_id[tool_id]) == "TOOL"
    ]
    mapped_tool_count_matched = (
        mapped_tool_records_valid and len(mapped_tools) >= required_tool_count
    )
    mapped_tool_ids_non_empty = mapped_tool_count_matched and all(mapped_tool_ids)
    mapped_tool_ids_unique = (
        mapped_tool_ids_non_empty
        and len(set(mapped_tool_ids)) == len(mapped_tool_ids)
    )
    remote_mapped_tool_count_matched = (
        mapped_tool_count_matched
        and mapped_tool_ids_non_empty
        and mapped_tool_ids_unique
        and len(remote_mapped_tools) == len(mapped_tools)
    )
    remote_tool_parent_match = (
        remote_mapped_tool_count_matched
        and all(
            str(mapping_tool.get("parent_observation_id") or "")
            == _observation_parent_id(by_id[tool_id])
            and _observation_parent_id(by_id[tool_id])
            in confirmed_generation_ids
            for mapping_tool, tool_id in zip(mapped_tools, mapped_tool_ids)
        )
    )

    checks = {
        "trace_id_matched": str(trace.get("id") or "") == trace_id,
        "observations_is_array": isinstance(observations_value, list),
        "root_agent_observation_matched": isinstance(root_agent, dict)
        and _observation_type(root_agent) == "AGENT",
        "generation_ids_valid": generation_ids_valid,
        "all_generation_observations_matched": all_generation_observations_matched,
        "generation_observation_matched": all_generation_observations_matched,
        "generation_usage_fields_valid": generation_usage_fields_valid,
        "generation_usage_total_positive": generation_usage_fields_valid
        and bool(positive_generation_tokens),
        "agent_observation_latency_exceeds_one_second": agent_latency_seconds > 1.0,
        "mapped_tool_records_valid": mapped_tool_records_valid,
        "mapped_tool_count_matched": mapped_tool_count_matched,
        "mapped_tool_ids_non_empty": mapped_tool_ids_non_empty,
        "mapped_tool_ids_unique": mapped_tool_ids_unique,
        "remote_mapped_tool_count_matched": remote_mapped_tool_count_matched,
        "mapped_tool_parent_ids_matched": remote_tool_parent_match,
    }
    details = {
        "traceId": str(trace.get("id") or ""),
        "observationCount": len(observation_rows),
        "matchedGenerationIds": [
            str(generation.get("id")) for generation in matched_generations
        ],
        "generationTotalTokens": generation_tokens,
        "agentLatencySeconds": agent_latency_seconds,
        "traceLatencySeconds": trace_latency_seconds,
        "requiredMappedToolCount": required_tool_count,
        "mappedToolIds": mapped_tool_ids,
        "remoteMappedToolCount": len(remote_mapped_tools),
        "traceTotalCost": trace.get("totalCost")
        if trace.get("totalCost") is not None
        else trace.get("calculatedTotalCost"),
    }
    return checks, details


def _http_status_code(error: BaseException) -> Optional[int]:
    current: Optional[BaseException] = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, urllib.error.HTTPError):
            return int(current.code)
        current = current.__cause__ or current.__context__
    return None


def _is_permanent_http_error(error: BaseException) -> bool:
    status = _http_status_code(error)
    return (
        status is not None
        and 400 <= status < 500
        and status not in {404, 408, 425, 429}
    )


def _poll_remote_trace(
    *,
    api: LangfuseClient,
    trace_id: str,
    mapping: Dict[str, Any],
    expected_tool_count: int,
    timeout_seconds: float,
    poll_interval_seconds: float,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[Dict[str, bool], Dict[str, Any]]:
    deadline = monotonic_fn() + max(0.0, float(timeout_seconds))
    attempts = 0
    last_checks: Dict[str, bool] = {}
    last_details: Dict[str, Any] = {}
    last_error = ""
    encoded_trace_id = urllib.parse.quote(trace_id, safe="")

    while True:
        remaining = deadline - monotonic_fn()
        if remaining <= 0:
            break
        attempts += 1
        original_timeout = api.timeout_seconds
        request_timeout = min(
            remaining,
            max(0.1, min(float(original_timeout), remaining)),
        )
        try:
            api.timeout_seconds = request_timeout
            try:
                payload = api.request("GET", f"/traces/{encoded_trace_id}")
            finally:
                api.timeout_seconds = original_timeout
            last_checks, last_details = _remote_trace_checks(
                payload=payload,
                mapping=mapping,
                expected_tool_count=expected_tool_count,
            )
            last_error = ""
            if _checks_passed(last_checks, REMOTE_REQUIRED_CHECKS):
                break
        except Exception as exc:
            api.timeout_seconds = original_timeout
            if _is_permanent_http_error(exc):
                raise
            last_error = f"{type(exc).__name__}: {exc}"

        remaining = deadline - monotonic_fn()
        if remaining <= 0:
            break
        sleep_fn(min(max(0.1, poll_interval_seconds), remaining))

    last_details["pollAttempts"] = attempts
    last_details["ingestionReady"] = _checks_passed(
        last_checks,
        REMOTE_REQUIRED_CHECKS,
    )
    if last_error:
        last_details["lastApiError"] = last_error
    return last_checks, last_details


def _run_self_test() -> None:
    assert _path_from_raw("") is None
    assert _path_from_raw("   ") is None
    assert _path_from_raw(None) is None
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        fallback_path = temp_path / "run" / "summary.json"
        written_path = _write_summary_with_fallback(
            primary_path=temp_path,
            fallback_path=fallback_path,
            data={"status": "test"},
        )
        assert written_path == fallback_path
        assert fallback_path.is_file()

    assert not _checks_passed({}, REMOTE_REQUIRED_CHECKS)
    assert not _checks_passed({}, LOCAL_REQUIRED_CHECKS)
    assert not _checks_passed(
        {key: True for key in REMOTE_REQUIRED_CHECKS - {"trace_id_matched"}},
        REMOTE_REQUIRED_CHECKS,
    )
    assert not _checks_passed(
        {
            key: True
            for key in LOCAL_REQUIRED_CHECKS - {"mapping_file_created"}
        },
        LOCAL_REQUIRED_CHECKS,
    )
    assert _result_code_after_shutdown(
        result_code=0,
        main_error=None,
        shutdown_error=RuntimeError("shutdown failed"),
    ) == 3
    assert _result_code_after_shutdown(
        result_code=2,
        main_error=RuntimeError("main failed"),
        shutdown_error=RuntimeError("shutdown failed"),
    ) == 2

    mapping = {
        "mapping_source": "codex_app_server_live_event",
        "trace_id": "trace-1",
        "root_observation_id": "agent-1",
        "generation_observation_ids": ["generation-1", "generation-2", "generation-3"],
        "tool_observation_ids": [
            {
                "id": "tool-1",
                "parent_observation_id": "generation-1",
            }
        ],
        "token_usage_total": {"totalTokens": 35},
        "turn_duration_ms": 2500,
    }
    events = [
        {"method": "turn/started", "_received_at": "2026-06-19T00:00:00Z"},
        {
            "method": "thread/tokenUsage/updated",
            "_received_at": "2026-06-19T00:00:01Z",
        },
        {"method": "turn/completed", "_received_at": "2026-06-19T00:00:02.5Z"},
    ]
    local_checks, _ = _local_mapping_checks(
        mapping=mapping,
        sidecar={"trace_owner": "aieval_app_server_live_event"},
        events=events,
        expected_tool_count=1,
        min_turn_duration_ms=1000,
    )
    assert _checks_passed(
        local_checks,
        LOCAL_MAPPING_REQUIRED_CHECKS,
    ), local_checks

    payload = {
        "id": "trace-1",
        "latency": 2.5,
        "observations": [
            {
                "id": "agent-1",
                "type": "AGENT",
                "startTime": "2026-06-19T00:00:00Z",
                "endTime": "2026-06-19T00:00:02.5Z",
            },
            {
                "id": "generation-1",
                "type": "GENERATION",
                "usageDetails": {"input": 10, "output": 2, "total": 12},
            },
            {
                "id": "generation-2",
                "type": "GENERATION",
                "usage": {"total": 23},
            },
            {
                "id": "generation-3",
                "type": "GENERATION",
                "totalTokens": 7,
            },
            {
                "id": "tool-1",
                "type": "TOOL",
                "parentObservationId": "generation-1",
            },
        ],
    }
    remote_checks, details = _remote_trace_checks(
        payload=payload,
        mapping=mapping,
        expected_tool_count=1,
    )
    assert _checks_passed(remote_checks, REMOTE_REQUIRED_CHECKS), remote_checks
    assert details["generationTotalTokens"] == {
        "generation-1": 12.0,
        "generation-2": 23.0,
        "generation-3": 7.0,
    }

    missing_tool_id_mapping = {
        **mapping,
        "tool_observation_ids": [
            {
                "parent_observation_id": "generation-1",
            }
        ],
    }
    missing_tool_id_checks, _ = _remote_trace_checks(
        payload=payload,
        mapping=missing_tool_id_mapping,
        expected_tool_count=1,
    )
    assert not missing_tool_id_checks["mapped_tool_ids_non_empty"]
    assert not missing_tool_id_checks["remote_mapped_tool_count_matched"]
    assert not missing_tool_id_checks["mapped_tool_parent_ids_matched"]

    trace_only_latency_payload = {
        **payload,
        "observations": [
            {
                "id": "agent-1",
                "type": "AGENT",
            },
            *payload["observations"][1:],
        ],
    }
    trace_only_latency_checks, _ = _remote_trace_checks(
        payload=trace_only_latency_payload,
        mapping=mapping,
        expected_tool_count=1,
    )
    assert not trace_only_latency_checks[
        "agent_observation_latency_exceeds_one_second"
    ]

    explicit_agent_latency_payload = {
        **payload,
        "latency": 0.5,
        "observations": [
            {
                "id": "agent-1",
                "type": "AGENT",
                "latency": 1.5,
            },
            *payload["observations"][1:],
        ],
    }
    explicit_agent_latency_checks, _ = _remote_trace_checks(
        payload=explicit_agent_latency_payload,
        mapping=mapping,
        expected_tool_count=1,
    )
    assert explicit_agent_latency_checks[
        "agent_observation_latency_exceeds_one_second"
    ]

    two_tool_payload = {
        **payload,
        "observations": [
            *payload["observations"],
            {
                "id": "tool-2",
                "type": "TOOL",
                "parentObservationId": "generation-2",
            },
        ],
    }
    second_tool_missing_id_mapping = {
        **mapping,
        "tool_observation_ids": [
            mapping["tool_observation_ids"][0],
            {
                "parent_observation_id": "generation-2",
            },
        ],
    }
    second_tool_missing_id_checks, _ = _remote_trace_checks(
        payload=two_tool_payload,
        mapping=second_tool_missing_id_mapping,
        expected_tool_count=1,
    )
    assert not second_tool_missing_id_checks["mapped_tool_ids_non_empty"]
    assert not second_tool_missing_id_checks["mapped_tool_parent_ids_matched"]

    second_tool_wrong_parent_payload = {
        **two_tool_payload,
        "observations": [
            *payload["observations"],
            {
                "id": "tool-2",
                "type": "TOOL",
                "parentObservationId": "agent-1",
            },
        ],
    }
    two_tool_mapping = {
        **mapping,
        "tool_observation_ids": [
            mapping["tool_observation_ids"][0],
            {
                "id": "tool-2",
                "parent_observation_id": "generation-2",
            },
        ],
    }
    second_tool_wrong_parent_checks, _ = _remote_trace_checks(
        payload=second_tool_wrong_parent_payload,
        mapping=two_tool_mapping,
        expected_tool_count=1,
    )
    assert not second_tool_wrong_parent_checks["mapped_tool_parent_ids_matched"]

    two_tool_checks, _ = _remote_trace_checks(
        payload=two_tool_payload,
        mapping=two_tool_mapping,
        expected_tool_count=1,
    )
    assert _checks_passed(
        two_tool_checks,
        REMOTE_REQUIRED_CHECKS,
    ), two_tool_checks

    expected_zero_bad_tool_checks, _ = _remote_trace_checks(
        payload=second_tool_wrong_parent_payload,
        mapping=two_tool_mapping,
        expected_tool_count=0,
    )
    assert not expected_zero_bad_tool_checks["mapped_tool_parent_ids_matched"]

    duplicate_tool_id_mapping = {
        **mapping,
        "tool_observation_ids": [
            mapping["tool_observation_ids"][0],
            {
                "id": "tool-1",
                "parent_observation_id": "generation-1",
            },
        ],
    }
    duplicate_tool_id_checks, _ = _remote_trace_checks(
        payload=payload,
        mapping=duplicate_tool_id_mapping,
        expected_tool_count=0,
    )
    assert not duplicate_tool_id_checks["mapped_tool_ids_unique"]

    malformed_tool_mapping = {
        **mapping,
        "tool_observation_ids": ["malformed"],
    }
    malformed_tool_checks, _ = _remote_trace_checks(
        payload=payload,
        mapping=malformed_tool_mapping,
        expected_tool_count=0,
    )
    assert not malformed_tool_checks["mapped_tool_records_valid"]

    non_list_tool_mapping = {
        **mapping,
        "tool_observation_ids": "malformed",
    }
    non_list_tool_checks, _ = _remote_trace_checks(
        payload=payload,
        mapping=non_list_tool_mapping,
        expected_tool_count=0,
    )
    assert not non_list_tool_checks["mapped_tool_records_valid"]

    remote_other_generation_payload = {
        **payload,
        "observations": [
            *payload["observations"][:-1],
            {
                "id": "tool-1",
                "type": "TOOL",
                "parentObservationId": "generation-2",
            },
        ],
    }
    remote_other_generation_checks, _ = _remote_trace_checks(
        payload=remote_other_generation_payload,
        mapping=mapping,
        expected_tool_count=0,
    )
    assert not remote_other_generation_checks["mapped_tool_parent_ids_matched"]

    ghost_generation_mapping = {
        **mapping,
        "generation_observation_ids": [
            "generation-1",
            "generation-ghost",
        ],
        "tool_observation_ids": [
            {
                "id": "tool-1",
                "parent_observation_id": "generation-ghost",
            }
        ],
    }
    ghost_generation_payload = {
        **payload,
        "observations": [
            *payload["observations"][:-1],
            {
                "id": "tool-1",
                "type": "TOOL",
                "parentObservationId": "generation-ghost",
            },
        ],
    }
    ghost_generation_checks, _ = _remote_trace_checks(
        payload=ghost_generation_payload,
        mapping=ghost_generation_mapping,
        expected_tool_count=1,
    )
    assert not ghost_generation_checks[
        "all_generation_observations_matched"
    ]
    assert not ghost_generation_checks["mapped_tool_parent_ids_matched"]

    for invalid_generation_ids in (
        ["generation-1", ""],
        ["generation-1", "generation-1"],
        "generation-1",
    ):
        invalid_generation_mapping = {
            **mapping,
            "generation_observation_ids": invalid_generation_ids,
        }
        invalid_generation_checks, _ = _remote_trace_checks(
            payload=payload,
            mapping=invalid_generation_mapping,
            expected_tool_count=1,
        )
        assert not invalid_generation_checks["generation_ids_valid"]
        assert not invalid_generation_checks[
            "all_generation_observations_matched"
        ]

    for invalid_tokens in ("12", 12.5, -1, True):
        invalid_usage_payload = {
            **payload,
            "observations": [
                payload["observations"][0],
                {
                    **payload["observations"][1],
                    "usageDetails": {
                        "input": 10,
                        "output": 2,
                        "total": invalid_tokens,
                    },
                },
                *payload["observations"][2:],
            ],
        }
        invalid_usage_checks, _ = _remote_trace_checks(
            payload=invalid_usage_payload,
            mapping=mapping,
            expected_tool_count=1,
        )
        assert not invalid_usage_checks["generation_usage_fields_valid"]
        assert not invalid_usage_checks["generation_usage_total_positive"]

        invalid_local_mapping = {
            **mapping,
            "token_usage_total": {"totalTokens": invalid_tokens},
        }
        invalid_local_usage_checks, _ = _local_mapping_checks(
            mapping=invalid_local_mapping,
            sidecar={"trace_owner": "aieval_app_server_live_event"},
            events=events,
            expected_tool_count=1,
            min_turn_duration_ms=1000,
        )
        assert not invalid_local_usage_checks["token_usage_total_valid"]
        assert not invalid_local_usage_checks["token_usage_total_positive"]

    malformed_usage_container_payload = {
        **payload,
        "observations": [
            payload["observations"][0],
            {
                **payload["observations"][1],
                "usageDetails": "bad",
                "totalTokens": 12,
            },
            *payload["observations"][2:],
        ],
    }
    malformed_usage_container_checks, _ = _remote_trace_checks(
        payload=malformed_usage_container_payload,
        mapping=mapping,
        expected_tool_count=1,
    )
    assert not malformed_usage_container_checks[
        "generation_usage_fields_valid"
    ]
    assert not malformed_usage_container_checks[
        "generation_usage_total_positive"
    ]

    business_io_total, business_io_valid = _observation_total_tokens(
        {
            "usageDetails": {
                "input": 10,
                "output": 2,
                "total": 12,
            },
            "input": "用户问题",
            "output": {"answer": "完成"},
        }
    )
    assert business_io_valid
    assert business_io_total == 12

    invalid_top_level_total, invalid_top_level_valid = (
        _observation_total_tokens(
            {
                "inputTokens": "10",
                "outputTokens": 2,
                "totalTokens": 12,
            }
        )
    )
    assert not invalid_top_level_valid
    assert invalid_top_level_total is None

    for invalid_received_at in (
        "2026-06-19T00:00:00",
        "not-a-datetime",
        "2026-06-19T08:00:00+08:00",
    ):
        invalid_time_events = [
            {**events[0], "_received_at": invalid_received_at},
            *events[1:],
        ]
        invalid_time_checks, _ = _local_mapping_checks(
            mapping=mapping,
            sidecar={"trace_owner": "aieval_app_server_live_event"},
            events=invalid_time_events,
            expected_tool_count=1,
            min_turn_duration_ms=1000,
        )
        assert not invalid_time_checks["event_received_at_utc"]

    mixed_timezone_payload = {
        **payload,
        "observations": [
            {
                **payload["observations"][0],
                "startTime": "2026-06-19T00:00:00",
                "endTime": "2026-06-19T00:00:02.5Z",
            },
            *payload["observations"][1:],
        ],
    }
    mixed_timezone_checks, _ = _remote_trace_checks(
        payload=mixed_timezone_payload,
        mapping=mapping,
        expected_tool_count=1,
    )
    assert not mixed_timezone_checks[
        "agent_observation_latency_exceeds_one_second"
    ]

    non_utc_remote_payload = {
        **payload,
        "observations": [
            {
                **payload["observations"][0],
                "startTime": "2026-06-19T08:00:00+08:00",
                "endTime": "2026-06-19T08:00:02.5+08:00",
            },
            *payload["observations"][1:],
        ],
    }
    non_utc_remote_checks, _ = _remote_trace_checks(
        payload=non_utc_remote_payload,
        mapping=mapping,
        expected_tool_count=1,
    )
    assert non_utc_remote_checks[
        "agent_observation_latency_exceeds_one_second"
    ]

    for invalid_latency in ("1.5", -1, True):
        invalid_latency_payload = {
            **payload,
            "observations": [
                {
                    "id": "agent-1",
                    "type": "AGENT",
                    "latency": invalid_latency,
                },
                *payload["observations"][1:],
            ],
        }
        invalid_latency_checks, _ = _remote_trace_checks(
            payload=invalid_latency_payload,
            mapping=mapping,
            expected_tool_count=1,
        )
        assert not invalid_latency_checks[
            "agent_observation_latency_exceeds_one_second"
        ]

    class FakeApi:
        def __init__(
            self,
            outcomes: list[Any],
            *,
            clock: Optional["FakeClock"] = None,
            budget: Optional[float] = None,
        ) -> None:
            self.timeout_seconds: float = 30
            self.outcomes = list(outcomes)
            self.clock = clock
            self.budget = budget
            self.seen_timeouts: list[float] = []
            self.seen_remaining: list[float] = []
            self.calls = 0

        def request(self, _method: str, _path: str) -> Dict[str, Any]:
            self.calls += 1
            self.seen_timeouts.append(self.timeout_seconds)
            if self.clock is not None and self.budget is not None:
                self.seen_remaining.append(self.budget - self.clock.now)
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0

        def monotonic(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.now += seconds

    def http_failure(status: int) -> RuntimeError:
        error = RuntimeError(f"HTTP {status}")
        error.__cause__ = urllib.error.HTTPError(
            url="https://langfuse.example/api/public/traces/trace-1",
            code=status,
            msg="test",
            hdrs=None,
            fp=None,
        )
        return error

    for permanent_status in (400, 401, 403):
        permanent_api = FakeApi([http_failure(permanent_status), payload])
        try:
            _poll_remote_trace(
                api=permanent_api,  # type: ignore[arg-type]
                trace_id="trace-1",
                mapping=mapping,
                expected_tool_count=1,
                timeout_seconds=2,
                poll_interval_seconds=0.01,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"{permanent_status} 必须立即抛出")
        assert permanent_api.calls == 1
        assert permanent_api.timeout_seconds == 30

    retry_api = FakeApi([http_failure(404), payload])
    retry_checks, retry_details = _poll_remote_trace(
        api=retry_api,  # type: ignore[arg-type]
        trace_id="trace-1",
        mapping=mapping,
        expected_tool_count=1,
        timeout_seconds=2,
        poll_interval_seconds=0.01,
    )
    assert _checks_passed(retry_checks, REMOTE_REQUIRED_CHECKS)
    assert retry_details["pollAttempts"] == 2
    assert retry_api.calls == 2
    assert all(0 < timeout <= 2 for timeout in retry_api.seen_timeouts)
    assert retry_api.timeout_seconds == 30

    for retry_status in (408, 425):
        fake_clock = FakeClock()
        deadline_api = FakeApi(
            [http_failure(retry_status), payload],
            clock=fake_clock,
            budget=0.25,
        )
        deadline_checks, _ = _poll_remote_trace(
            api=deadline_api,  # type: ignore[arg-type]
            trace_id="trace-1",
            mapping=mapping,
            expected_tool_count=1,
            timeout_seconds=0.25,
            poll_interval_seconds=0.05,
            monotonic_fn=fake_clock.monotonic,
            sleep_fn=fake_clock.sleep,
        )
        assert _checks_passed(deadline_checks, REMOTE_REQUIRED_CHECKS)
        assert deadline_api.calls == 2
        assert all(
            timeout <= remaining + 1e-9
            for timeout, remaining in zip(
                deadline_api.seen_timeouts,
                deadline_api.seen_remaining,
            )
        )
        assert deadline_api.timeout_seconds == 30

    zero_budget_clock = FakeClock()
    zero_budget_api = FakeApi([])
    zero_budget_checks, zero_budget_details = _poll_remote_trace(
        api=zero_budget_api,  # type: ignore[arg-type]
        trace_id="trace-1",
        mapping=mapping,
        expected_tool_count=1,
        timeout_seconds=0,
        poll_interval_seconds=0.01,
        monotonic_fn=zero_budget_clock.monotonic,
        sleep_fn=zero_budget_clock.sleep,
    )
    assert zero_budget_api.calls == 0
    assert zero_budget_checks == {}
    assert zero_budget_details["pollAttempts"] == 0


def parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="验证 runner 内 app-server live trace、mapping 与 Langfuse 远端 observation"
    )
    parser.add_argument("--run-id", default="", help="eval run id；默认使用 live trace canary 时间戳")
    parser.add_argument("--case-id", default="live-trace-canary", help="case id")
    parser.add_argument("--cwd", default=str(PROJECT_ROOT), help="Codex working directory")
    parser.add_argument("--model", default="", help="可选模型覆盖")
    parser.add_argument("--sandbox", default="danger-full-access", help="Codex sandbox mode")
    parser.add_argument(
        "--prompt",
        default="Reply exactly: app-server live trace canary ok",
        help="发送给 app-server turn 的简单提示词",
    )
    parser.add_argument(
        "--expected-tool-count",
        type=int,
        default=0,
        help="要求的最少 Tool observation 数；默认简单提示词不要求工具",
    )
    parser.add_argument(
        "--mapping-timeout-seconds",
        type=int,
        default=120,
        help="兼容旧调用保留；live trace mapping 在 runner 内完成，不再等待 Stop hook",
    )
    parser.add_argument("--turn-timeout-seconds", type=int, default=240, help="app-server turn 超时")
    parser.add_argument("--request-timeout-seconds", type=int, default=90, help="app-server 请求超时")
    parser.add_argument(
        "--langfuse-ingestion-timeout-seconds",
        type=int,
        default=60,
        help="轮询 Langfuse Public API 等待 trace ingestion 的最长秒数",
    )
    parser.add_argument(
        "--langfuse-poll-interval-seconds",
        type=float,
        default=2.0,
        help="Langfuse Public API 轮询间隔秒数",
    )
    parser.add_argument(
        "--min-turn-duration-ms",
        type=int,
        default=1000,
        help="本地 mapping 要求的最小 turn duration；默认必须大于 1000ms",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="只运行无网络解析 fixture",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        _run_self_test()
        print(json.dumps({"selfTest": True, "status": "passed"}, ensure_ascii=False))
        return 0

    eval_run_id = args.run_id or f"app-server-live-trace-{int(time.time())}"
    config = load_config(
        [
            "--run-id",
            eval_run_id,
            "--codex-cwd",
            str(Path(args.cwd).resolve()),
            "--codex-sandbox",
            args.sandbox,
            "--app-server-request-timeout-seconds",
            str(args.request_timeout_seconds),
            "--app-server-turn-timeout-seconds",
            str(args.turn_timeout_seconds),
            "--mapping-timeout-seconds",
            str(args.mapping_timeout_seconds),
        ]
        + (["--codex-model", args.model] if args.model else [])
    )
    ctx = EvalRunContext(
        eval_run_id=config.run_id,
        dataset_name=config.dataset,
        agent_version=config.agent_version,
        client=config.client,
        skill_version=config.skill_version,
        db_snapshot_id=config.db_snapshot_id,
    )
    case = EvalCase(
        dataset_name=config.dataset,
        dataset_item_id=f"canary-{args.case_id}",
        case_id=args.case_id,
        original_request=args.prompt,
        expected_output={
            "case_id": args.case_id,
            "main_query": "app-server live trace canary",
            "evaluation_focus": ["live trace mapping"],
        },
        tool_contract={"expected_tools": [], "tool_match_mode": "any"},
    )

    default_output_dir = PROJECT_ROOT / ".aieval" / "runs" / ctx.eval_run_id
    default_summary_path = (
        default_output_dir / "app_server_live_trace_canary_summary.json"
    )
    summary_path = default_summary_path
    summary: Dict[str, Any] = {
        "evalRunId": ctx.eval_run_id,
        "caseId": case.case_id,
        "localChecks": {},
        "remoteChecks": {},
        "shutdownSucceeded": True,
    }
    result_code = 2
    main_error: Optional[Exception] = None

    try:
        execution = run_codex_app_server(case, ctx, config)
        output = execution.output if isinstance(execution.output, dict) else {}
        event_path = _path_from_raw(output.get("event_path"))
        mapping_path = _path_from_raw(output.get("mapping_path"))
        sidecar_path = _path_from_raw(output.get("sidecar_path"))
        event_file_created = event_path is not None and event_path.is_file()
        mapping_file_created = mapping_path is not None and mapping_path.is_file()
        sidecar_file_created = sidecar_path is not None and sidecar_path.is_file()
        if mapping_file_created and mapping_path is not None:
            summary_path = (
                mapping_path.parent / "app_server_live_trace_canary_summary.json"
            )

        events = read_jsonl(event_path)
        mapping_records = (
            read_mapping_records(mapping_path)
            if mapping_file_created and mapping_path is not None
            else []
        )
        matching_mapping_records = [
            record
            for record in mapping_records
            if record.get("eval_run_id") == ctx.eval_run_id
            and record.get("case_id") == case.case_id
        ]
        mapping = matching_mapping_records[-1] if matching_mapping_records else {}

        sidecar_records = [
            record
            for record in read_jsonl(sidecar_path)
            if record.get("eval_run_id") == ctx.eval_run_id
            and record.get("case_id") == case.case_id
            and record.get("thread_id") == output.get("thread_id")
            and record.get("turn_id") == output.get("turn_id")
        ]
        sidecar = sidecar_records[-1] if sidecar_records else {}

        local_checks, local_details = _local_mapping_checks(
            mapping=mapping,
            sidecar=sidecar,
            events=events,
            expected_tool_count=args.expected_tool_count,
            min_turn_duration_ms=args.min_turn_duration_ms,
        )
        local_checks["event_file_created"] = event_file_created
        local_checks["mapping_file_created"] = mapping_file_created
        local_checks["sidecar_file_created"] = sidecar_file_created
        local_checks["mapping_record_matched"] = bool(matching_mapping_records)
        local_checks["sidecar_record_matched"] = bool(sidecar_records)
        local_checks["mapping_turn_key_matched"] = bool(
            output.get("turn_id")
            and str(output.get("turn_id")) in str(mapping.get("turn_key") or "")
        )

        api = api_from_config(config)
        if api is None:
            raise RuntimeError("真实 live trace canary 需要启用 Langfuse 并配置 Public/Secret Key")
        trace_id = str(mapping.get("trace_id") or "")
        if not trace_id:
            raise RuntimeError("live trace mapping 缺少 trace_id，无法回读 Langfuse")
        remote_checks, remote_details = _poll_remote_trace(
            api=api,
            trace_id=trace_id,
            mapping=mapping,
            expected_tool_count=args.expected_tool_count,
            timeout_seconds=args.langfuse_ingestion_timeout_seconds,
            poll_interval_seconds=args.langfuse_poll_interval_seconds,
        )

        summary.update(
            {
                "threadId": output.get("thread_id"),
                "turnId": output.get("turn_id"),
                "traceId": trace_id,
                "eventPath": str(event_path) if event_path is not None else "",
                "mappingPath": str(mapping_path) if mapping_path is not None else "",
                "sidecarPath": str(sidecar_path) if sidecar_path is not None else "",
                "localChecks": local_checks,
                "localDetails": local_details,
                "remoteChecks": remote_checks,
                "remoteDetails": remote_details,
            }
        )
        result_code = (
            0
            if _checks_passed(local_checks, LOCAL_REQUIRED_CHECKS)
            and _checks_passed(remote_checks, REMOTE_REQUIRED_CHECKS)
            else 4
        )
    except Exception as exc:
        main_error = exc
        summary["mainError"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        result_code = 2
    finally:
        shutdown_error: Optional[Exception] = None
        try:
            shutdown_codex_app_server()
        except Exception as exc:
            shutdown_error = exc
            summary["shutdownSucceeded"] = False
            summary["shutdownError"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        result_code = _result_code_after_shutdown(
            result_code=result_code,
            main_error=main_error,
            shutdown_error=shutdown_error,
        )

    summary["status"] = "passed" if result_code == 0 else "failed"
    written_summary_path = _write_summary_with_fallback(
        primary_path=summary_path,
        fallback_path=default_summary_path,
        data=summary,
    )
    summary["summaryPath"] = str(written_summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return result_code


if __name__ == "__main__":
    raise SystemExit(main())
