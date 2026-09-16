from __future__ import annotations

from typing import Any, Dict, List

from aieval_runner.core.models import EvaluatorResult
from aieval_runner.evaluation.events import (
    canonical_tool_name,
    tool_name_matches,
)
from aieval_runner.evaluation.evaluators.base import EvaluationInput


JsonObject = Dict[str, Any]
MAX_METADATA_TOOL_SAMPLES = 20
MAX_MATCH_SAMPLES_PER_EXPECTED_TOOL = 5


EXPECTED_LIFECYCLE_STATUS = "completed"


def _contract(data: EvaluationInput) -> tuple[List[str], str]:
    expected = data.case.tool_contract
    return (
        list(expected["expected_tools"]),
        str(expected["tool_match_mode"]),
    )


def _matches(calls: List[JsonObject], expected_tools: List[str]) -> Dict[str, List[JsonObject]]:
    return {
        expected: [call for call in calls if tool_name_matches(expected, call)]
        for expected in expected_tools
    }


def _aggregate(flags: List[bool], mode: str) -> bool:
    return any(flags) if mode == "any" else all(flags)


def evaluate_expected_tools_match(data: EvaluationInput) -> EvaluatorResult:
    expected_tools, mode = _contract(data)
    matches = _matches(data.calls, expected_tools)
    matched_flags = [bool(matches[expected]) for expected in expected_tools]
    missing_tools = [expected for expected in expected_tools if not matches[expected]]
    value = _aggregate(matched_flags, mode)
    actual_tools = [canonical_tool_name(call) for call in data.calls]
    return EvaluatorResult(
        value=value,
        data_type="BOOLEAN",
        comment=(
            f"工具调用满足 {mode} 匹配模式。"
            if value
            else f"工具调用不满足 {mode} 匹配模式。"
        ),
        metadata={
            "tool_match_mode": mode,
            "expected_tools": expected_tools,
            "actual_tool_count": len(actual_tools),
            "actual_tools": actual_tools[:MAX_METADATA_TOOL_SAMPLES],
            "actual_tools_truncated": len(actual_tools) > MAX_METADATA_TOOL_SAMPLES,
            "matched_tools": {
                expected: [
                    canonical_tool_name(call)
                    for call in calls[:MAX_MATCH_SAMPLES_PER_EXPECTED_TOOL]
                ]
                for expected, calls in matches.items()
            },
            "matched_tool_counts": {
                expected: len(calls) for expected, calls in matches.items()
            },
            "missing_tools": missing_tools,
            "event_error": data.events.error,
        },
    )


def evaluate_tool_result_status(data: EvaluationInput) -> EvaluatorResult:
    expected_tools, mode = _contract(data)
    matches = _matches(data.calls, expected_tools)
    observed_statuses: Dict[str, List[str]] = {}
    successful_tools: List[str] = []
    flags: List[bool] = []
    for expected in expected_tools:
        calls = matches[expected]
        statuses: List[str] = []
        passed = False
        for call in calls:
            lifecycle_status = str(call.get("status") or "").strip().lower()
            if lifecycle_status and lifecycle_status not in statuses:
                statuses.append(lifecycle_status)
            if (
                call.get("_completed_event") is True
                and lifecycle_status == EXPECTED_LIFECYCLE_STATUS
            ):
                passed = True
        observed_statuses[expected] = statuses
        if passed:
            successful_tools.append(expected)
        flags.append(passed)
    value = _aggregate(flags, mode)
    return EvaluatorResult(
        value=value,
        data_type="BOOLEAN",
        comment=(
            f"工具结果状态满足 {mode} 模式，生命周期状态为 {EXPECTED_LIFECYCLE_STATUS}。"
            if value
            else f"工具结果状态不满足 {mode} 模式，生命周期状态必须为 {EXPECTED_LIFECYCLE_STATUS}。"
        ),
        metadata={
            "tool_match_mode": mode,
            "expected_lifecycle_status": EXPECTED_LIFECYCLE_STATUS,
            "expected_tools": expected_tools,
            "successful_status_tools": successful_tools,
            "observed_statuses": observed_statuses,
            "event_error": data.events.error,
        },
    )
