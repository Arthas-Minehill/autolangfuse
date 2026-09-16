from __future__ import annotations

from typing import List, Optional

from aieval_runner.evaluation.events import (
    is_sql_execution_tool_call,
    is_successful_tool_call,
    tool_identity,
)
from aieval_runner.evaluation.evaluators.common import JsonObject, safe_text
from aieval_runner.core.models import AgentExecution, EvaluatorResult


def _event_error_recovered_by_successful_sql(
    event_error: Optional[str],
    *,
    successful_sql_count: int,
) -> bool:
    if not event_error or successful_sql_count <= 0:
        return False
    parts = [part.strip() for part in event_error.split(";") if part.strip()]
    return bool(parts) and all(part == "non_mcp_execution_error" for part in parts)


def evaluate_task_completed(
    execution: AgentExecution,
    calls: List[JsonObject],
    *,
    event_error: Optional[str] = None,
    failed_tool_count: Optional[int] = None,
    failed_tool_names: Optional[List[str]] = None,
    successful_tool_count: Optional[int] = None,
    allow_auxiliary_tool_recovery: bool = False,
) -> EvaluatorResult:
    output = execution.output if isinstance(execution.output, dict) else {}
    completed = output.get("turn_completed")
    turn = completed.get("turn") if isinstance(completed, dict) else None
    turn_status = turn.get("status") if isinstance(turn, dict) else None
    turn_error = safe_text(turn.get("error")) if isinstance(turn, dict) else None
    answer = output.get("answer")
    derived_failed_tools = [
        tool_identity(call)
        for call in calls
        if not is_successful_tool_call(call)
    ]
    failed_tools = failed_tool_names if failed_tool_names is not None else derived_failed_tools
    failure_count = (
        failed_tool_count
        if failed_tool_count is not None
        else len(derived_failed_tools)
    )
    successful_sql_count = sum(
        1
        for call in calls
        if is_sql_execution_tool_call(call) and is_successful_tool_call(call)
    )
    derived_successful_tool_count = sum(
        1 for call in calls if is_successful_tool_call(call)
    )
    total_successful_tool_count = (
        successful_tool_count
        if successful_tool_count is not None
        else derived_successful_tool_count
    )
    answer_present = isinstance(answer, str) and bool(answer.strip())
    recovered_auxiliary_tools = bool(
        allow_auxiliary_tool_recovery
        and failure_count
        and successful_sql_count == 0
        and total_successful_tool_count > 0
        and answer_present
    )
    reasons: List[str] = []
    recovered_event_error = _event_error_recovered_by_successful_sql(
        event_error,
        successful_sql_count=successful_sql_count,
    )
    if event_error and not recovered_event_error:
        reasons.append(event_error)
    if turn_status != "completed":
        reasons.append(f"turn 状态不是 completed: {turn_status!r}")
    if turn_error:
        reasons.append("turn_error")
    if failure_count and successful_sql_count == 0 and not recovered_auxiliary_tools:
        reasons.append(f"failed_tool_count={failure_count}")
    if not answer_present:
        reasons.append("最终答案为空")

    return EvaluatorResult(
        value=not reasons,
        data_type="BOOLEAN",
        comment="任务链路正常完成且最终答案非空。" if not reasons else "; ".join(reasons),
        metadata={
            "turn_status": turn_status,
            "failed_tool_count": failure_count,
            "failed_tool_names": failed_tools[:10],
            "failed_tools_recovered_by_successful_sql": bool(
                failure_count and successful_sql_count > 0
            ),
            "failed_auxiliary_tools_recovered_by_answer": recovered_auxiliary_tools,
            "event_error_recovered_by_successful_sql": recovered_event_error,
            "successful_sql_tool_count": successful_sql_count,
            "successful_tool_count": total_successful_tool_count,
            "answer_present": answer_present,
        },
    )
