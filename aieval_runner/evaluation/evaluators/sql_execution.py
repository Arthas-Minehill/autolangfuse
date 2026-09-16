from __future__ import annotations

from typing import List, Optional

from aieval_runner.evaluation.events import (
    is_sql_execution_tool_call,
    is_successful_tool_call,
    tool_identity,
)
from aieval_runner.evaluation.evaluators.common import JsonObject
from aieval_runner.core.models import EvaluatorResult


def evaluate_sql_execution_success(
    calls: List[JsonObject],
    *,
    event_error: Optional[str] = None,
    successful_sql_tool_count: Optional[int] = None,
) -> EvaluatorResult:
    successful = [
        call
        for call in calls
        if is_sql_execution_tool_call(call) and is_successful_tool_call(call)
    ]
    count = successful_sql_tool_count
    if count is None:
        count = len(successful)
    if count > 0:
        selected = successful[-1] if successful else {}
        return EvaluatorResult(
            value=True,
            data_type="BOOLEAN",
            comment=(
                f"检测到成功 SQL 类工具调用: {tool_identity(selected)}。"
                if selected
                else "检测到成功 SQL 类工具调用。"
            ),
            metadata={
                "selected_tool": tool_identity(selected) if selected else None,
                "successful_sql_tool_count": count,
            },
        )
    reason = event_error or "未检测到成功完成且无 error 的当前受支持数据查询工具调用"
    return EvaluatorResult(
        value=False,
        data_type="BOOLEAN",
        comment=reason,
        metadata={"successful_sql_tool_count": 0},
    )
