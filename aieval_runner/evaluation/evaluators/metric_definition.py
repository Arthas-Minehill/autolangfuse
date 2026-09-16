from __future__ import annotations

from typing import Any, List, Optional

from aieval_runner.evaluation.events import (
    extract_data_table,
    extract_metric_name,
    extract_schema_table,
    is_metric_data_tool_call,
    is_metric_definition_tool_call,
    is_sql_execution_tool_call,
    is_successful_tool_call,
    last_matching_call,
    tool_identity,
)
from aieval_runner.evaluation.evaluators.common import JsonObject, safe_text
from aieval_runner.core.models import EvaluatorResult

def evaluate_metric_definition_consistency(
    calls: List[JsonObject],
    main_query: str,
    *,
    expected_contract: Optional[JsonObject] = None,
    expected_result: Any = None,
    event_error: Optional[str] = None,
) -> EvaluatorResult:
    expected = safe_text(main_query)
    for selected in reversed(calls):
        actual_raw = extract_metric_name(selected)
        if (
            actual_raw == main_query
            and is_metric_data_tool_call(selected)
            and is_successful_tool_call(selected)
            and (
                extract_data_table(selected) is not None
                or extract_schema_table(selected) is not None
            )
        ):
            return EvaluatorResult(
                value=True,
                data_type="BOOLEAN",
                comment="成功取数的指标应用与 main_query 一致。",
                metadata={
                    "selected_tool": tool_identity(selected),
                    "expected_metric": expected,
                    "actual_metric": safe_text(actual_raw),
                    "selection_strategy": "matched_metric_data_result",
                },
            )

    contract = expected_contract if isinstance(expected_contract, dict) else {}
    if str(contract.get("score_mode") or "") == "schema_only":
        expected_columns = (
            expected_result.get("columns")
            if isinstance(expected_result, dict)
            else None
        )
        expected_column_set = {
            str(column).strip().lower()
            for column in expected_columns or []
            if str(column).strip()
        }
        selected = last_matching_call(
            calls,
            lambda call: is_sql_execution_tool_call(call)
            and (
                extract_data_table(call) is not None
                or extract_schema_table(call) is not None
            ),
            successful_only=True,
        )
        if selected is not None:
            table = extract_data_table(selected) or extract_schema_table(selected) or {}
            actual_columns = table.get("columns") if isinstance(table, dict) else []
            actual_column_set = {
                str(column).strip().lower()
                for column in actual_columns or []
                if str(column).strip()
            }
            if not expected_column_set or not expected_column_set.issubset(actual_column_set):
                return EvaluatorResult(
                    value=False,
                    data_type="BOOLEAN",
                    comment="schema_only 查询返回字段与 Gold 字段集合不一致。",
                    metadata={
                        "selected_tool": tool_identity(selected),
                        "expected_columns": sorted(expected_column_set),
                        "actual_columns": sorted(actual_column_set),
                        "selection_strategy": "schema_only_column_mismatch",
                    },
                )
            return EvaluatorResult(
                value=True,
                data_type="BOOLEAN",
                comment="schema_only 契约下检测到成功 SQL 投影，可用于验证目标字段集合。",
                metadata={
                    "selected_tool": tool_identity(selected),
                    "expected_metric": expected,
                    "actual_metric": "SQL projection",
                    "selection_strategy": "schema_only_sql_projection",
                },
            )

    selected = last_matching_call(
        calls,
        lambda call: is_metric_definition_tool_call(call)
        and extract_metric_name(call) is not None,
    )
    if selected is None:
        return EvaluatorResult(
            value=False,
            data_type="BOOLEAN",
            comment=event_error or "未检测到指标查询或指标 SQL 工具调用。",
            metadata={"expected_metric": expected},
        )
    actual_raw = extract_metric_name(selected)
    matched = actual_raw == main_query
    actual = safe_text(actual_raw)
    return EvaluatorResult(
        value=matched,
        data_type="BOOLEAN",
        comment=(
            "最后一个可识别名称的指标与 main_query 一致。"
            if matched
            else "最后一个可识别名称的指标与 main_query 不一致。"
        ),
        metadata={
            "selected_tool": tool_identity(selected),
            "expected_metric": expected,
            "actual_metric": actual,
            "selection_strategy": "last_named_metric_definition",
        },
    )
