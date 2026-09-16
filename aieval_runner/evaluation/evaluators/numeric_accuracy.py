from __future__ import annotations

from typing import List, Optional, Tuple

from aieval_runner.evaluation.events import (
    extract_data_table,
    extract_schema_table,
    is_metric_data_tool_call,
    is_sql_execution_tool_call,
    is_successful_tool_call,
    tool_identity,
)
from aieval_runner.evaluation.evaluators.common import JsonObject
from aieval_runner.core.models import EvaluatorResult
from aieval_runner.evaluation.tables.scoring import score_tables


SCHEMA_ONLY_SCORE_MODE = "schema_only"


def _target_result_table_candidates(
    calls: List[JsonObject],
    *,
    use_schema_table: bool = False,
) -> Tuple[List[Tuple[JsonObject, JsonObject]], Optional[JsonObject], Optional[str]]:
    target_calls = [
        call
        for call in calls
        if is_metric_data_tool_call(call) or is_sql_execution_tool_call(call)
    ]
    if not target_calls:
        return [], None, "missing_target_result"

    latest_error: Optional[str] = None
    candidates: List[Tuple[JsonObject, JsonObject]] = []
    for call in target_calls:
        if not is_successful_tool_call(call):
            latest_error = "target_call_failed"
            continue
        table = extract_data_table(call)
        if table is None and use_schema_table:
            table = extract_schema_table(call)
        if table is None:
            latest_error = str(
                call.get("_data_table_error") or "target_result_unparseable"
            )
            continue
        candidates.append((call, table))
    return candidates, target_calls[-1], latest_error


def _best_scored_result_table_call(
    calls: List[JsonObject],
    expected_result: JsonObject,
    *,
    score_mode: str = "",
) -> Tuple[Optional[JsonObject], Optional[JsonObject], Optional[JsonObject], int, Optional[str]]:
    candidates, latest_target, result_error = _target_result_table_candidates(
        calls,
        use_schema_table=score_mode == SCHEMA_ONLY_SCORE_MODE,
    )
    if not candidates:
        return latest_target, None, None, 0, result_error

    best: Optional[Tuple[int, JsonObject, JsonObject, JsonObject]] = None
    for index, (call, table) in enumerate(candidates):
        detail = (
            score_table_schema(expected_result, table)
            if score_mode == SCHEMA_ONLY_SCORE_MODE
            else score_tables(expected_result, table)
        )
        if best is None:
            best = (index, call, table, detail)
            continue
        best_index, _best_call, _best_table, best_detail = best
        current_key = (
            detail["score"],
            bool(detail["row_count_match"]),
            detail["matched_column_count"],
            -detail["extra_column_count"],
            index,
        )
        best_key = (
            best_detail["score"],
            bool(best_detail["row_count_match"]),
            best_detail["matched_column_count"],
            -best_detail["extra_column_count"],
            best_index,
        )
        if current_key > best_key:
            best = (index, call, table, detail)

    assert best is not None
    _index, selected, table, detail = best
    return selected, table, detail, len(candidates), None


def score_table_schema(expected: JsonObject, predicted: JsonObject) -> JsonObject:
    expected_columns = [str(column) for column in expected.get("columns") or []]
    predicted_columns = [str(column) for column in predicted.get("columns") or []]
    expected_set = set(expected_columns)
    predicted_set = set(predicted_columns)
    matched = len(expected_set & predicted_set)
    expected_count = len(expected_set)
    predicted_count = len(predicted_set)
    extra = max(0, predicted_count - matched)
    recall = matched / expected_count if expected_count else 0.0
    penalty = 0.01 * (extra / predicted_count) if predicted_count else 0.0
    score = max(0.0, recall - penalty)
    return {
        "score": round(score, 6),
        "recall": round(recall, 6),
        "penalty": round(penalty, 6),
        "matched_column_count": matched,
        "expected_column_count": expected_count,
        "predicted_column_count": predicted_count,
        "expected_row_count": len(expected.get("rows") or []),
        "predicted_row_count": len(predicted.get("rows") or []),
        "row_count_match": len(expected.get("rows") or []) == len(predicted.get("rows") or []),
        "extra_column_count": extra,
    }


def evaluate_numeric_accuracy(
    calls: List[JsonObject],
    expected_result: Optional[JsonObject],
    *,
    expected_contract: Optional[JsonObject] = None,
    event_error: Optional[str] = None,
) -> EvaluatorResult:
    if expected_result is None:
        return EvaluatorResult(
            value=0.0,
            data_type="NUMERIC",
            comment="DatasetItem.expectedOutput.expected_result 缺失，无法计算数值准确性。",
            metadata={"reason": "missing_expected_result"},
        )
    contract = expected_contract if isinstance(expected_contract, dict) else {}
    expected_specs = [
        {
            "name": "primary",
            "expected_result": expected_result,
            "score_mode": str(contract.get("score_mode") or ""),
        }
    ]
    alternatives = contract.get("alternative_expected_results")
    if isinstance(alternatives, list):
        for index, alternative in enumerate(alternatives):
            if not isinstance(alternative, dict):
                continue
            alternative_result = alternative.get("expected_result")
            if not isinstance(alternative_result, dict):
                continue
            expected_specs.append(
                {
                    "name": str(alternative.get("name") or f"alternative_{index + 1}"),
                    "expected_result": alternative_result,
                    "score_mode": str(
                        alternative.get("score_mode")
                        or contract.get("score_mode")
                        or ""
                    ),
                }
            )

    best_result = None
    for spec in expected_specs:
        selected, predicted, detail, candidate_count, result_error = (
            _best_scored_result_table_call(
                calls,
                spec["expected_result"],
                score_mode=spec["score_mode"],
            )
        )
        if selected is None or predicted is None or detail is None:
            current = (selected, predicted, detail, candidate_count, result_error, spec)
        else:
            current = (selected, predicted, detail, candidate_count, result_error, spec)
        if best_result is None:
            best_result = current
            continue
        _best_selected, _best_predicted, best_detail, _best_count, _best_error, _best_spec = best_result
        if detail is None:
            continue
        if best_detail is None:
            best_result = current
            continue
        current_key = (
            detail["score"],
            bool(detail["row_count_match"]),
            detail["matched_column_count"],
            -detail["extra_column_count"],
        )
        best_key = (
            best_detail["score"],
            bool(best_detail["row_count_match"]),
            best_detail["matched_column_count"],
            -best_detail["extra_column_count"],
        )
        if current_key > best_key:
            best_result = current

    assert best_result is not None
    selected, predicted, detail, candidate_count, result_error, selected_spec = best_result
    score_mode = str(selected_spec["score_mode"])
    if selected is None or predicted is None or detail is None:
        reason = event_error or result_error or "missing_predicted_result"
        return EvaluatorResult(
            value=0.0,
            data_type="NUMERIC",
            comment=f"无法取得目标工具的结构化表格: {reason}。",
            metadata={"reason": reason},
        )

    if score_mode == SCHEMA_ONLY_SCORE_MODE:
        comment = (
            f"字段集合匹配 {detail['matched_column_count']}/"
            f"{detail['expected_column_count']}，多余列 {detail['extra_column_count']}。"
        )
    else:
        comment = (
            f"列内容匹配 {detail['matched_column_count']}/"
            f"{detail['expected_column_count']}，多余列 {detail['extra_column_count']}。"
        )

    return EvaluatorResult(
        value=detail["score"],
        data_type="NUMERIC",
        comment=comment,
        metadata={
            "selected_tool": tool_identity(selected),
            "candidate_result_count": candidate_count,
            "selection_strategy": "best_matching_target_result",
            "expected_result_variant": selected_spec["name"],
            "score_mode": score_mode or "value_column_signature",
            **detail,
        },
    )
