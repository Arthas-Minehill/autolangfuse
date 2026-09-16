from __future__ import annotations

from typing import Tuple

from aieval_runner.evaluation.evaluators.base import (
    CaseEvaluator,
    EvaluationInput,
    FunctionCaseEvaluator,
)
from aieval_runner.evaluation.evaluators.tool_contract import (
    evaluate_expected_tools_match,
    evaluate_tool_result_status,
)


DETERMINISTIC_EVALUATORS: Tuple[CaseEvaluator, ...] = (
    FunctionCaseEvaluator("expected_tools_match", evaluate_expected_tools_match),
    FunctionCaseEvaluator("tool_result_status", evaluate_tool_result_status),
)


def selected_deterministic_evaluators(
    _data: EvaluationInput,
) -> Tuple[CaseEvaluator, ...]:
    return DETERMINISTIC_EVALUATORS
