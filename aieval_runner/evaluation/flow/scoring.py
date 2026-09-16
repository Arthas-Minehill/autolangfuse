from __future__ import annotations

from typing import Any, Dict, List

from aieval_runner.core.constants import SCHEMA_VERSION
from aieval_runner.evaluation.evaluators.base import EvaluationInput
from aieval_runner.evaluation.evaluators.event_loading import load_execution_events
from aieval_runner.evaluation.evaluators.registry import selected_deterministic_evaluators
from aieval_runner.core.models import (
    AgentExecution,
    EvalCase,
    EvalRunContext,
    EvalScore,
    EvaluatorResult,
)


TURN_LEVEL_SCORE_NAMES = {"expected_tools_match", "tool_result_status"}


def base_score_metadata(
    case: EvalCase,
    execution: AgentExecution,
    ctx: EvalRunContext,
) -> Dict[str, Any]:
    root_observation_id = execution.mapping.get("root_observation_id")
    target_observation_id = execution.observation_id
    return {
        "schema_version": SCHEMA_VERSION,
        "eval_run_id": ctx.eval_run_id,
        "case_id": case.case_id,
        "dataset_name": ctx.dataset_name,
        "dataset_item_id": case.dataset_item_id,
        "trace_id": execution.trace_id,
        "observation_id": execution.observation_id,
        "root_observation_id": str(root_observation_id) if root_observation_id else None,
        "target_observation_id": target_observation_id,
        "target_observation_missing": bool(
            root_observation_id and target_observation_id == str(root_observation_id)
        ),
        "agent_version": ctx.agent_version,
        "client": ctx.client,
        "skill_version": ctx.skill_version,
        "db_snapshot_id": ctx.db_snapshot_id,
    }


def _to_score(
    name: str,
    result: EvaluatorResult,
    *,
    base_metadata: Dict[str, Any],
) -> EvalScore:
    return EvalScore(
        name=name,
        value=result.value,
        data_type=result.data_type,
        comment=result.comment,
        metadata={**base_metadata, **result.metadata},
    )


def evaluate_case(
    case: EvalCase,
    execution: AgentExecution,
    ctx: EvalRunContext,
) -> List[EvalScore]:
    meta = base_score_metadata(case, execution, ctx)
    events = load_execution_events(execution)
    data = EvaluationInput(case=case, execution=execution, ctx=ctx, events=events)
    evaluators = selected_deterministic_evaluators(data)
    meta = {
        **meta,
        "deterministic_evaluators": [evaluator.name for evaluator in evaluators],
    }
    return [
        _to_score(evaluator.name, evaluator.evaluate(data), base_metadata=meta)
        for evaluator in evaluators
    ]


def score_observation_id(score: EvalScore, execution: AgentExecution) -> str | None:
    root_observation_id = execution.mapping.get("root_observation_id")
    return str(root_observation_id) if root_observation_id else None
