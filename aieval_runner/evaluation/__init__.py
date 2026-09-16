from __future__ import annotations

from typing import Any, Dict, List

from aieval_runner.core.models import (
    AgentExecution,
    EvalCase,
    EvalRunContext,
    EvalScore,
    EvaluatorResult,
    RunnerConfig,
)


def run_agent_for_case(*args: Any, **kwargs: Any) -> AgentExecution:
    from aieval_runner.agent import run_agent_for_case as run

    return run(*args, **kwargs)


def get_langfuse_sdk(*args: Any, **kwargs: Any) -> Any:
    from aieval_runner.integrations.langfuse import get_langfuse_sdk as get_sdk

    return get_sdk(*args, **kwargs)


def write_score(*args: Any, **kwargs: Any) -> Any:
    from aieval_runner.integrations.langfuse import write_score as write

    return write(*args, **kwargs)


def base_score_metadata(
    case: EvalCase,
    execution: AgentExecution,
    ctx: EvalRunContext,
) -> Dict[str, Any]:
    from aieval_runner.evaluation.flow.scoring import base_score_metadata as build

    return build(case, execution, ctx)


def evaluate_case(
    case: EvalCase,
    execution: AgentExecution,
    ctx: EvalRunContext,
) -> List[EvalScore]:
    from aieval_runner.evaluation.flow.scoring import evaluate_case as evaluate

    return evaluate(case, execution, ctx)


def score_observation_id(score: EvalScore, execution: AgentExecution) -> str | None:
    from aieval_runner.evaluation.flow.scoring import score_observation_id as select

    return select(score, execution)


def create_run_item_payload(
    case: EvalCase,
    execution: AgentExecution,
    ctx: EvalRunContext,
) -> Dict[str, Any]:
    from aieval_runner.evaluation.flow.records import create_run_item_payload as build

    return build(case, execution, ctx)


def experiment_run_metadata(ctx: EvalRunContext) -> Dict[str, Any]:
    from aieval_runner.evaluation.flow.records import experiment_run_metadata as build

    return build(ctx)


def dataset_item_to_raw(item: Any) -> Dict[str, Any]:
    from aieval_runner.evaluation.flow.dataset_items import dataset_item_to_raw as convert

    return convert(item)


def _dataset_item_attr(item: Any, name: str, default: Any = None) -> Any:
    from aieval_runner.evaluation.flow.dataset_items import dataset_item_attr

    return dataset_item_attr(item, name, default)


def _experiment_item_id(item: Any) -> str:
    from aieval_runner.evaluation.flow.dataset_items import experiment_item_id

    return experiment_item_id(item)


def _require_experiment_trace_context(
    trace_id: Any,
    observation_id: Any,
) -> tuple[str, str]:
    from aieval_runner.evaluation.flow.hosted_experiment import (
        require_experiment_trace_context,
    )

    return require_experiment_trace_context(trace_id, observation_id)


def _answer_from_execution(execution: AgentExecution) -> str:
    from aieval_runner.evaluation.flow.records import answer_from_execution

    return answer_from_execution(execution)


def _case_result(
    *,
    case: EvalCase,
    execution: AgentExecution,
    scores: List[EvalScore],
    experiment: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    from aieval_runner.evaluation.flow.records import case_result

    return case_result(
        case=case,
        execution=execution,
        scores=scores,
        experiment=experiment,
    )


def _experiment_result_summary(experiment_result: Any) -> Dict[str, Any]:
    from aieval_runner.evaluation.flow.records import experiment_result_summary

    return experiment_result_summary(experiment_result)


def _experiment_item_summary(item_result: Any) -> Dict[str, Any]:
    from aieval_runner.evaluation.flow.records import experiment_item_summary

    return experiment_item_summary(item_result)


def process_case(
    case: EvalCase,
    ctx: EvalRunContext,
    *,
    api: Any,
    config: RunnerConfig,
) -> Dict[str, Any]:
    from aieval_runner.evaluation.flow.local_run import process_case as run

    return run(
        case,
        ctx,
        api=api,
        config=config,
        run_agent=run_agent_for_case,
        evaluate=evaluate_case,
        write_score_fn=write_score,
    )


def run_hosted_dataset_experiment(
    ctx: EvalRunContext,
    config: RunnerConfig,
) -> List[Dict[str, Any]]:
    from aieval_runner.evaluation.flow.hosted_experiment import (
        run_hosted_dataset_experiment as run,
    )

    return run(
        ctx,
        config,
        get_langfuse_sdk_fn=get_langfuse_sdk,
        run_agent=run_agent_for_case,
        evaluate=evaluate_case,
        write_score_fn=write_score,
    )


__all__ = [
    "AgentExecution",
    "EvalCase",
    "EvalRunContext",
    "EvalScore",
    "EvaluatorResult",
    "RunnerConfig",
    "_answer_from_execution",
    "_case_result",
    "_dataset_item_attr",
    "_experiment_item_id",
    "_experiment_item_summary",
    "_experiment_result_summary",
    "_require_experiment_trace_context",
    "base_score_metadata",
    "create_run_item_payload",
    "dataset_item_to_raw",
    "evaluate_case",
    "experiment_run_metadata",
    "get_langfuse_sdk",
    "process_case",
    "run_agent_for_case",
    "run_hosted_dataset_experiment",
    "score_observation_id",
    "write_score",
]
