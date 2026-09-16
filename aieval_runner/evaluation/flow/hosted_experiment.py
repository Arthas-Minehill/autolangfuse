from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, Dict, List

from aieval_runner.datasets.cases import parse_eval_case
from aieval_runner.datasets import filter_items_by_id
from aieval_runner.evaluation.flow.dataset_items import (
    dataset_item_to_raw,
    experiment_item_id,
)
from aieval_runner.evaluation.flow.local_run import EvaluateCase, RunAgent, WriteScore
from aieval_runner.evaluation.flow.records import (
    case_result,
    experiment_item_summary,
    experiment_result_summary,
    experiment_run_metadata,
    judge_output_from_execution,
)
from aieval_runner.evaluation.flow.scoring import score_observation_id
from aieval_runner.core.models import EvalRunContext, EvalScore, RunnerConfig


GetLangfuseSdk = Callable[[RunnerConfig], Any]


def require_experiment_trace_context(
    trace_id: Any,
    observation_id: Any,
) -> tuple[str, str]:
    if not trace_id or not observation_id:
        raise RuntimeError(
            "Langfuse experiment task did not expose current trace id and observation id"
        )
    return str(trace_id), str(observation_id)


def run_hosted_dataset_experiment(
    ctx: EvalRunContext,
    config: RunnerConfig,
    *,
    get_langfuse_sdk_fn: GetLangfuseSdk,
    run_agent: RunAgent,
    evaluate: EvaluateCase,
    write_score_fn: WriteScore,
) -> List[Dict[str, Any]]:
    lf = get_langfuse_sdk_fn(config)
    dataset = lf.get_dataset(config.dataset)
    dataset.items = filter_items_by_id(dataset.items, config.item_id)
    if config.limit > 0:
        dataset.items = list(dataset.items)[: config.limit]

    results_by_item_id: Dict[str, Dict[str, Any]] = {}
    results_lock = threading.Lock()

    def _run_item_task(
        *,
        item_id: str,
        raw: Dict[str, Any],
        current_trace_id: str,
        current_observation_id: str,
    ) -> Dict[str, Any]:
        try:
            case = parse_eval_case(raw, dataset_name=config.dataset)
            execution = run_agent(
                case,
                ctx,
                config=config,
                langfuse_trace_id=current_trace_id,
                langfuse_parent_observation_id=current_observation_id,
            )
            scores: List[EvalScore] = []
            score_errors: List[Dict[str, str]] = []
            try:
                scores = evaluate(case, execution, ctx)
            except Exception as score_exc:
                score_errors.append(
                    {"stage": "evaluate_case", "error": str(score_exc)}
                )
            else:
                for score in scores:
                    try:
                        write_score_fn(
                            score,
                            config=config,
                            trace_id=execution.trace_id,
                            observation_id=score_observation_id(score, execution),
                        )
                    except Exception as score_exc:
                        score_errors.append(
                            {
                                "stage": "write_score",
                                "score": score.name,
                                "error": str(score_exc),
                            }
                        )
            result = case_result(
                case=case,
                execution=execution,
                scores=scores,
                experiment={
                    "trace_id": current_trace_id,
                    "root_observation_id": current_observation_id,
                },
            )
            result["status"] = "success"
            result["deterministic_score_status"] = (
                "failed" if score_errors else "success"
            )
            if score_errors:
                result["deterministic_score_errors"] = score_errors
            with results_lock:
                results_by_item_id[case.dataset_item_id] = result
            return judge_output_from_execution(execution, case)
        except Exception as exc:
            with results_lock:
                if item_id not in results_by_item_id:
                    results_by_item_id[item_id] = {
                        "status": "failed",
                        "dataset_item_id": item_id or raw.get("id"),
                        "error": str(exc),
                    }
            raise

    async def task(*, item: Any, **_: Any) -> Dict[str, Any]:
        item_id = experiment_item_id(item)
        raw = dataset_item_to_raw(item)
        try:
            current_trace_id, current_observation_id = require_experiment_trace_context(
                lf.get_current_trace_id(),
                lf.get_current_observation_id(),
            )
            return await asyncio.to_thread(
                _run_item_task,
                item_id=item_id,
                raw=raw,
                current_trace_id=current_trace_id,
                current_observation_id=current_observation_id,
            )
        except Exception as exc:
            with results_lock:
                if item_id not in results_by_item_id:
                    results_by_item_id[item_id] = {
                        "status": "failed",
                        "dataset_item_id": item_id or raw.get("id"),
                        "error": str(exc),
                    }
            raise

    experiment_result = dataset.run_experiment(
        name="aieval external runner",
        run_name=ctx.eval_run_id,
        description="aieval external runner automated evaluation",
        task=task,
        max_concurrency=config.max_concurrency,
        metadata=experiment_run_metadata(ctx),
    )
    experiment_summary = experiment_result_summary(experiment_result)
    for item_result in getattr(experiment_result, "item_results", []):
        item_id = experiment_item_id(getattr(item_result, "item", None))
        if item_id in results_by_item_id:
            results_by_item_id[item_id]["experiment"] = {
                **results_by_item_id[item_id].get("experiment", {}),
                **experiment_summary,
                "item_result": experiment_item_summary(item_result),
            }

    results: List[Dict[str, Any]] = []
    for item in dataset.items:
        item_id = experiment_item_id(item)
        result = results_by_item_id.get(item_id)
        if result is None:
            result = {
                "status": "failed",
                "dataset_item_id": item_id,
                "error": "experiment task did not produce a result",
                "experiment": experiment_summary,
            }
        elif "experiment" not in result:
            result["experiment"] = experiment_summary
        else:
            result["experiment"] = {
                **experiment_summary,
                **result["experiment"],
            }
        results.append(result)
    return results
