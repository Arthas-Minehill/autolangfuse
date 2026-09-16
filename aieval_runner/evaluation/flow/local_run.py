from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, List

from aieval_runner.evaluation.flow.records import create_run_item_payload
from aieval_runner.evaluation.flow.scoring import score_observation_id
from aieval_runner.integrations.langfuse import LangfuseClient
from aieval_runner.core.models import (
    AgentExecution,
    EvalCase,
    EvalRunContext,
    EvalScore,
    RunnerConfig,
)


RunAgent = Callable[..., AgentExecution]
EvaluateCase = Callable[[EvalCase, AgentExecution, EvalRunContext], List[EvalScore]]
WriteScore = Callable[..., Any]


def process_case(
    case: EvalCase,
    ctx: EvalRunContext,
    *,
    api: LangfuseClient,
    config: RunnerConfig,
    run_agent: RunAgent,
    evaluate: EvaluateCase,
    write_score_fn: WriteScore,
) -> Dict[str, Any]:
    execution = run_agent(case, ctx, config=config)
    run_item_payload = create_run_item_payload(case, execution, ctx)
    scores = evaluate(case, execution, ctx)

    run_item = api.request("POST", "/dataset-run-items", body=run_item_payload)
    for score in scores:
        write_score_fn(
            score,
            config=config,
            trace_id=execution.trace_id,
            observation_id=score_observation_id(score, execution),
        )

    return {
        "case": {
            "dataset_name": case.dataset_name,
            "dataset_item_id": case.dataset_item_id,
            "case_id": case.case_id,
            "original_request": case.original_request,
            "expected_tools": case.tool_contract["expected_tools"],
            "tool_match_mode": case.tool_contract["tool_match_mode"],
        },
        "trace_metadata": execution.trace_metadata,
        "dataset_run_item_request": run_item_payload,
        "dataset_run_item_response": run_item,
        "agent_output": execution.output,
        "trace_mapping": execution.mapping,
        "scores": [asdict(score) for score in scores],
    }
