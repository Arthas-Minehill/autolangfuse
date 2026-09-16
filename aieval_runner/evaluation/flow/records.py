from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from aieval_runner.core.constants import SCHEMA_VERSION
from aieval_runner.core.models import AgentExecution, EvalCase, EvalRunContext, EvalScore
from aieval_runner.evaluation.events import load_event_summary
from aieval_runner.evaluation.events import tool_name_matches


MAX_JUDGE_TOOL_EVIDENCE = 16
MAX_JUDGE_RESULT_CHARS = 4_000


def create_run_item_payload(
    case: EvalCase,
    execution: AgentExecution,
    ctx: EvalRunContext,
) -> Dict[str, Any]:
    return {
        "runName": ctx.eval_run_id,
        "datasetItemId": case.dataset_item_id,
        "traceId": execution.trace_id,
        "observationId": execution.observation_id,
        "runDescription": "aieval external runner automated evaluation",
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "eval_run_id": ctx.eval_run_id,
            "case_id": case.case_id,
            "agent_version": ctx.agent_version,
            "client": ctx.client,
            "skill_version": ctx.skill_version,
            "db_snapshot_id": ctx.db_snapshot_id,
        },
    }


def experiment_run_metadata(ctx: EvalRunContext) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "eval_run_id": ctx.eval_run_id,
        "agent_version": ctx.agent_version,
        "client": ctx.client,
        "skill_version": ctx.skill_version,
        "db_snapshot_id": ctx.db_snapshot_id,
    }


def answer_from_execution(execution: AgentExecution) -> str:
    output = execution.output
    if isinstance(output, dict):
        answer = output.get("answer")
        if isinstance(answer, str):
            return answer
    return ""


def _select_judge_calls(
    calls: List[Dict[str, Any]], expected_tools: List[str]
) -> List[Dict[str, Any]]:
    if len(calls) <= MAX_JUDGE_TOOL_EVIDENCE:
        return calls
    ranked = sorted(
        enumerate(calls),
        key=lambda pair: (
            any(tool_name_matches(expected, pair[1]) for expected in expected_tools),
            str(pair[1].get("status") or "").lower() == "completed",
            pair[1].get("result") not in (None, "", [], {}),
            pair[0],
        ),
        reverse=True,
    )[:MAX_JUDGE_TOOL_EVIDENCE]
    return [call for _, call in sorted(ranked, key=lambda pair: pair[0])]


def judge_output_from_execution(
    execution: AgentExecution, case: EvalCase
) -> Dict[str, Any]:
    """为 experiment Judge 提供最终回答和同次运行的有界真实工具证据。"""

    answer = answer_from_execution(execution)
    output: Dict[str, Any] = {
        "answer": answer,
        "tool_evidence": [],
        "tool_evidence_total": 0,
        "tool_evidence_truncated": False,
    }
    raw_output = execution.output
    event_path_value = raw_output.get("event_path") if isinstance(raw_output, dict) else None
    if not event_path_value:
        output["tool_evidence_error"] = "missing_event_path"
        return output
    try:
        summary = load_event_summary(Path(str(event_path_value)))
    except (OSError, TypeError, ValueError) as exc:
        output["tool_evidence_error"] = f"event_summary_failed:{type(exc).__name__}"
        return output

    calls = summary.tool_calls
    selected = _select_judge_calls(
        calls, list(case.tool_contract.get("expected_tools") or [])
    )
    evidence = []
    for call in selected:
        result_json = json.dumps(
            call.get("result"),
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        error_json = json.dumps(
            call.get("error"),
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        evidence.append(
            {
                "server": call.get("server"),
                "tool": call.get("tool"),
                "status": call.get("status"),
                "has_error": call.get("error") not in (None, "", False),
                "error_excerpt": error_json[:MAX_JUDGE_RESULT_CHARS],
                "result_excerpt": result_json[:MAX_JUDGE_RESULT_CHARS],
                "result_truncated": len(result_json) > MAX_JUDGE_RESULT_CHARS,
            }
        )
    output["tool_evidence"] = evidence
    output["tool_evidence_total"] = len(calls)
    output["tool_evidence_truncated"] = len(calls) > len(selected)
    if summary.error:
        output["tool_evidence_diagnostic"] = summary.error
    return output


def case_result(
    *,
    case: EvalCase,
    execution: AgentExecution,
    scores: List[EvalScore],
    experiment: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    result = {
        "case": {
            "dataset_name": case.dataset_name,
            "dataset_item_id": case.dataset_item_id,
            "case_id": case.case_id,
            "original_request": case.original_request,
            "expected_tools": case.tool_contract["expected_tools"],
            "tool_match_mode": case.tool_contract["tool_match_mode"],
        },
        "trace_metadata": execution.trace_metadata,
        "agent_output": execution.output,
        "trace_mapping": execution.mapping,
        "scores": [asdict(score) for score in scores],
    }
    if experiment is not None:
        result["experiment"] = experiment
    return result


def experiment_result_summary(experiment_result: Any) -> Dict[str, Any]:
    return {
        "name": getattr(experiment_result, "name", None),
        "run_name": getattr(experiment_result, "run_name", None),
        "experiment_id": getattr(experiment_result, "experiment_id", None),
        "dataset_run_id": getattr(experiment_result, "dataset_run_id", None),
        "dataset_run_url": getattr(experiment_result, "dataset_run_url", None),
    }


def experiment_item_summary(item_result: Any) -> Dict[str, Any]:
    return {
        "trace_id": getattr(item_result, "trace_id", None),
        "dataset_run_id": getattr(item_result, "dataset_run_id", None),
        "output": getattr(item_result, "output", None),
    }
