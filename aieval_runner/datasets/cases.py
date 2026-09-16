from __future__ import annotations

from typing import Any, Dict

from aieval_runner.core.constants import SCHEMA_VERSION
from aieval_runner.core.models import EvalCase, EvalRunContext
from aieval_runner.core.validation import ensure_str
from aieval_runner.datasets.validation import (
    validate_active_dataset_item,
    validate_real_cases_dataset,
)


def parse_eval_case(raw: Dict[str, Any], *, dataset_name: str) -> EvalCase:
    validate_real_cases_dataset(dataset_name)
    validate_active_dataset_item(raw)
    item_id = ensure_str(raw.get("id"), field_name="dataset_item.id")
    input_obj = raw["input"]
    metadata = raw["metadata"]
    expected = raw["expectedOutput"]
    return EvalCase(
        dataset_name=dataset_name,
        dataset_item_id=item_id,
        case_id=ensure_str(metadata.get("case_id"), field_name="metadata.case_id"),
        original_request=ensure_str(input_obj.get("input"), field_name="input.input"),
        expected_output={
            "expected_result": ensure_str(
                expected.get("expected_result"),
                field_name="expectedOutput.expected_result",
            ),
            "expected_values": dict(expected["expected_values"]),
        },
        tool_contract={
            "expected_tools": list(metadata["expected_tools"]),
            "tool_match_mode": ensure_str(
                metadata.get("tool_match_mode"),
                field_name="metadata.tool_match_mode",
            ),
        },
        item_metadata=dict(metadata),
    )


def build_trace_metadata(case: EvalCase, ctx: EvalRunContext) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "eval_run_id": ctx.eval_run_id,
        "case_id": case.case_id,
        "dataset_name": ctx.dataset_name,
        "dataset_item_id": case.dataset_item_id,
        "agent_version": ctx.agent_version,
        "client": ctx.client,
        "skill_version": ctx.skill_version,
        "db_snapshot_id": ctx.db_snapshot_id,
        "run_mode": ctx.run_mode,
        "response_profile": ctx.response_profile,
        "response_instructions_sha256": ctx.response_instructions_sha256,
    }
