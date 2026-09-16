from __future__ import annotations

from typing import Any, Dict, Iterable

from aieval_runner.core.constants import (
    MAX_CASE_ID_CHARS,
    MAX_DATASET_ITEM_ID_CHARS,
    MAX_EXPECTED_TOOLS,
    MAX_INPUT_CHARS,
    MAX_TOOL_NAME_CHARS,
    REAL_CASES_DATASET,
)
from aieval_runner.core.validation import ensure_str


TOOL_MATCH_MODES = {"any", "all"}


def validate_real_cases_dataset(dataset_name: str) -> None:
    if str(dataset_name or "").strip() != REAL_CASES_DATASET:
        raise ValueError(
            f"仅支持 Langfuse dataset {REAL_CASES_DATASET}，收到: {dataset_name}"
        )


def validate_active_dataset_item(raw: Dict[str, Any]) -> None:
    """校验 aieval/real_cases 的最小工具契约。"""

    if not isinstance(raw, dict):
        raise ValueError("DatasetItem 必须是 object")
    item_id = ensure_str(raw.get("id"), field_name="dataset_item.id")
    if len(item_id) > MAX_DATASET_ITEM_ID_CHARS:
        raise ValueError(f"dataset_item.id 最多 {MAX_DATASET_ITEM_ID_CHARS} 个字符")
    input_obj = raw.get("input")
    if not isinstance(input_obj, dict):
        raise ValueError("DatasetItem.input 必须是 object")
    request = ensure_str(input_obj.get("input"), field_name="input.input")
    if len(request) > MAX_INPUT_CHARS:
        raise ValueError(f"input.input 最多 {MAX_INPUT_CHARS} 个字符")
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("DatasetItem.metadata 必须是 object")
    case_id = ensure_str(metadata.get("case_id"), field_name="metadata.case_id")
    if len(case_id) > MAX_CASE_ID_CHARS:
        raise ValueError(f"metadata.case_id 最多 {MAX_CASE_ID_CHARS} 个字符")
    expected = raw.get("expectedOutput")
    if not isinstance(expected, dict):
        raise ValueError("DatasetItem.expectedOutput 必须是 object")
    if "expected_tools" in expected or "tool_match_mode" in expected:
        raise ValueError(
            "expected_tools 和 tool_match_mode 已迁移到 metadata，不得保留在 expectedOutput"
        )
    if "expected_result" in metadata or "expected_values" in metadata:
        raise ValueError(
            "expected_result 和 expected_values 已迁移到 expectedOutput，不得保留在 metadata"
        )
    expected_result = ensure_str(
        expected.get("expected_result"),
        field_name="expectedOutput.expected_result",
    )
    if len(expected_result) > MAX_INPUT_CHARS:
        raise ValueError(
            f"expectedOutput.expected_result 最多 {MAX_INPUT_CHARS} 个字符"
        )
    if not isinstance(expected.get("expected_values"), dict):
        raise ValueError("expectedOutput.expected_values 必须是 object")
    tools = metadata.get("expected_tools")
    if not isinstance(tools, list):
        raise ValueError("metadata.expected_tools 必须是字符串数组")
    tools = [
        ensure_str(tool, field_name="metadata.expected_tools[]")
        for tool in tools
    ]
    if not tools:
        raise ValueError("metadata.expected_tools 必须是非空数组")
    if len(tools) > MAX_EXPECTED_TOOLS:
        raise ValueError(
            f"metadata.expected_tools 最多允许 {MAX_EXPECTED_TOOLS} 项"
        )
    if any(len(tool) > MAX_TOOL_NAME_CHARS for tool in tools):
        raise ValueError(
            f"metadata.expected_tools 单项最多 {MAX_TOOL_NAME_CHARS} 个字符"
        )
    if len(set(tools)) != len(tools):
        raise ValueError("metadata.expected_tools 不得包含重复工具")
    from aieval_runner.evaluation.events import expected_tool_leaf

    normalized_tools = [expected_tool_leaf(tool) for tool in tools]
    if any(not tool for tool in normalized_tools):
        raise ValueError("metadata.expected_tools 必须包含有效工具标识")
    if len(set(normalized_tools)) != len(normalized_tools):
        raise ValueError("metadata.expected_tools 归一化后不得重复")
    mode = ensure_str(
        metadata.get("tool_match_mode"),
        field_name="metadata.tool_match_mode",
    )
    if mode not in TOOL_MATCH_MODES:
        raise ValueError("metadata.tool_match_mode 仅允许 any 或 all")


def validate_active_dataset(items: Iterable[Dict[str, Any]]) -> None:
    errors = []
    for raw in items:
        try:
            validate_active_dataset_item(raw)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError("real_cases 数据集校验失败:\n" + "\n".join(errors))
