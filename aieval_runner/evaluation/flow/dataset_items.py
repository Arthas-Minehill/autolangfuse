from __future__ import annotations

from typing import Any, Dict


def dataset_item_attr(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def dataset_item_to_raw(item: Any) -> Dict[str, Any]:
    expected_output = dataset_item_attr(item, "expected_output")
    if expected_output is None:
        expected_output = dataset_item_attr(item, "expectedOutput")
    metadata = dataset_item_attr(item, "metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {"metadata": metadata}
    return {
        "id": str(dataset_item_attr(item, "id", "") or ""),
        "input": dataset_item_attr(item, "input"),
        "expectedOutput": expected_output,
        "metadata": metadata,
    }


def experiment_item_id(item: Any) -> str:
    value = dataset_item_attr(item, "id", "")
    return str(value or "")
