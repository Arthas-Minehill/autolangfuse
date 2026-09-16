from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from aieval_runner.integrations.langfuse import LangfuseClient
from aieval_runner.storage.local import load_items_json
from aieval_runner.core.models import RunnerConfig
from aieval_runner.datasets.validation import validate_active_dataset


def list_dataset_items(api: LangfuseClient, dataset_name: str, *, limit: int) -> Iterable[Dict[str, Any]]:
    page = 1
    remaining: Optional[int] = limit if limit > 0 else None
    while True:
        page_limit = 100 if remaining is None else max(1, min(100, remaining))
        result = api.request(
            "GET",
            "/dataset-items",
            query={"datasetName": dataset_name, "page": page, "limit": page_limit},
        )
        items = result.get("data") or result.get("items") or []
        if not items:
            return
        for item in items:
            if isinstance(item, dict):
                yield item
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        return
        meta = result.get("meta") or {}
        total_pages = int(meta.get("totalPages") or meta.get("total_pages") or page)
        if page >= total_pages:
            return
        page += 1


def filter_items_by_id(items: Iterable[Any], item_id: str) -> List[Any]:
    target = str(item_id or "").strip()
    if not target:
        return list(items)
    matched: List[Any] = []
    for item in items:
        current = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
        if str(current or "") == target:
            matched.append(item)
    if not matched:
        raise ValueError(f"DatasetItem id 未找到: {target}")
    return matched


def load_raw_items(config: RunnerConfig, api: LangfuseClient) -> List[Dict[str, Any]]:
    local_items = load_items_json(config.items_json_path)
    if local_items is not None:
        raw_items = filter_items_by_id(local_items, config.item_id)
        raw_items = raw_items if config.limit <= 0 else raw_items[: config.limit]
    else:
        list_limit = 0 if config.item_id else config.limit
        raw_items = list(list_dataset_items(api, config.dataset, limit=list_limit))
        raw_items = filter_items_by_id(raw_items, config.item_id)
    validate_active_dataset(raw_items)
    return raw_items
