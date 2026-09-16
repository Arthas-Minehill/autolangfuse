from __future__ import annotations

from typing import Any


def parse_eval_case(*args: Any, **kwargs: Any) -> Any:
    from aieval_runner.datasets.cases import parse_eval_case as parse

    return parse(*args, **kwargs)


def build_trace_metadata(*args: Any, **kwargs: Any) -> Any:
    from aieval_runner.datasets.cases import build_trace_metadata as build

    return build(*args, **kwargs)


def load_raw_items(*args: Any, **kwargs: Any) -> Any:
    from aieval_runner.datasets.loading import load_raw_items as load

    return load(*args, **kwargs)


def filter_items_by_id(*args: Any, **kwargs: Any) -> Any:
    from aieval_runner.datasets.loading import filter_items_by_id as filter_items

    return filter_items(*args, **kwargs)


__all__ = [
    "build_trace_metadata",
    "filter_items_by_id",
    "load_raw_items",
    "parse_eval_case",
]
