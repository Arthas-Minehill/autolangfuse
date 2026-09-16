from __future__ import annotations

from typing import Any


MAX_SAFE_INTEGER = 9_007_199_254_740_991


def json_default(value: Any) -> str:
    return str(value)
