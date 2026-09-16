from __future__ import annotations

import json
from typing import Any, Dict, List


def ensure_dict(value: Any, *, field_name: str) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} 必须是 JSON object") from exc
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"{field_name} 必须是 object")


def ensure_str(value: Any, *, field_name: str) -> str:
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError(f"{field_name} 必须是非空字符串")


def ensure_str_list(value: Any, *, field_name: str) -> List[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [part.strip() for part in value.split(",") if part.strip()]
    if not isinstance(value, list):
        raise ValueError(f"{field_name} 必须是字符串数组")
    return [ensure_str(item, field_name=f"{field_name}[]") for item in value]

