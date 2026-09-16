from __future__ import annotations

from typing import Any, Dict, Optional


JsonObject = Dict[str, Any]


def safe_text(value: Any, *, limit: int = 200) -> Optional[str]:
    if value is None:
        return None
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return text[:limit] if text else None
