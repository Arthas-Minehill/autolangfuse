from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from aieval_runner.core.constants import PROJECT_ROOT
from aieval_runner.core.models import EvalRunContext, RunnerConfig


def load_items_json(path: str) -> Optional[List[Dict[str, Any]]]:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        data = data.get("items") or data.get("data")
    if not isinstance(data, list):
        raise ValueError("--items-json 必须是数组，或包含 items/data 数组的 object")
    return [item for item in data if isinstance(item, dict)]


def safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return safe.strip("._") or "case"


def mapping_path(config: RunnerConfig, ctx: EvalRunContext) -> Path:
    if config.mapping_path.strip():
        return Path(config.mapping_path)
    return PROJECT_ROOT / ".aieval" / "runs" / ctx.eval_run_id / "trace-map.jsonl"


def read_mapping_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def wait_for_mapping(path: Path, *, eval_run_id: str, case_id: str, timeout_seconds: int) -> Dict[str, Any]:
    deadline = time.monotonic() + max(1, timeout_seconds)
    while time.monotonic() <= deadline:
        matches = [
            record
            for record in read_mapping_records(path)
            if record.get("eval_run_id") == eval_run_id and record.get("case_id") == case_id
        ]
        if matches:
            return matches[-1]
        time.sleep(0.5)
    raise TimeoutError(f"等待 hook 写入 trace mapping 超时: case_id={case_id}, path={path}")
