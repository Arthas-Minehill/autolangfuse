"""将 real_cases 查询类 Item 收敛为业务结果契约。"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.upload_dataitems import _api_request, _load_env_file


DATASET = "aieval/real_cases"


def _items() -> List[Dict[str, Any]]:
    response = _api_request(
        "GET", "/dataset-items?datasetName=aieval%2Freal_cases&page=1&limit=100"
    )
    return list(response.get("data") or [])


def _scalar_values(metadata: Dict[str, Any], row_count: int, columns: List[str]) -> Dict[str, Any]:
    values: Dict[str, Any] = {"row_count": row_count, "columns": columns}
    summary = metadata.get("expected_summary")
    if isinstance(summary, dict):
        for key, value in summary.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                values[key] = copy.deepcopy(value)
    return values


def _expected_result(item: Dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    question = (item.get("input") or {}).get("input") or ""
    columns = metadata.get("result_columns") or []
    scope = metadata.get("result_scope") or "按问题返回可验证结果"
    rules = metadata.get("validation_rules") or []
    result_rules = [
        str(rule)
        for rule in rules
        if not any(token in str(rule).lower() for token in ("construct_query", "execute_query", "status=completed", "最终回答"))
    ]
    parts = [f"围绕问题“{question}”返回正确的 Metabase 查询结果。", f"结果范围：{scope}。"]
    if columns:
        parts.append("结果至少包含字段：" + "、".join(map(str, columns)) + "。")
    if result_rules:
        parts.append("业务要求：" + "；".join(result_rules) + "。")
    return "".join(parts)


def build_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    metadata = item.get("metadata") or {}
    rows = metadata.get("result_rows") if isinstance(metadata.get("result_rows"), list) else []
    columns = metadata.get("result_columns") if isinstance(metadata.get("result_columns"), list) else []
    new_metadata = {
        "case_id": metadata.get("case_id"),
        "type": metadata.get("type"),
        "expected_result": _expected_result(item),
        "expected_values": _scalar_values(metadata, len(rows), list(columns)),
    }
    return {
        "id": item.get("id"),
        "datasetName": DATASET,
        "input": item.get("input"),
        "expectedOutput": item.get("expectedOutput"),
        "metadata": new_metadata,
        "status": item.get("status", "ACTIVE"),
    }


def main() -> int:
    _load_env_file(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", action="store_true")
    args = parser.parse_args()
    items = [item for item in _items() if (item.get("metadata") or {}).get("type") == "查询取数"]
    if len(items) != 11:
        raise RuntimeError(f"预期 11 条查询取数 Item，实际 {len(items)} 条")
    payloads = [build_payload(item) for item in items]
    if not args.commit:
        print(json.dumps(payloads, ensure_ascii=False, indent=2))
        return 0
    results = []
    for payload in payloads:
        response = _api_request("POST", "/dataset-items", body=payload)
        results.append({"id": payload["id"], "response_id": response.get("id")})
    print(json.dumps({"updated": len(results), "items": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
