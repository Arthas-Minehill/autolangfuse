"""校验并上传 aieval/real_cases DatasetItem。"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aieval_runner.core.constants import REAL_CASES_DATASET
from aieval_runner.datasets.cases import parse_eval_case


MAX_INPUT_FILE_BYTES = 64 * 1024 * 1024


class LangfuseApiError(RuntimeError):
    """Langfuse Public API 调用失败。"""


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def _require_dataset(dataset: str) -> str:
    name = str(dataset or "").strip()
    if name != REAL_CASES_DATASET:
        raise ValueError(f"仅支持 dataset {REAL_CASES_DATASET}，收到: {name}")
    return name


def _stable_item_id(dataset: str, case_id: str) -> str:
    digest = hashlib.sha256(f"{dataset}\0{case_id}".encode("utf-8")).hexdigest()
    return f"aieval-{digest[:32]}"


def _read_json_items(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}")
    if path.stat().st_size > MAX_INPUT_FILE_BYTES:
        raise ValueError("输入文件超过 64MB 限制")
    if path.suffix.lower() == ".jsonl":
        items: List[Dict[str, Any]] = []
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw_line.strip():
                continue
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 第 {line_number} 行不是合法 JSON") from exc
            if not isinstance(item, dict):
                raise ValueError(f"JSONL 第 {line_number} 行必须是 object")
            items.append(item)
        return items

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("输入文件不是合法 JSON") from exc
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        data = data["items"]
    elif isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("JSON 必须是 DatasetItem object、object 数组或 {items: [...]} ")
    return list(data)


def to_dataset_item_payload(
    dataset: str,
    row: Dict[str, Any],
    *,
    allow_update: bool = False,
) -> Dict[str, Any]:
    dataset = _require_dataset(dataset)
    if not isinstance(row, dict):
        raise ValueError("DatasetItem 必须是 object")
    metadata = row.get("metadata")
    case_id = str(metadata.get("case_id") if isinstance(metadata, dict) else "").strip()
    provided_id = str(row.get("id") or "").strip()
    if provided_id and not allow_update:
        raise ValueError("输入包含已有 DatasetItem id；更新时必须传入 --allow-update")
    payload: Dict[str, Any] = {
        "id": provided_id if provided_id else _stable_item_id(dataset, case_id),
        "datasetName": dataset,
        "input": row.get("input"),
        "expectedOutput": {
            key: row.get("expectedOutput", {}).get(key)
            for key in (
                "expected_result",
                "expected_values",
            )
        }
        if isinstance(row.get("expectedOutput"), dict)
        else row.get("expectedOutput"),
        "metadata": dict(metadata)
        if isinstance(metadata, dict)
        else metadata,
    }
    if row.get("status") in {"ACTIVE", "ARCHIVED"}:
        payload["status"] = row["status"]
    parse_eval_case(payload, dataset_name=dataset)
    return payload


def _api_request(method: str, path: str, *, body: Any = None) -> Dict[str, Any]:
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
    if not public_key or not secret_key:
        raise LangfuseApiError("缺少 LANGFUSE_PUBLIC_KEY 或 LANGFUSE_SECRET_KEY")
    token = base64.b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode("ascii")
    request = urllib.request.Request(
        f"{host}/api/public{path}",
        data=(
            json.dumps(body, ensure_ascii=False).encode("utf-8")
            if body is not None
            else None
        ),
        headers={"Authorization": f"Basic {token}", "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw_bytes = response.read(8 * 1024 * 1024 + 1)
            if len(raw_bytes) > 8 * 1024 * 1024:
                raise LangfuseApiError("Langfuse API 响应超过 8MB 限制")
            raw = raw_bytes.decode("utf-8")
            value = json.loads(raw) if raw else {}
            return value if isinstance(value, dict) else {"data": value}
    except urllib.error.HTTPError as exc:
        detail = exc.read(64 * 1024).decode("utf-8", errors="replace")[:1000]
        raise LangfuseApiError(f"Langfuse API {method} {path} 失败: {exc.code} {detail}") from exc


def _dataset_item_exists(dataset_item_id: str) -> bool:
    quoted_id = urllib.parse.quote(dataset_item_id, safe="")
    try:
        _api_request("GET", f"/dataset-items/{quoted_id}")
    except LangfuseApiError as exc:
        if " 404 " in f" {exc} ":
            return False
        raise
    return True


def upload_items(
    dataset: str,
    rows: Iterable[Dict[str, Any]],
    *,
    commit: bool,
    limit: int,
    allow_update: bool = False,
) -> List[Dict[str, Any]]:
    dataset = _require_dataset(dataset)
    selected = list(rows)
    if limit > 0:
        selected = selected[:limit]
    payloads = [
        to_dataset_item_payload(dataset, row, allow_update=allow_update)
        for row in selected
    ]
    ids = [str(payload["id"]) for payload in payloads]
    if len(ids) != len(set(ids)):
        raise ValueError("上传批次包含重复 DatasetItem id")
    if commit and not allow_update:
        existing = [item_id for item_id in ids if _dataset_item_exists(item_id)]
        if existing:
            raise ValueError(f"DatasetItem id 已存在: {existing[0]}；更新时传入 --allow-update")

    results: List[Dict[str, Any]] = []
    for payload in payloads:
        response = _api_request("POST", "/dataset-items", body=payload) if commit else {}
        result = {
            "case_id": payload["metadata"]["case_id"],
            "dataset_item_id": payload["id"],
            "uploaded": commit,
            "response_id": response.get("id") if commit else None,
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help=f"仅支持 {REAL_CASES_DATASET}")
    parser.add_argument("--file", required=True, help="JSON 或 JSONL 文件路径")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少条，0 表示全部")
    parser.add_argument("--commit", action="store_true", help="真正写入；默认 dry-run")
    parser.add_argument("--allow-update", action="store_true", help="允许保留输入 id 并更新")
    return parser


def main() -> int:
    _load_env_file(ROOT / ".env")
    args = build_arg_parser().parse_args()
    dataset = _require_dataset(
        args.dataset or os.getenv("LANGFUSE_DATASET_NAME") or REAL_CASES_DATASET
    )
    results = upload_items(
        dataset,
        _read_json_items(Path(args.file)),
        commit=args.commit,
        limit=args.limit,
        allow_update=args.allow_update,
    )
    print(
        json.dumps(
            {"summary": {"dataset": dataset, "commit": args.commit, "total": len(results)}},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
