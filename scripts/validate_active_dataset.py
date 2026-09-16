"""从 Langfuse 读取并校验当前 aieval/real_cases 全量数据。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aieval_runner.datasets.cases import parse_eval_case
from aieval_runner.datasets.loading import list_dataset_items
from aieval_runner.integrations.langfuse import LangfuseClient


DATASET_NAME = "aieval/real_cases"


def main() -> int:
    load_dotenv(ROOT / ".env")
    api = LangfuseClient(
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
    )
    raw_items = list(list_dataset_items(api, DATASET_NAME, limit=0))
    cases = [parse_eval_case(item, dataset_name=DATASET_NAME) for item in raw_items]
    if not cases:
        raise ValueError(f"{DATASET_NAME} 不能为空")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("metadata.case_id 不得重复")
    print(f"通过：{DATASET_NAME}（{len(cases)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
