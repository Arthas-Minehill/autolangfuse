from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

from aieval_runner.agent import shutdown_agent_runtime
from aieval_runner.datasets.cases import parse_eval_case
from aieval_runner.runner.config import load_config
from aieval_runner.datasets import load_raw_items
from aieval_runner.evaluation import process_case, run_hosted_dataset_experiment
from aieval_runner.core.json_utils import json_default
from aieval_runner.integrations.langfuse import api_from_config, flush_scores
from aieval_runner.core.models import EvalRunContext
from aieval_runner.runner.schemas import print_protocol_schemas


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _require_langfuse_config(config: Any) -> None:
    if config.langfuse_enabled and config.langfuse_public_key and config.langfuse_secret_key:
        return
    raise SystemExit(
        "必须配置 Langfuse API。请填写 LANGFUSE_PUBLIC_KEY、LANGFUSE_SECRET_KEY 和 LANGFUSE_HOST"
    )


def main(argv: Optional[List[str]] = None) -> int:
    _configure_stdio()
    config = load_config(argv)
    if config.print_schemas:
        print_protocol_schemas()
        return 0

    _require_langfuse_config(config)
    ctx = EvalRunContext(
        eval_run_id=config.run_id,
        dataset_name=config.dataset,
        agent_version=config.agent_version,
        client=config.client,
        skill_version=config.skill_version,
        db_snapshot_id=config.db_snapshot_id,
        response_profile=config.response_profile,
        response_instructions_sha256=config.response_instructions_sha256,
    )

    results: List[Dict[str, Any]] = []
    try:
        if config.items_json_path:
            print(
                json.dumps(
                    {
                        "notice": "--items-json 使用本地调试路径；托管 Langfuse experiment evaluator 不保证触发。"
                    },
                    ensure_ascii=False,
                )
            )
            api = api_from_config(config)
            if api is None:
                _require_langfuse_config(config)
                raise SystemExit("Langfuse API 配置不可用")
            raw_items = load_raw_items(config, api)
            for raw in raw_items:
                try:
                    case = parse_eval_case(raw, dataset_name=config.dataset)
                    result = process_case(case, ctx, api=api, config=config)
                    result["status"] = "success"
                except Exception as exc:
                    result = {
                        "status": "failed",
                        "dataset_item_id": raw.get("id"),
                        "error": str(exc),
                    }
                results.append(result)
                print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))
        else:
            results = run_hosted_dataset_experiment(ctx, config)
            for result in results:
                print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))
    finally:
        shutdown_agent_runtime()
        flush_scores()

    summary = {
        "eval_run_id": ctx.eval_run_id,
        "dataset": ctx.dataset_name,
        "total": len(results),
        "success": len([row for row in results if row.get("status") == "success"]),
        "failed": len([row for row in results if row.get("status") == "failed"]),
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 else 1
