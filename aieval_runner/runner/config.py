from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from aieval_runner.core.constants import (
    AGENT_WORKSPACE_ROOT,
    EVAL_AGENT_INSTRUCTIONS_PATH,
    EVAL_RESPONSE_PROFILE,
    PROJECT_ROOT,
    REAL_CASES_DATASET,
    RUN_CONFIG,
)
from aieval_runner.core.models import RunnerConfig
from aieval_runner.agent.workspace import response_instruction_digest
from aieval_runner.storage.local import safe_filename


def now_run_id() -> str:
    return "eval-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def resolve_from_root(value: str, default: Path) -> str:
    path = Path(value) if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AIEval external Langfuse Dataset runner")
    parser.add_argument(
        "--dataset",
        default=None,
        help=f"Langfuse dataset name（仅支持 {REAL_CASES_DATASET}）",
    )
    parser.add_argument("--run-id", default=None, help="Eval Run ID / Langfuse runName")
    parser.add_argument("--agent-version", default=None, help="Agent version")
    parser.add_argument("--client", default=None, help="Execution client name")
    parser.add_argument("--skill-version", default=None, help="Skill or tool package version")
    parser.add_argument("--db-snapshot-id", default=None, help="Database snapshot id")
    parser.add_argument("--limit", type=int, default=None, help="Maximum DatasetItem count; 0 means unlimited")
    parser.add_argument("--item-id", default=None, help="Run exactly one DatasetItem by id")
    parser.add_argument("--codex-cwd", default=None, help="Codex working directory")
    parser.add_argument("--codex-bin", default=None, help="Codex executable path")
    parser.add_argument("--codex-model", default=None, help="Codex model override")
    parser.add_argument("--codex-sandbox", default=None, help="Codex sandbox mode")
    parser.add_argument(
        "--app-server-request-timeout-seconds",
        type=int,
        default=None,
        help="codex app-server JSON-RPC request timeout",
    )
    parser.add_argument(
        "--app-server-turn-timeout-seconds",
        type=int,
        default=None,
        help="codex app-server turn timeout",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent DatasetItem tasks for hosted Langfuse experiments",
    )
    parser.add_argument("--mapping-path", default=None, help="Trace mapping JSONL path written by hook")
    parser.add_argument("--mapping-timeout-seconds", type=int, default=None, help="Trace mapping wait timeout")
    parser.add_argument("--items-json", default=None, help="Local DatasetItem JSON")
    parser.add_argument("--print-schemas", action="store_true", help="Print protocol schemas")
    return parser


def config_from_env() -> Dict[str, Any]:
    config = dict(RUN_CONFIG)
    config.update(
        {
            "langfuse_enabled": env_bool("LANGFUSE_ENABLED", True),
            "langfuse_public_key": os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            "langfuse_secret_key": os.getenv("LANGFUSE_SECRET_KEY", ""),
            "langfuse_host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            "dataset": os.getenv("LANGFUSE_DATASET_NAME", str(config["dataset"])),
            "run_id": os.getenv("AIEVAL_RUN_ID", str(config["run_id"])),
            "agent_version": os.getenv("AIEVAL_AGENT_VERSION", str(config["agent_version"])),
            "client": os.getenv("AIEVAL_CLIENT", str(config["client"])),
            "skill_version": os.getenv("AIEVAL_SKILL_VERSION", str(config["skill_version"])),
            "db_snapshot_id": os.getenv("AIEVAL_DB_SNAPSHOT_ID", str(config["db_snapshot_id"])),
            "limit": env_int("AIEVAL_LIMIT", int(config["limit"])),
            "item_id": os.getenv("AIEVAL_ITEM_ID", str(config["item_id"])),
            "codex_cwd": os.getenv("AIEVAL_CODEX_CWD", str(config["codex_cwd"])),
            "codex_bin": os.getenv("AIEVAL_CODEX_BIN", str(config.get("codex_bin", ""))),
            "codex_model": os.getenv("AIEVAL_CODEX_MODEL", str(config["codex_model"])),
            "codex_sandbox": os.getenv("AIEVAL_CODEX_SANDBOX", str(config["codex_sandbox"])),
            "app_server_request_timeout_seconds": env_int(
                "AIEVAL_APP_SERVER_REQUEST_TIMEOUT_SECONDS",
                int(config["app_server_request_timeout_seconds"]),
            ),
            "app_server_turn_timeout_seconds": env_int(
                "AIEVAL_APP_SERVER_TURN_TIMEOUT_SECONDS",
                int(config["app_server_turn_timeout_seconds"]),
            ),
            "max_concurrency": env_int("AIEVAL_MAX_CONCURRENCY", int(config["max_concurrency"])),
            "mapping_path": os.getenv("AIEVAL_MAPPING_PATH", str(config["mapping_path"])),
            "mapping_timeout_seconds": env_int(
                "AIEVAL_MAPPING_TIMEOUT_SECONDS",
                int(config["mapping_timeout_seconds"]),
            ),
            "items_json_path": os.getenv("AIEVAL_ITEMS_JSON", str(config["items_json_path"])),
        }
    )
    return config


def apply_arg_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    for arg_name, config_name in [
        ("dataset", "dataset"),
        ("run_id", "run_id"),
        ("agent_version", "agent_version"),
        ("client", "client"),
        ("skill_version", "skill_version"),
        ("db_snapshot_id", "db_snapshot_id"),
        ("limit", "limit"),
        ("item_id", "item_id"),
        ("codex_cwd", "codex_cwd"),
        ("codex_bin", "codex_bin"),
        ("codex_model", "codex_model"),
        ("codex_sandbox", "codex_sandbox"),
        ("app_server_request_timeout_seconds", "app_server_request_timeout_seconds"),
        ("app_server_turn_timeout_seconds", "app_server_turn_timeout_seconds"),
        ("max_concurrency", "max_concurrency"),
        ("mapping_path", "mapping_path"),
        ("mapping_timeout_seconds", "mapping_timeout_seconds"),
        ("items_json", "items_json_path"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            config[config_name] = value

    if args.print_schemas:
        config["print_schemas"] = True
    return config


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    dataset = str(config.get("dataset") or REAL_CASES_DATASET).strip()
    if dataset != REAL_CASES_DATASET and not bool(config.get("print_schemas")):
        raise ValueError(
            f"仅支持 Langfuse dataset {REAL_CASES_DATASET}，收到: {dataset}"
        )
    config["dataset"] = dataset
    if not config["run_id"]:
        config["run_id"] = now_run_id()
    raw_cwd = str(config.get("codex_cwd") or "").strip()
    resolved_cwd = Path(
        resolve_from_root(raw_cwd, AGENT_WORKSPACE_ROOT)
    )
    if not raw_cwd or resolved_cwd == AGENT_WORKSPACE_ROOT.resolve():
        resolved_cwd = AGENT_WORKSPACE_ROOT.resolve() / safe_filename(str(config["run_id"]))
    config["codex_cwd"] = str(resolved_cwd)
    Path(config["codex_cwd"]).mkdir(parents=True, exist_ok=True)
    raw_codex_bin = str(config.get("codex_bin") or "").strip()
    if raw_codex_bin:
        codex_bin_path = Path(raw_codex_bin).expanduser()
        if not codex_bin_path.is_absolute():
            discovered = shutil.which(raw_codex_bin)
            if not discovered:
                raise ValueError(f"找不到 Codex 可执行文件: {raw_codex_bin}")
            codex_bin_path = Path(discovered)
        codex_bin_path = codex_bin_path.resolve()
        if not codex_bin_path.is_file():
            raise ValueError(f"Codex 可执行文件不存在: {codex_bin_path}")
        config["codex_bin"] = str(codex_bin_path)
    else:
        config["codex_bin"] = ""
    config["limit"] = int(config["limit"] or 0)
    config["item_id"] = str(config["item_id"] or "").strip()
    config["app_server_request_timeout_seconds"] = int(config["app_server_request_timeout_seconds"] or 90)
    config["app_server_turn_timeout_seconds"] = int(config["app_server_turn_timeout_seconds"] or 1800)
    config["max_concurrency"] = max(1, int(config["max_concurrency"] or 1))
    config["response_profile"] = EVAL_RESPONSE_PROFILE
    config["response_instructions_path"] = str(EVAL_AGENT_INSTRUCTIONS_PATH.resolve())
    config["response_instructions_sha256"] = response_instruction_digest(
        EVAL_AGENT_INSTRUCTIONS_PATH
    )
    return config


def to_runner_config(config: Dict[str, Any]) -> RunnerConfig:
    return RunnerConfig(
        langfuse_enabled=bool(config["langfuse_enabled"]),
        langfuse_public_key=str(config["langfuse_public_key"] or ""),
        langfuse_secret_key=str(config["langfuse_secret_key"] or ""),
        langfuse_host=str(config["langfuse_host"] or "https://cloud.langfuse.com"),
        dataset=str(config["dataset"] or REAL_CASES_DATASET),
        run_id=str(config["run_id"]),
        agent_version=str(config["agent_version"] or "agent-v0"),
        client=str(config["client"] or "codex-desktop"),
        skill_version=str(config["skill_version"] or "skill-v0"),
        db_snapshot_id=str(config["db_snapshot_id"] or "snapshot-v0"),
        limit=int(config["limit"]),
        item_id=str(config["item_id"] or ""),
        codex_cwd=str(config["codex_cwd"]),
        codex_model=str(config["codex_model"] or ""),
        codex_sandbox=str(config["codex_sandbox"] or ""),
        app_server_request_timeout_seconds=int(config["app_server_request_timeout_seconds"]),
        app_server_turn_timeout_seconds=int(config["app_server_turn_timeout_seconds"]),
        max_concurrency=int(config["max_concurrency"]),
        mapping_path=str(config["mapping_path"] or ""),
        mapping_timeout_seconds=int(config["mapping_timeout_seconds"]),
        items_json_path=str(config["items_json_path"] or ""),
        print_schemas=bool(config["print_schemas"]),
        response_profile=str(config["response_profile"]),
        response_instructions_path=str(config["response_instructions_path"]),
        response_instructions_sha256=str(config["response_instructions_sha256"]),
        codex_bin=str(config["codex_bin"]),
    )


def load_config(argv: Optional[List[str]]) -> RunnerConfig:
    load_env_file(PROJECT_ROOT / ".env")
    args = build_arg_parser().parse_args(argv)
    config = config_from_env()
    config = apply_arg_overrides(config, args)
    config = normalize_config(config)
    return to_runner_config(config)
