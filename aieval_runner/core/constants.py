from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_WORKSPACE_ROOT = Path(tempfile.gettempdir()) / "aieval-agent-workspace"
EVAL_AGENT_INSTRUCTIONS_PATH = PROJECT_ROOT / "config" / "eval-agent" / "AGENTS.override.md"
EVAL_RESPONSE_PROFILE = "complete_data_result_v1"
REAL_CASES_DATASET = "aieval/real_cases"
SCHEMA_VERSION = "aieval-real-cases-v2"
MAX_EXPECTED_TOOLS = 50
MAX_TOOL_NAME_CHARS = 500
MAX_INPUT_CHARS = 100_000
MAX_CASE_ID_CHARS = 500
MAX_DATASET_ITEM_ID_CHARS = 500


RUN_CONFIG: Dict[str, Any] = {
    "dataset": REAL_CASES_DATASET,
    "run_id": "",
    "agent_version": "agent-v0",
    "client": "codex-desktop",
    "skill_version": "skill-v0",
    "db_snapshot_id": "snapshot-v0",
    "limit": 0,
    "item_id": "",
    # Agent 默认在不含题库与 Gold 的空目录运行，防止通过仓库文件泄漏答案。
    "codex_cwd": str(AGENT_WORKSPACE_ROOT),
    "codex_model": "",
    "codex_sandbox": "workspace-write",
    "app_server_request_timeout_seconds": 90,
    "app_server_turn_timeout_seconds": 1800,
    "max_concurrency": 4,
    "mapping_path": "",
    "mapping_timeout_seconds": 90,
    "items_json_path": "",
    "print_schemas": False,
}


INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "input": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_INPUT_CHARS,
        },
    },
    "required": ["input"],
    "additionalProperties": True,
}


EXPECTED_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "expected_result": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_INPUT_CHARS,
        },
        "expected_values": {
            "type": "object",
        },
    },
    "required": ["expected_result", "expected_values"],
    "additionalProperties": True,
}


METADATA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "case_id": {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_CASE_ID_CHARS,
        },
        "expected_tools": {
            "type": "array",
            "items": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_TOOL_NAME_CHARS,
            },
            "minItems": 1,
            "maxItems": MAX_EXPECTED_TOOLS,
            "uniqueItems": True,
        },
        "tool_match_mode": {"type": "string", "enum": ["any", "all"]},
    },
    "required": ["case_id", "expected_tools", "tool_match_mode"],
    "additionalProperties": True,
}


TRACE_METADATA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "string"},
        "eval_run_id": {"type": "string"},
        "case_id": {"type": "string"},
        "dataset_name": {"type": "string"},
        "dataset_item_id": {"type": "string"},
        "agent_version": {"type": "string"},
        "client": {"type": "string"},
        "skill_version": {"type": "string"},
        "db_snapshot_id": {"type": "string"},
        "run_mode": {"type": "string"},
        "response_profile": {"type": "string"},
        "response_instructions_sha256": {"type": "string"},
    },
    "required": [
        "schema_version",
        "eval_run_id",
        "case_id",
        "dataset_name",
        "dataset_item_id",
        "agent_version",
        "client",
        "skill_version",
        "db_snapshot_id",
        "run_mode",
        "response_profile",
        "response_instructions_sha256",
    ],
    "additionalProperties": False,
}


SCORE_METADATA_KEYS = [
    "schema_version",
    "eval_run_id",
    "case_id",
    "dataset_name",
    "dataset_item_id",
    "trace_id",
    "observation_id",
    "root_observation_id",
    "target_observation_id",
    "target_observation_missing",
    "agent_version",
    "client",
    "skill_version",
    "db_snapshot_id",
]
