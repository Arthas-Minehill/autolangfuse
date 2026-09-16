from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class LangfuseApiError(RuntimeError):
    """Langfuse Public API 调用失败。"""


@dataclass(frozen=True)
class RunnerConfig:
    langfuse_enabled: bool
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str
    dataset: str
    run_id: str
    agent_version: str
    client: str
    skill_version: str
    db_snapshot_id: str
    limit: int
    item_id: str
    codex_cwd: str
    codex_model: str
    codex_sandbox: str
    app_server_request_timeout_seconds: int
    app_server_turn_timeout_seconds: int
    max_concurrency: int
    mapping_path: str
    mapping_timeout_seconds: int
    items_json_path: str
    print_schemas: bool
    response_profile: str = ""
    response_instructions_path: str = ""
    response_instructions_sha256: str = ""
    codex_bin: str = ""


@dataclass(frozen=True)
class EvalCase:
    dataset_name: str
    dataset_item_id: str
    case_id: str
    original_request: str
    expected_output: Dict[str, Any]
    tool_contract: Dict[str, Any]
    item_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalRunContext:
    eval_run_id: str
    dataset_name: str
    agent_version: str
    client: str
    skill_version: str
    db_snapshot_id: str
    run_mode: str = "scheme_b_external_runner"
    response_profile: str = ""
    response_instructions_sha256: str = ""


@dataclass(frozen=True)
class AgentExecution:
    trace_id: str
    observation_id: Optional[str]
    output: Any
    trace_metadata: Dict[str, Any]
    mapping: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalScore:
    name: str
    value: Any
    data_type: str
    comment: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class EvaluatorResult:
    value: Any
    data_type: str
    comment: str
    metadata: Dict[str, Any] = field(default_factory=dict)
