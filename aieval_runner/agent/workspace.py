from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from aieval_runner.core.constants import (
    EVAL_AGENT_INSTRUCTIONS_PATH,
    EVAL_RESPONSE_PROFILE,
)


FORBIDDEN_TEMPLATE_MARKERS = (
    "expected_result",
    "expected_values",
    "expected_tools",
    "tool_match_mode",
    "case_id",
)


@dataclass(frozen=True)
class AgentWorkspaceState:
    cwd: Path
    response_profile: str
    instructions_sha256: str


def _validated_template_content(template_path: Path) -> bytes:
    content = template_path.read_bytes()
    text = content.decode("utf-8")
    leaking_marker = next(
        (marker for marker in FORBIDDEN_TEMPLATE_MARKERS if marker in text),
        None,
    )
    if leaking_marker:
        raise ValueError(f"回答规范不得包含评测契约字段: {leaking_marker}")
    return content


def response_instruction_digest(
    template_path: Path = EVAL_AGENT_INSTRUCTIONS_PATH,
) -> str:
    return hashlib.sha256(_validated_template_content(template_path)).hexdigest()


def prepare_agent_workspace(
    cwd: Path,
    *,
    template_path: Path = EVAL_AGENT_INSTRUCTIONS_PATH,
    response_profile: str = EVAL_RESPONSE_PROFILE,
) -> AgentWorkspaceState:
    """将项目维护的回答规范同步到实际 Agent 工作区。"""

    content = _validated_template_content(template_path)
    resolved_cwd = cwd.resolve()
    resolved_cwd.mkdir(parents=True, exist_ok=True)
    target = resolved_cwd / "AGENTS.override.md"
    if target.exists():
        if target.read_bytes() != content:
            raise FileExistsError(f"工作区已有不同内容的 {target}")
    else:
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.write_bytes(content)
        temporary.replace(target)
    return AgentWorkspaceState(
        cwd=resolved_cwd,
        response_profile=response_profile,
        instructions_sha256=hashlib.sha256(content).hexdigest(),
    )
