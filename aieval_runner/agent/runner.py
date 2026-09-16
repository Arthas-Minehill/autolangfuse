from __future__ import annotations

from aieval_runner.agent.app_server.backend import run_codex_app_server, shutdown_codex_app_server
from aieval_runner.core.models import AgentExecution, EvalCase, EvalRunContext, RunnerConfig


def run_agent_for_case(
    case: EvalCase,
    ctx: EvalRunContext,
    *,
    config: RunnerConfig,
    langfuse_trace_id: str | None = None,
    langfuse_parent_observation_id: str | None = None,
) -> AgentExecution:
    return run_codex_app_server(
        case,
        ctx,
        config,
        langfuse_trace_id=langfuse_trace_id,
        langfuse_parent_observation_id=langfuse_parent_observation_id,
    )


def shutdown_agent_runtime() -> None:
    shutdown_codex_app_server()
