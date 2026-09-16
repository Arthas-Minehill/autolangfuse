from __future__ import annotations

from typing import Any


def run_agent_for_case(*args: Any, **kwargs: Any) -> Any:
    from aieval_runner.agent.runner import run_agent_for_case as run

    return run(*args, **kwargs)


def shutdown_agent_runtime() -> None:
    from aieval_runner.agent.runner import shutdown_agent_runtime as shutdown

    shutdown()


__all__ = ["run_agent_for_case", "shutdown_agent_runtime"]
