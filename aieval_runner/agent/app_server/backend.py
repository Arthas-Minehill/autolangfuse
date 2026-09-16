from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from aieval_runner.agent.common import (
    DATA_QUERY_DEVELOPER_INSTRUCTIONS,
    build_user_prompt,
    select_target_observation_id,
)
from aieval_runner.agent.app_server.live_trace import AppServerLiveTrace
from aieval_runner.agent.app_server.protocol import (
    AppServerClient,
    extract_session_id,
    extract_thread_id,
    extract_turn_id,
    latest_final_agent_message,
    mcp_tool_events,
    messages_for_turn,
    write_jsonl,
)
from aieval_runner.datasets.cases import build_trace_metadata
from aieval_runner.storage.local import mapping_path, safe_filename
from aieval_runner.core.models import AgentExecution, EvalCase, EvalRunContext, RunnerConfig
from aieval_runner.agent.workspace import AgentWorkspaceState, prepare_agent_workspace
from aieval_runner.core.constants import EVAL_AGENT_INSTRUCTIONS_PATH, EVAL_RESPONSE_PROFILE
from aieval_runner.core.validation import ensure_str


@dataclass
class AppServerRuntime:
    client: AppServerClient
    run_dir: Path
    cwd: Path
    workspace_state: AgentWorkspaceState

    def write_run_artifacts(self) -> None:
        write_jsonl(self.run_dir / "app-server-messages.jsonl", self.client.messages_snapshot())
        stderr = self.client.stderr_snapshot()
        if stderr:
            (self.run_dir / "app-server-stderr.txt").write_text(stderr, encoding="utf-8")

    def close(self) -> None:
        errors: list[Exception] = []
        try:
            self.write_run_artifacts()
        except Exception as exc:
            errors.append(exc)
        try:
            self.client.close()
        except Exception as exc:
            errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            raise ExceptionGroup("app-server runtime shutdown failed", errors)


_runtime: Optional[AppServerRuntime] = None


def _runtime_key(config: RunnerConfig, ctx: EvalRunContext) -> tuple[str, str]:
    return (str(Path(config.codex_cwd).resolve()), ctx.eval_run_id)


_active_runtime_key: Optional[tuple[str, str]] = None


def _event_log_filename(case: EvalCase) -> str:
    case_part = safe_filename(case.case_id)
    item_part = safe_filename(case.dataset_item_id)
    if item_part and item_part != case_part:
        return f"{case_part}__{item_part}.jsonl"
    return f"{case_part}.jsonl"


def _ensure_runtime(config: RunnerConfig, ctx: EvalRunContext) -> AppServerRuntime:
    global _runtime, _active_runtime_key
    key = _runtime_key(config, ctx)
    if _runtime is not None and _active_runtime_key == key:
        return _runtime
    shutdown_codex_app_server()

    cwd = Path(config.codex_cwd).resolve()
    workspace_state = prepare_agent_workspace(
        cwd,
        template_path=(
            Path(config.response_instructions_path)
            if config.response_instructions_path
            else EVAL_AGENT_INSTRUCTIONS_PATH
        ),
        response_profile=config.response_profile or EVAL_RESPONSE_PROFILE,
    )
    if (
        config.response_instructions_sha256
        and workspace_state.instructions_sha256
        != config.response_instructions_sha256
    ):
        raise RuntimeError("回答规范模板在配置加载后发生变化，请重新启动评测")
    run_dir = mapping_path(config, ctx).parent
    run_dir.mkdir(parents=True, exist_ok=True)
    client = AppServerClient(
        cwd=cwd,
        request_timeout_seconds=config.app_server_request_timeout_seconds,
        codex_bin=config.codex_bin,
    )
    client.start()
    initialize_result = client.initialize()
    (run_dir / "app-server-initialize.json").write_text(
        json.dumps(initialize_result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    _runtime = AppServerRuntime(
        client=client,
        run_dir=run_dir,
        cwd=cwd,
        workspace_state=workspace_state,
    )
    _active_runtime_key = key
    return _runtime


def shutdown_codex_app_server() -> None:
    global _runtime, _active_runtime_key
    runtime = _runtime
    _runtime = None
    _active_runtime_key = None
    if runtime is not None:
        runtime.close()


def _append_sidecar(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str, separators=(",", ":")) + "\n")


def _sidecar_record(
    *,
    config: RunnerConfig,
    ctx: EvalRunContext,
    case: EvalCase,
    target_mapping_path: Path,
    session_id: str,
    thread_id: str,
    turn_id: str,
    user_prompt: str,
    event_path: Path,
    langfuse_trace_id: Optional[str] = None,
    langfuse_parent_observation_id: Optional[str] = None,
) -> Dict[str, Any]:
    record = {
        "session_id": session_id,
        "thread_id": thread_id,
        "turn_id": turn_id,
        "eval_run_id": ctx.eval_run_id,
        "case_id": case.case_id,
        "dataset_name": ctx.dataset_name,
        "dataset_item_id": case.dataset_item_id,
        "mapping_path": str(target_mapping_path),
        "cwd": str(Path(config.codex_cwd).resolve()),
        "user_prompt": user_prompt,
        "event_path": str(event_path),
        "response_profile": ctx.response_profile,
        "response_instructions_sha256": ctx.response_instructions_sha256,
        "trace_owner": "langfuse_experiment_runner"
        if langfuse_trace_id
        else "aieval_app_server_live_event",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if langfuse_trace_id:
        record["langfuse_trace_id"] = str(langfuse_trace_id)
    if langfuse_parent_observation_id:
        record["langfuse_parent_observation_id"] = str(langfuse_parent_observation_id)
    return record


def _tracing_infrastructure_error(stage: str, error: Exception) -> RuntimeError:
    wrapped = RuntimeError(f"app-server live tracing {stage} failed: {error}")
    wrapped.__cause__ = error
    return wrapped


def _unique_leaf_errors(
    errors: list[BaseException],
    *,
    seen: Optional[set[int]] = None,
) -> list[Exception]:
    unique: list[Exception] = []
    seen_ids = seen if seen is not None else set()
    for error in errors:
        if isinstance(error, BaseExceptionGroup):
            unique.extend(
                _unique_leaf_errors(
                    list(error.exceptions),
                    seen=seen_ids,
                )
            )
            continue
        if not isinstance(error, Exception) or id(error) in seen_ids:
            continue
        seen_ids.add(id(error))
        unique.append(error)
    return unique


def run_codex_app_server(
    case: EvalCase,
    ctx: EvalRunContext,
    config: RunnerConfig,
    *,
    langfuse_trace_id: Optional[str] = None,
    langfuse_parent_observation_id: Optional[str] = None,
) -> AgentExecution:
    runtime = _ensure_runtime(config, ctx)
    ctx = replace(
        ctx,
        response_profile=runtime.workspace_state.response_profile,
        response_instructions_sha256=runtime.workspace_state.instructions_sha256,
    )
    target_mapping_path = mapping_path(config, ctx)
    run_dir = target_mapping_path.parent
    events_dir = run_dir / "app-server-events"
    event_path = events_dir / _event_log_filename(case)
    sidecar_path = run_dir / "app-server-eval-context.jsonl"

    thread_response = runtime.client.start_thread(
        cwd=runtime.cwd,
        sandbox=config.codex_sandbox,
        model=config.codex_model,
        developer_instructions=DATA_QUERY_DEVELOPER_INSTRUCTIONS,
    )
    thread_id = extract_thread_id(thread_response)
    session_id = extract_session_id(thread_response)

    #user_prompt = build_user_prompt(case)
    user_prompt = case.original_request
    turn_response = runtime.client.start_turn(
        thread_id=thread_id,
        cwd=runtime.cwd,
        prompt=user_prompt,
        sandbox=config.codex_sandbox,
        model=config.codex_model,
    )
    turn_id = extract_turn_id(turn_response)
    runtime.client.register_turn_event_log(thread_id=thread_id, turn_id=turn_id, path=event_path)

    live_trace: Optional[AppServerLiveTrace] = None
    listener_registration_attempted = False
    tracing_errors: list[Exception] = []
    cleanup_errors: list[Exception] = []
    turn_error: Optional[Exception] = None
    mapping: Optional[Dict[str, Any]] = None

    try:
        _append_sidecar(
            sidecar_path,
            _sidecar_record(
                config=config,
                ctx=ctx,
                case=case,
                target_mapping_path=target_mapping_path,
                session_id=session_id,
                thread_id=thread_id,
                turn_id=turn_id,
                user_prompt=user_prompt,
                event_path=event_path,
                langfuse_trace_id=langfuse_trace_id,
                langfuse_parent_observation_id=langfuse_parent_observation_id,
            ),
        )
    except Exception as exc:
        tracing_errors.append(_tracing_infrastructure_error("sidecar write", exc))
    try:
        live_trace = AppServerLiveTrace(
            case=case,
            ctx=ctx,
            config=config,
            mapping_path=target_mapping_path,
            thread_id=thread_id,
            turn_id=turn_id,
            session_id=session_id,
            user_prompt=user_prompt,
            event_path=event_path,
            model=config.codex_model,
            langfuse_trace_id=langfuse_trace_id,
            langfuse_parent_observation_id=langfuse_parent_observation_id,
        )
    except Exception as exc:
        tracing_errors.append(_tracing_infrastructure_error("initialization", exc))
    if live_trace is not None:
        listener_registration_attempted = True
        try:
            runtime.client.register_turn_event_listener(
                thread_id=thread_id,
                turn_id=turn_id,
                callback=live_trace.handle_event,
            )
        except Exception as exc:
            tracing_errors.append(
                _tracing_infrastructure_error("listener registration", exc)
            )

    try:
        completed = runtime.client.wait_for_turn_completed(
            thread_id=thread_id,
            turn_id=turn_id,
            timeout_seconds=config.app_server_turn_timeout_seconds,
        )
    except Exception as exc:
        turn_error = exc
    finally:
        try:
            runtime.client.unregister_turn_event_log(
                thread_id=thread_id,
                turn_id=turn_id,
            )
        except Exception as exc:
            cleanup_errors.append(exc)
        if listener_registration_attempted:
            try:
                tracing_errors.extend(
                    runtime.client.unregister_turn_event_listener(
                        thread_id=thread_id,
                        turn_id=turn_id,
                    )
                )
            except Exception as exc:
                tracing_errors.append(
                    _tracing_infrastructure_error("listener cleanup", exc)
                )
        if live_trace is not None:
            try:
                mapping = live_trace.finish()
            except Exception as exc:
                tracing_errors.append(exc)

    has_tracing_errors = bool(tracing_errors)
    has_cleanup_errors = bool(cleanup_errors)
    if turn_error is not None and not has_tracing_errors and not has_cleanup_errors:
        raise turn_error
    failures = _unique_leaf_errors(
        [
            *([turn_error] if turn_error is not None else []),
            *tracing_errors,
            *cleanup_errors,
        ]
    )
    if failures:
        if turn_error is not None:
            message = "app-server turn failed with tracing or cleanup errors"
        elif has_tracing_errors and has_cleanup_errors:
            message = "app-server tracing and cleanup failed"
        elif has_tracing_errors:
            message = "app-server live tracing failed"
        else:
            message = "app-server cleanup failed"
        raise ExceptionGroup(message, failures)
    if mapping is None:
        raise RuntimeError("app-server live tracing returned no mapping")

    case_events = messages_for_turn(runtime.client.messages_snapshot(), thread_id=thread_id, turn_id=turn_id)

    trace_id = ensure_str(mapping.get("trace_id"), field_name="mapping.trace_id")
    observation_id = select_target_observation_id(mapping, case) or mapping.get("root_observation_id")
    answer = latest_final_agent_message(case_events)
    stderr_tail = runtime.client.stderr_snapshot()[-4000:]

    return AgentExecution(
        trace_id=trace_id,
        observation_id=str(observation_id) if observation_id else None,
        output={
            "runner": "codex_app_server",
            "answer": answer,
            "thread_id": thread_id,
            "turn_id": turn_id,
            "session_id": session_id,
            "event_path": str(event_path),
            "response_profile": ctx.response_profile,
            "response_instructions_sha256": ctx.response_instructions_sha256,
            "sidecar_path": str(sidecar_path),
            "mapping_path": str(target_mapping_path),
            "mapping_source": mapping.get("mapping_source"),
            "mcp_tool_call_event_count": len(mcp_tool_events(case_events)),
            "turn_completed": completed,
            "stderr_tail": stderr_tail,
        },
        trace_metadata=build_trace_metadata(case, ctx),
        mapping=mapping,
    )
