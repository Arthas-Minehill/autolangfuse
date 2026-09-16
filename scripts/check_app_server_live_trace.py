from __future__ import annotations

import json
import sys
import tempfile
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import aieval_runner.agent.app_server.live_trace as live_trace_module  # noqa: E402
from aieval_runner.agent.app_server.live_trace import AppServerLiveTrace  # noqa: E402
from aieval_runner.core.models import EvalCase, EvalRunContext, RunnerConfig  # noqa: E402


class FakeObservation:
    _next_id = 1

    def __init__(
        self,
        *,
        parent: FakeObservation | None = None,
        fail_end_types: set[str] | None = None,
        fail_update_types: set[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.id = f"obs-{FakeObservation._next_id}"
        FakeObservation._next_id += 1
        self.parent = parent
        self.fail_end_types = fail_end_types or set()
        self.fail_update_types = fail_update_types or set()
        self.children: list[FakeObservation] = []
        self.kwargs = dict(kwargs)
        self.update_count = 0
        self.end_count = 0

    def start_observation(self, **kwargs: Any) -> FakeObservation:
        child = FakeObservation(
            parent=self,
            fail_end_types=self.fail_end_types,
            fail_update_types=self.fail_update_types,
            **kwargs,
        )
        self.children.append(child)
        return child

    def update(self, **kwargs: Any) -> FakeObservation:
        self.kwargs.update(kwargs)
        self.update_count += 1
        if self.kwargs.get("as_type") in self.fail_update_types:
            raise RuntimeError(
                f"fixture {self.kwargs.get('as_type')} update failure"
            )
        return self

    def end(self, **kwargs: Any) -> FakeObservation:
        if kwargs:
            self.kwargs.update(kwargs)
        self.end_count += 1
        if self.kwargs.get("as_type") in self.fail_end_types:
            raise RuntimeError(
                f"fixture {self.kwargs.get('as_type')} end failure"
            )
        return self


class FakeLangfuse:
    def __init__(
        self,
        *,
        fail_end_types: set[str] | None = None,
        fail_update_types: set[str] | None = None,
    ) -> None:
        self.root: FakeObservation | None = None
        self.roots: list[FakeObservation] = []
        self.trace_seed: str | None = None
        self.flush_count = 0
        self.fail_end_types = fail_end_types or set()
        self.fail_update_types = fail_update_types or set()

    def create_trace_id(self, *, seed: str) -> str:
        self.trace_seed = seed
        return "trace-live"

    def start_observation(self, **kwargs: Any) -> FakeObservation:
        self.root = FakeObservation(
            fail_end_types=self.fail_end_types,
            fail_update_types=self.fail_update_types,
            **kwargs,
        )
        self.roots.append(self.root)
        return self.root

    def flush(self) -> None:
        self.flush_count += 1


def make_config() -> RunnerConfig:
    return RunnerConfig(
        langfuse_enabled=True,
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        langfuse_host="https://langfuse.invalid",
        dataset="aieval/data_analysis",
        run_id="run-live",
        agent_version="agent-test",
        client="codex",
        skill_version="skill-test",
        db_snapshot_id="db-test",
        limit=1,
        item_id="",
        codex_cwd=str(PROJECT_ROOT),
        codex_model="gpt-5",
        codex_sandbox="workspace-write",
        app_server_request_timeout_seconds=30,
        app_server_turn_timeout_seconds=60,
        max_concurrency=1,
        mapping_path="",
        mapping_timeout_seconds=30,
        items_json_path="",
        print_schemas=False,
    )


def make_case() -> EvalCase:
    return EvalCase(
        dataset_name="aieval/data_analysis",
        dataset_item_id="item-live",
        case_id="case-live",
        original_request="请查询中文指标",
        expected_output={},
        tool_contract={},
    )


def make_context() -> EvalRunContext:
    return EvalRunContext(
        eval_run_id="eval-live",
        dataset_name="aieval/data_analysis",
        agent_version="agent-test",
        client="codex",
        skill_version="skill-test",
        db_snapshot_id="db-test",
    )


def event(
    method: str,
    *,
    thread_id: str = "thread-live",
    turn_id: str = "turn-live",
    **params: Any,
) -> dict[str, Any]:
    return {
        "method": method,
        "params": {
            "threadId": thread_id,
            "turnId": turn_id,
            **params,
        },
    }


def assert_degraded_finish() -> None:
    current_lf = FakeLangfuse()
    original_get_langfuse_sdk = live_trace_module.get_langfuse_sdk
    live_trace_module.get_langfuse_sdk = lambda _config: current_lf
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace = AppServerLiveTrace(
                case=make_case(),
                ctx=make_context(),
                config=make_config(),
                mapping_path=Path(temp_dir) / "failed-map.jsonl",
                thread_id="thread-failed",
                turn_id="turn-failed",
                session_id="session-failed",
                user_prompt="降级请求",
                event_path=Path(temp_dir) / "failed-events.jsonl",
                model="gpt-5",
            )
            trace.handle_event(
                event(
                    "thread/tokenUsage/updated",
                    thread_id="thread-failed",
                    turn_id="turn-failed",
                    tokenUsage={
                        "last": {"inputTokens": 3, "totalTokens": 3},
                        "total": {"inputTokens": 3, "totalTokens": 3},
                    },
                )
            )
            assert current_lf.root is not None
            assert current_lf.root.children == []
            trace.handle_event(
                event(
                    "item/completed",
                    thread_id="thread-failed",
                    turn_id="turn-failed",
                    item={
                        "type": "agentMessage",
                        "id": "message-degraded",
                        "phase": "final_answer",
                        "text": "降级中文回答",
                    },
                )
            )
            trace.handle_event(
                event(
                    "thread/tokenUsage/updated",
                    thread_id="thread-failed",
                    turn_id="turn-failed",
                    tokenUsage={
                        "last": {
                            "inputTokens": True,
                            "cachedInputTokens": False,
                            "outputTokens": True,
                            "reasoningOutputTokens": False,
                            "totalTokens": True,
                        },
                        "total": {
                            "inputTokens": 9_007_199_254_740_993,
                            "totalTokens": 9_007_199_254_740_993,
                        },
                    },
                )
            )
            trace.handle_event(
                event(
                    "item/started",
                    thread_id="thread-failed",
                    turn_id="turn-failed",
                    item={
                        "type": "commandExecution",
                        "id": "unfinished-command",
                        "command": "Start-Sleep 1",
                        "status": "inProgress",
                    },
                )
            )
            trace.handle_event(
                event(
                    "item/completed",
                    thread_id="thread-failed",
                    turn_id="turn-failed",
                    item={
                        "type": "commandExecution",
                        "id": "failed-command",
                        "command": "exit 17",
                        "status": "failed",
                        "aggregatedOutput": "命令失败输出",
                        "exitCode": 17,
                        "durationMs": 4,
                    },
                )
            )
            trace.handle_event(
                event(
                    "turn/completed",
                    thread_id="thread-failed",
                    turn_id="turn-failed",
                    turn={
                        "id": "turn-failed",
                        "status": "failed",
                        "error": {"message": "模型执行失败"},
                    },
                    durationMs=9_007_199_254_740_993,
                )
            )
            failed_record = trace.finish()

            failed_root = current_lf.root
            assert failed_root is not None
            failed_generations = [
                child
                for child in failed_root.children
                if child.kwargs["as_type"] == "generation"
            ]
            assert len(failed_generations) == 1
            assert "usage_details" not in failed_generations[0].kwargs
            assert failed_generations[0].end_count == 1
            unfinished_tool = failed_generations[0].children[0]
            assert unfinished_tool.kwargs["level"] == "WARNING"
            assert unfinished_tool.kwargs["metadata"]["codex.incomplete"] is True
            assert unfinished_tool.end_count == 1
            failed_tool = failed_generations[0].children[1]
            assert failed_tool.kwargs["level"] == "ERROR"
            assert failed_tool.kwargs["status_message"] == (
                "tool status: failed (exitCode=17)"
            )
            assert failed_root.kwargs["level"] == "ERROR"
            assert failed_root.kwargs["status_message"] == "模型执行失败"
            assert failed_root.kwargs["metadata"]["codex.turn_status"] == "failed"
            assert failed_root.kwargs["metadata"]["codex.turn_error"] == {
                "message": "模型执行失败"
            }
            assert failed_root.kwargs["metadata"]["codex.usage_incomplete"] is True
            assert failed_root.kwargs["metadata"]["codex.token_usage_total"] == {
                "inputTokens": "9007199254740993",
                "totalTokens": "9007199254740993",
            }
            assert failed_record["turn_duration_ms"] == "9007199254740993"

            current_lf = FakeLangfuse()
            cancelled_trace = AppServerLiveTrace(
                case=make_case(),
                ctx=make_context(),
                config=make_config(),
                mapping_path=Path(temp_dir) / "cancelled-map.jsonl",
                thread_id="thread-cancelled",
                turn_id="turn-cancelled",
                session_id="session-cancelled",
                user_prompt="取消请求",
                event_path=Path(temp_dir) / "cancelled-events.jsonl",
                model="gpt-5",
            )
            cancelled_trace.handle_event(
                event(
                    "turn/cancelled",
                    thread_id="thread-cancelled",
                    turn_id="turn-cancelled",
                    turn={
                        "id": "turn-cancelled",
                        "status": "cancelled",
                    },
                )
            )
            cancelled_trace.finish()
            assert current_lf.root is not None
            assert current_lf.root.kwargs["level"] == "WARNING"
            assert current_lf.root.kwargs["status_message"] == "turn cancelled"
    finally:
        live_trace_module.get_langfuse_sdk = original_get_langfuse_sdk


def assert_mapping_retry_is_idempotent() -> None:
    fake_lf = FakeLangfuse()
    append_attempts = 0
    original_get_langfuse_sdk = live_trace_module.get_langfuse_sdk
    original_append_jsonl = live_trace_module._append_jsonl
    live_trace_module.get_langfuse_sdk = lambda _config: fake_lf

    def flaky_append(path: Path, record: dict[str, Any]) -> None:
        nonlocal append_attempts
        append_attempts += 1
        if append_attempts == 1:
            original_append_jsonl(path, record)
            raise OSError("fixture mapping write failure")
        original_append_jsonl(path, record)

    live_trace_module._append_jsonl = flaky_append
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "retry-map.jsonl"
            trace = AppServerLiveTrace(
                case=make_case(),
                ctx=make_context(),
                config=make_config(),
                mapping_path=mapping_path,
                thread_id="thread-retry",
                turn_id="turn-retry",
                session_id="session-retry",
                user_prompt="重试请求",
                event_path=Path(temp_dir) / "retry-events.jsonl",
                model="gpt-5",
            )
            try:
                trace.finish()
            except OSError as exc:
                assert str(exc) == "fixture mapping write failure"
            else:
                raise AssertionError("首次 mapping 写入必须失败")

            assert fake_lf.root is not None
            assert fake_lf.root.end_count == 1
            assert fake_lf.flush_count == 1
            retried_record = trace.finish()
            assert fake_lf.root.end_count == 1
            assert fake_lf.flush_count == 1
            assert append_attempts == 1
            lines = mapping_path.read_text(encoding="utf-8").splitlines()
            assert len(lines) == 1
            assert json.loads(lines[0]) == retried_record
    finally:
        live_trace_module.get_langfuse_sdk = original_get_langfuse_sdk
        live_trace_module._append_jsonl = original_append_jsonl


def assert_concurrent_finish_is_idempotent() -> None:
    fake_lf = FakeLangfuse()
    original_get_langfuse_sdk = live_trace_module.get_langfuse_sdk
    live_trace_module.get_langfuse_sdk = lambda _config: fake_lf
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "concurrent-map.jsonl"
            trace = AppServerLiveTrace(
                case=make_case(),
                ctx=make_context(),
                config=make_config(),
                mapping_path=mapping_path,
                thread_id="thread-concurrent",
                turn_id="turn-concurrent",
                session_id="session-concurrent",
                user_prompt="并发收尾请求",
                event_path=Path(temp_dir) / "concurrent-events.jsonl",
                model="gpt-5",
            )
            records: list[dict[str, Any]] = []
            errors: list[BaseException] = []

            def finish_trace() -> None:
                try:
                    records.append(trace.finish())
                except BaseException as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=finish_trace) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2)
                assert not thread.is_alive()

            assert errors == []
            assert len(records) == 8
            assert all(record is records[0] for record in records)
            assert fake_lf.root is not None
            assert fake_lf.root.end_count == 1
            assert fake_lf.flush_count == 1
            assert len(mapping_path.read_text(encoding="utf-8").splitlines()) == 1
    finally:
        live_trace_module.get_langfuse_sdk = original_get_langfuse_sdk


def assert_observation_end_failure_is_stable() -> None:
    fake_lf = FakeLangfuse(fail_end_types={"generation"})
    original_get_langfuse_sdk = live_trace_module.get_langfuse_sdk
    live_trace_module.get_langfuse_sdk = lambda _config: fake_lf
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "failure-map.jsonl"
            trace = AppServerLiveTrace(
                case=make_case(),
                ctx=make_context(),
                config=make_config(),
                mapping_path=mapping_path,
                thread_id="thread-sdk-failure",
                turn_id="turn-sdk-failure",
                session_id="session-sdk-failure",
                user_prompt="SDK 异常请求",
                event_path=Path(temp_dir) / "failure-events.jsonl",
                model="gpt-5",
            )
            trace.handle_event(
                event(
                    "item/completed",
                    thread_id="thread-sdk-failure",
                    turn_id="turn-sdk-failure",
                    item={
                        "type": "agentMessage",
                        "id": "message-before-failure",
                        "phase": "commentary",
                        "text": "异常前文本",
                    },
                )
            )
            usage_event = event(
                "thread/tokenUsage/updated",
                thread_id="thread-sdk-failure",
                turn_id="turn-sdk-failure",
                tokenUsage={
                    "last": {
                        "inputTokens": 6,
                        "outputTokens": 4,
                        "totalTokens": 10,
                    },
                    "total": {
                        "inputTokens": 6,
                        "outputTokens": 4,
                        "totalTokens": 10,
                    },
                },
            )
            try:
                trace.handle_event(usage_event)
            except RuntimeError as exc:
                listener_error = exc
                assert str(exc) == "fixture generation end failure"
            else:
                raise AssertionError("Generation end 异常必须传播给 listener")

            assert fake_lf.root is not None
            generations = [
                child
                for child in fake_lf.root.children
                if child.kwargs["as_type"] == "generation"
            ]
            assert len(generations) == 1
            failed_generation = generations[0]
            assert failed_generation.end_count == 1
            assert failed_generation.kwargs["usage_details"]["total"] == 10

            trace.handle_event(
                event(
                    "item/completed",
                    thread_id="thread-sdk-failure",
                    turn_id="turn-sdk-failure",
                    item={
                        "type": "agentMessage",
                        "id": "message-after-failure",
                        "phase": "final_answer",
                        "text": "不得处理的迟到文本",
                    },
                )
            )
            trace.handle_event(
                event(
                    "thread/tokenUsage/updated",
                    thread_id="thread-sdk-failure",
                    turn_id="turn-sdk-failure",
                    tokenUsage={
                        "last": {
                            "inputTokens": 10,
                            "outputTokens": 10,
                            "totalTokens": 20,
                        },
                        "total": {
                            "inputTokens": 16,
                            "outputTokens": 14,
                            "totalTokens": 30,
                        },
                    },
                )
            )
            assert len(fake_lf.root.children) == 1
            assert failed_generation.end_count == 1
            assert failed_generation.kwargs["usage_details"]["total"] == 10

            try:
                trace.finish()
            except RuntimeError as exc:
                assert exc is listener_error
            else:
                raise AssertionError("finish 必须重新暴露已锁存的 SDK 异常")
            assert fake_lf.root.end_count == 1
            assert fake_lf.flush_count == 1
            assert len(mapping_path.read_text(encoding="utf-8").splitlines()) == 1

            try:
                trace.finish()
            except RuntimeError as exc:
                assert exc is listener_error
            else:
                raise AssertionError("重复 finish 必须稳定暴露同一 SDK 异常")
            assert fake_lf.root.end_count == 1
            assert fake_lf.flush_count == 1
    finally:
        live_trace_module.get_langfuse_sdk = original_get_langfuse_sdk


def assert_observation_update_failure_still_ends() -> None:
    fake_lf = FakeLangfuse(fail_update_types={"generation"})
    original_get_langfuse_sdk = live_trace_module.get_langfuse_sdk
    live_trace_module.get_langfuse_sdk = lambda _config: fake_lf
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "update-failure-map.jsonl"
            trace = AppServerLiveTrace(
                case=make_case(),
                ctx=make_context(),
                config=make_config(),
                mapping_path=mapping_path,
                thread_id="thread-update-failure",
                turn_id="turn-update-failure",
                session_id="session-update-failure",
                user_prompt="SDK update 异常请求",
                event_path=Path(temp_dir) / "update-failure-events.jsonl",
                model="gpt-5",
            )
            trace.handle_event(
                event(
                    "item/completed",
                    thread_id="thread-update-failure",
                    turn_id="turn-update-failure",
                    item={
                        "type": "agentMessage",
                        "id": "update-failure-message",
                        "phase": "commentary",
                        "text": "触发 update 异常",
                    },
                )
            )
            try:
                trace.handle_event(
                    event(
                        "thread/tokenUsage/updated",
                        thread_id="thread-update-failure",
                        turn_id="turn-update-failure",
                        tokenUsage={
                            "last": {
                                "inputTokens": 6,
                                "outputTokens": 4,
                                "totalTokens": 10,
                            },
                            "total": {
                                "inputTokens": 6,
                                "outputTokens": 4,
                                "totalTokens": 10,
                            },
                        },
                    )
                )
            except RuntimeError as exc:
                listener_error = exc
                assert str(exc) == "fixture generation update failure"
            else:
                raise AssertionError("Generation update 异常必须传播给 listener")

            assert fake_lf.root is not None
            generation = fake_lf.root.children[0]
            assert generation.update_count == 1
            assert generation.end_count == 0
            try:
                trace.finish()
            except ExceptionGroup as exc:
                finish_error = exc
                assert listener_error in exc.exceptions
                assert any(
                    str(error) == "fixture generation update failure"
                    for error in exc.exceptions
                )
            else:
                raise AssertionError("finish 必须聚合 update 与清理异常")
            assert generation.update_count == 2
            assert generation.end_count == 1
            assert fake_lf.root.end_count == 1
            assert fake_lf.flush_count == 1
            assert len(mapping_path.read_text(encoding="utf-8").splitlines()) == 1

            try:
                trace.finish()
            except ExceptionGroup as exc:
                assert exc is finish_error
            else:
                raise AssertionError("重复 finish 必须稳定暴露聚合异常")
            assert generation.update_count == 2
            assert generation.end_count == 1
    finally:
        live_trace_module.get_langfuse_sdk = original_get_langfuse_sdk


def assert_usage_updates_are_monotonic() -> None:
    fake_lf = FakeLangfuse()
    original_get_langfuse_sdk = live_trace_module.get_langfuse_sdk
    live_trace_module.get_langfuse_sdk = lambda _config: fake_lf
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace = AppServerLiveTrace(
                case=make_case(),
                ctx=make_context(),
                config=make_config(),
                mapping_path=Path(temp_dir) / "usage-map.jsonl",
                thread_id="thread-usage",
                turn_id="turn-usage",
                session_id="session-usage",
                user_prompt="usage 单调请求",
                event_path=Path(temp_dir) / "usage-events.jsonl",
                model="gpt-5",
            )
            trace.handle_event(
                event(
                    "item/completed",
                    thread_id="thread-usage",
                    turn_id="turn-usage",
                    item={
                        "type": "agentMessage",
                        "id": "usage-step-1",
                        "phase": "commentary",
                        "text": "第一步",
                    },
                )
            )
            first_usage = event(
                "thread/tokenUsage/updated",
                thread_id="thread-usage",
                turn_id="turn-usage",
                tokenUsage={
                    "last": {
                        "inputTokens": 6,
                        "outputTokens": 4,
                        "totalTokens": 10,
                    },
                    "total": {
                        "inputTokens": 6,
                        "outputTokens": 4,
                        "totalTokens": 10,
                    },
                },
            )
            trace.handle_event(first_usage)
            trace.handle_event(
                event(
                    "item/completed",
                    thread_id="thread-usage",
                    turn_id="turn-usage",
                    item={
                        "type": "agentMessage",
                        "id": "usage-step-2",
                        "phase": "final_answer",
                        "text": "第二步",
                    },
                )
            )
            assert fake_lf.root is not None
            generations = [
                child
                for child in fake_lf.root.children
                if child.kwargs["as_type"] == "generation"
            ]
            assert len(generations) == 2
            second_generation = generations[1]
            assert second_generation.end_count == 0
            assert "usage_details" not in second_generation.kwargs

            trace.handle_event(first_usage)
            trace.handle_event(
                event(
                    "thread/tokenUsage/updated",
                    thread_id="thread-usage",
                    turn_id="turn-usage",
                    tokenUsage={
                        "last": {
                            "inputTokens": 5,
                            "outputTokens": 4,
                            "totalTokens": 9,
                        },
                        "total": {
                            "inputTokens": 5,
                            "outputTokens": 4,
                            "totalTokens": 9,
                        },
                    },
                )
            )
            assert second_generation.end_count == 0
            assert "usage_details" not in second_generation.kwargs

            trace.handle_event(
                event(
                    "thread/tokenUsage/updated",
                    thread_id="thread-usage",
                    turn_id="turn-usage",
                    tokenUsage={
                        "last": {
                            "inputTokens": 9,
                            "outputTokens": 6,
                            "totalTokens": 15,
                        },
                        "total": {
                            "inputTokens": 15,
                            "outputTokens": 10,
                            "totalTokens": 25,
                        },
                    },
                )
            )
            assert second_generation.end_count == 1
            assert second_generation.kwargs["usage_details"]["total"] == 15
            record = trace.finish()
            assert record["token_usage_total"]["totalTokens"] == 25
    finally:
        live_trace_module.get_langfuse_sdk = original_get_langfuse_sdk


def assert_tool_without_id_is_correlated() -> None:
    fake_lf = FakeLangfuse()
    original_get_langfuse_sdk = live_trace_module.get_langfuse_sdk
    live_trace_module.get_langfuse_sdk = lambda _config: fake_lf
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace = AppServerLiveTrace(
                case=make_case(),
                ctx=make_context(),
                config=make_config(),
                mapping_path=Path(temp_dir) / "no-id-map.jsonl",
                thread_id="thread-no-id",
                turn_id="turn-no-id",
                session_id="session-no-id",
                user_prompt="无 ID 工具请求",
                event_path=Path(temp_dir) / "no-id-events.jsonl",
                model="gpt-5",
            )
            trace.handle_event(
                event(
                    "item/started",
                    thread_id="thread-no-id",
                    turn_id="turn-no-id",
                    item={
                        "type": "commandExecution",
                        "command": "Write-Output no-id",
                        "status": "inProgress",
                    },
                )
            )
            trace.handle_event(
                event(
                    "item/completed",
                    thread_id="thread-no-id",
                    turn_id="turn-no-id",
                    item={
                        "type": "commandExecution",
                        "command": "Write-Output no-id",
                        "status": "completed",
                        "aggregatedOutput": "no-id output",
                        "exitCode": 0,
                    },
                )
            )
            assert fake_lf.root is not None
            generations = [
                child
                for child in fake_lf.root.children
                if child.kwargs["as_type"] == "generation"
            ]
            assert len(generations) == 1
            assert len(generations[0].children) == 1
            tool = generations[0].children[0]
            assert tool.kwargs["name"] == "exec_command"
            assert tool.kwargs["output"] == "no-id output"
            assert tool.kwargs["metadata"]["codex.synthetic_start"] is False
            assert tool.kwargs["metadata"]["codex.orphan"] is False
            assert tool.end_count == 1
            record = trace.finish()
            assert len(record["tool_observation_ids"]) == 1
    finally:
        live_trace_module.get_langfuse_sdk = original_get_langfuse_sdk


def assert_no_phase_agent_message_becomes_final_answer() -> None:
    fake_lf = FakeLangfuse()
    original_get_langfuse_sdk = live_trace_module.get_langfuse_sdk
    live_trace_module.get_langfuse_sdk = lambda _config: fake_lf
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace = AppServerLiveTrace(
                case=make_case(),
                ctx=make_context(),
                config=make_config(),
                mapping_path=Path(temp_dir) / "fallback-map.jsonl",
                thread_id="thread-fallback",
                turn_id="turn-fallback",
                session_id="session-fallback",
                user_prompt="fallback prompt",
                event_path=Path(temp_dir) / "fallback-events.jsonl",
                model="gpt-5",
            )
            trace.handle_event(
                event(
                    "item/completed",
                    thread_id="thread-fallback",
                    turn_id="turn-fallback",
                    item={
                        "type": "agentMessage",
                        "id": "message-fallback-1",
                        "text": "intermediate answer",
                    },
                )
            )
            trace.handle_event(
                event(
                    "item/completed",
                    thread_id="thread-fallback",
                    turn_id="turn-fallback",
                    item={
                        "type": "agentMessage",
                        "id": "message-fallback-2",
                        "text": "final fallback answer",
                    },
                )
            )
            trace.finish()

            assert fake_lf.root is not None
            assert fake_lf.root.kwargs["output"] == "final fallback answer"
            generations = [
                child
                for child in fake_lf.root.children
                if child.kwargs["as_type"] == "generation"
            ]
            assert generations[-1].kwargs["output"]["content"][-1] == {
                "phase": "assistant",
                "text": "final fallback answer",
            }
    finally:
        live_trace_module.get_langfuse_sdk = original_get_langfuse_sdk


def assert_external_trace_context_is_used() -> None:
    fake_lf = FakeLangfuse()
    original_get_langfuse_sdk = live_trace_module.get_langfuse_sdk
    live_trace_module.get_langfuse_sdk = lambda _config: fake_lf
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            mapping_path = Path(temp_dir) / "external-map.jsonl"
            trace = AppServerLiveTrace(
                case=make_case(),
                ctx=make_context(),
                config=make_config(),
                mapping_path=mapping_path,
                thread_id="thread-external",
                turn_id="turn-external",
                session_id="session-external",
                user_prompt="external prompt",
                event_path=Path(temp_dir) / "external-events.jsonl",
                model="gpt-5",
                langfuse_trace_id="trace-sdk",
                langfuse_parent_observation_id="span-sdk-root",
            )
            record = trace.finish()

            assert fake_lf.trace_seed is None
            assert fake_lf.root is not None
            assert fake_lf.root.kwargs["trace_context"] == {
                "trace_id": "trace-sdk",
                "parent_span_id": "span-sdk-root",
            }
            assert fake_lf.root.kwargs["metadata"]["codex.external_trace_context"] is True
            assert fake_lf.root.kwargs["metadata"][
                "codex.external_parent_observation_id"
            ] == "span-sdk-root"
            assert record["trace_id"] == "trace-sdk"
            assert record["external_trace_context"] is True
            assert record["external_parent_observation_id"] == "span-sdk-root"
            assert json.loads(mapping_path.read_text(encoding="utf-8")) == record
    finally:
        live_trace_module.get_langfuse_sdk = original_get_langfuse_sdk


def assert_terminal_turn_ignores_late_events() -> None:
    fake_lf = FakeLangfuse()
    original_get_langfuse_sdk = live_trace_module.get_langfuse_sdk
    live_trace_module.get_langfuse_sdk = lambda _config: fake_lf
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace = AppServerLiveTrace(
                case=make_case(),
                ctx=make_context(),
                config=make_config(),
                mapping_path=Path(temp_dir) / "terminal-map.jsonl",
                thread_id="thread-terminal",
                turn_id="turn-terminal",
                session_id="session-terminal",
                user_prompt="终态请求",
                event_path=Path(temp_dir) / "terminal-events.jsonl",
                model="gpt-5",
            )
            trace.handle_event(
                event(
                    "item/completed",
                    thread_id="thread-terminal",
                    turn_id="turn-terminal",
                    item={
                        "type": "agentMessage",
                        "id": "terminal-step-1",
                        "phase": "commentary",
                        "text": "终态前步骤",
                    },
                )
            )
            trace.handle_event(
                event(
                    "thread/tokenUsage/updated",
                    thread_id="thread-terminal",
                    turn_id="turn-terminal",
                    tokenUsage={
                        "last": {
                            "inputTokens": 6,
                            "outputTokens": 4,
                            "totalTokens": 10,
                        },
                        "total": {
                            "inputTokens": 6,
                            "outputTokens": 4,
                            "totalTokens": 10,
                        },
                    },
                )
            )
            terminal_event = event(
                "turn/completed",
                thread_id="thread-terminal",
                turn_id="turn-terminal",
                turn={
                    "id": "turn-terminal",
                    "status": "completed",
                    "durationMs": 100,
                },
            )
            trace.handle_event(terminal_event)
            trace.handle_event(
                event(
                    "item/completed",
                    thread_id="thread-terminal",
                    turn_id="turn-terminal",
                    item={
                        "type": "agentMessage",
                        "id": "late-message",
                        "phase": "final_answer",
                        "text": "不得记录的迟到回答",
                    },
                )
            )
            trace.handle_event(
                event(
                    "item/completed",
                    thread_id="thread-terminal",
                    turn_id="turn-terminal",
                    item={
                        "type": "mcpToolCall",
                        "id": "late-tool",
                        "server": "late",
                        "tool": "tool",
                        "status": "completed",
                        "result": "不得记录",
                    },
                )
            )
            trace.handle_event(
                event(
                    "thread/tokenUsage/updated",
                    thread_id="thread-terminal",
                    turn_id="turn-terminal",
                    tokenUsage={
                        "last": {
                            "inputTokens": 10,
                            "outputTokens": 10,
                            "totalTokens": 20,
                        },
                        "total": {
                            "inputTokens": 16,
                            "outputTokens": 14,
                            "totalTokens": 30,
                        },
                    },
                )
            )
            trace.handle_event(terminal_event)
            trace.handle_event(
                event(
                    "turn/failed",
                    thread_id="thread-terminal",
                    turn_id="turn-terminal",
                    turn={
                        "id": "turn-terminal",
                        "status": "failed",
                        "error": {"message": "冲突终态不得覆盖"},
                    },
                )
            )

            assert fake_lf.root is not None
            generations = [
                child
                for child in fake_lf.root.children
                if child.kwargs["as_type"] == "generation"
            ]
            assert len(generations) == 1
            assert generations[0].end_count == 1
            assert generations[0].children == []
            record = trace.finish()
            assert record["token_usage_total"]["totalTokens"] == 10
            assert record["turn_status"] == "completed"
            assert "turn_error" not in record
            assert len(record["generation_observation_ids"]) == 1
            assert record["tool_observation_ids"] == []
            assert fake_lf.root.kwargs["output"] == generations[0].kwargs["output"]["content"][0]["text"]
    finally:
        live_trace_module.get_langfuse_sdk = original_get_langfuse_sdk


def assert_missing_model_marks_cost_unavailable() -> None:
    current_lf = FakeLangfuse()
    original_get_langfuse_sdk = live_trace_module.get_langfuse_sdk
    live_trace_module.get_langfuse_sdk = lambda _config: current_lf
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(make_config(), codex_model="")
            trace = AppServerLiveTrace(
                case=make_case(),
                ctx=make_context(),
                config=config,
                mapping_path=Path(temp_dir) / "missing-model-map.jsonl",
                thread_id="thread-missing-model",
                turn_id="turn-missing-model",
                session_id="session-missing-model",
                user_prompt="missing model request",
                event_path=Path(temp_dir) / "missing-model-events.jsonl",
                model="",
            )
            trace.handle_event(
                event(
                    "item/completed",
                    thread_id="thread-missing-model",
                    turn_id="turn-missing-model",
                    item={
                        "type": "agentMessage",
                        "id": "message-missing-model",
                        "text": "answer",
                    },
                )
            )
            trace.handle_event(
                event(
                    "thread/tokenUsage/updated",
                    thread_id="thread-missing-model",
                    turn_id="turn-missing-model",
                    tokenUsage={
                        "last": {
                            "inputTokens": 3,
                            "outputTokens": 4,
                            "totalTokens": 7,
                        },
                        "total": {
                            "inputTokens": 3,
                            "outputTokens": 4,
                            "totalTokens": 7,
                        },
                    },
                )
            )
            trace.finish()
    finally:
        live_trace_module.get_langfuse_sdk = original_get_langfuse_sdk

    assert current_lf.root is not None
    root_metadata = current_lf.root.kwargs["metadata"]
    assert root_metadata["codex.cost_source"] == "langfuse_model_pricing"
    assert root_metadata["codex.cost_unavailable_reason"] == "missing_model"
    generations = [
        child
        for child in current_lf.root.children
        if child.kwargs["as_type"] == "generation"
    ]
    assert len(generations) == 1
    generation_metadata = generations[0].kwargs["metadata"]
    assert generation_metadata["codex.cost_source"] == "langfuse_model_pricing"
    assert generation_metadata["codex.cost_unavailable_reason"] == "missing_model"
    assert generations[0].kwargs["usage_details"]["total"] == 7
    assert "cost_details" not in generations[0].kwargs


def main() -> int:
    FakeObservation._next_id = 1
    fake_lf = FakeLangfuse()
    original_get_langfuse_sdk = live_trace_module.get_langfuse_sdk
    live_trace_module.get_langfuse_sdk = lambda _config: fake_lf
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace = AppServerLiveTrace(
                case=make_case(),
                ctx=make_context(),
                config=make_config(),
                mapping_path=Path(temp_dir) / "trace-map.jsonl",
                thread_id="thread-live",
                turn_id="turn-live",
                session_id="session-live",
                user_prompt="请查询中文指标",
                event_path=Path(temp_dir) / "events.jsonl",
                model="gpt-5",
            )
            assert trace is not None
            trace.handle_event(
                event(
                    "turn/started",
                    turn={
                        "id": "turn-live",
                        "status": "inProgress",
                        "startedAt": "2026-06-18T01:02:03Z",
                    },
                )
            )
            trace.handle_event(
                event(
                    "item/completed",
                    turn_id="turn-other",
                    item={
                        "type": "reasoning",
                        "id": "reasoning-other",
                        "content": [{"type": "text", "text": "不得记录"}],
                    },
                )
            )
            trace.handle_event(
                event(
                    "item/started",
                    item={
                        "type": "commandExecution",
                        "id": "command-1",
                        "command": "Write-Output 中文",
                        "cwd": str(PROJECT_ROOT),
                        "status": "inProgress",
                    },
                )
            )
            trace.handle_event(
                event(
                    "item/completed",
                    item={
                        "type": "commandExecution",
                        "id": "command-1",
                        "command": "Write-Output 中文",
                        "cwd": str(PROJECT_ROOT),
                        "status": "completed",
                        "aggregatedOutput": "中文命令输出",
                        "exitCode": 0,
                        "durationMs": 12,
                    },
                )
            )
            trace.handle_event(
                event(
                    "item/completed",
                    item={
                        "type": "reasoning",
                        "id": "reasoning-1",
                        "content": [{"type": "text", "text": "先分析中文指标"}],
                    },
                )
            )
            trace.handle_event(
                event(
                    "item/completed",
                    item={
                        "type": "agentMessage",
                        "id": "message-1",
                        "phase": "commentary",
                        "text": "正在查询。",
                    },
                )
            )
            trace.handle_event(
                event(
                    "thread/tokenUsage/updated",
                    tokenUsage={
                        "last": {
                            "inputTokens": 8,
                            "cachedInputTokens": 2,
                            "outputTokens": 4,
                            "reasoningOutputTokens": 1,
                            "totalTokens": 15,
                        },
                        "total": {
                            "inputTokens": 8,
                            "cachedInputTokens": 2,
                            "outputTokens": 4,
                            "reasoningOutputTokens": 1,
                            "totalTokens": 15,
                        },
                    },
                )
            )
            trace.handle_event(
                event(
                    "item/started",
                    item={
                        "type": "mcpToolCall",
                        "id": "mcp-1",
                        "server": "metric-mcp-remote",
                        "tool": "searchMetricApp",
                        "status": "inProgress",
                        "arguments": {"name": "中文客户"},
                    },
                )
            )
            trace.handle_event(
                event(
                    "item/completed",
                    item={
                        "type": "mcpToolCall",
                        "id": "mcp-1",
                        "server": "metric-mcp-remote",
                        "tool": "searchMetricApp",
                        "status": "failed",
                        "arguments": {"name": "中文客户"},
                        "result": None,
                        "error": {"message": "中文工具失败"},
                        "durationMs": 7,
                    },
                )
            )
            trace.handle_event(
                event(
                    "item/completed",
                    item={
                        "type": "dynamicToolCall",
                        "id": "dynamic-1",
                        "name": "中文动态工具",
                        "status": "completed",
                        "arguments": {"value": 9_007_199_254_740_993},
                        "result": {
                            "value": 9_007_199_254_740_993,
                            "text": "中" * 20_010,
                        },
                        "durationMs": 5,
                    },
                )
            )
            trace.handle_event(
                event(
                    "item/completed",
                    item={
                        "type": "agentMessage",
                        "id": "message-2",
                        "phase": "final_answer",
                        "text": "最终中文回答。",
                    },
                )
            )
            trace.handle_event(
                event(
                    "thread/tokenUsage/updated",
                    tokenUsage={
                        "last": {
                            "inputTokens": 10,
                            "cachedInputTokens": 3,
                            "outputTokens": 5,
                            "reasoningOutputTokens": 2,
                            "totalTokens": 20,
                        },
                        "total": {
                            "inputTokens": 18,
                            "cachedInputTokens": 5,
                            "outputTokens": 9,
                            "reasoningOutputTokens": 3,
                            "totalTokens": 35,
                        },
                    },
                )
            )
            trace.handle_event(
                event(
                    "turn/completed",
                    turn={
                        "id": "turn-live",
                        "status": "completed",
                        "startedAt": "2026-06-18T01:02:03Z",
                        "completedAt": "2026-06-18T01:02:03.321Z",
                        "durationMs": 321,
                    },
                )
            )
            record = trace.finish()
            repeated_record = trace.finish()
            mapping_lines = (
                Path(temp_dir) / "trace-map.jsonl"
            ).read_text(encoding="utf-8").splitlines()
    finally:
        live_trace_module.get_langfuse_sdk = original_get_langfuse_sdk

    assert fake_lf.trace_seed == "eval-live:case-live:turn-live"
    assert fake_lf.root is not None
    assert fake_lf.root.kwargs["name"] == "Codex Turn"
    assert fake_lf.root.kwargs["as_type"] == "agent"
    assert fake_lf.root.kwargs["input"] == "请查询中文指标"
    assert fake_lf.root.kwargs["metadata"]["codex.model"] == "gpt-5"
    generations = [
        child
        for child in fake_lf.root.children
        if child.kwargs["as_type"] == "generation"
    ]
    assert len(generations) == 2
    assert generations[0].kwargs["input"] == "请查询中文指标"
    assert generations[0].kwargs["metadata"]["codex.step_index"] == 1
    assert generations[0].kwargs["usage_details"] == {
        "input": 8,
        "cache_read_input_tokens": 2,
        "output": 4,
        "reasoning_tokens": 1,
        "total": 15,
    }
    assert generations[0].kwargs["output"]["reasoning"] == ["先分析中文指标"]
    assert generations[0].kwargs["output"]["content"] == [
        {"phase": "commentary", "text": "正在查询。"}
    ]
    assert generations[0].end_count == 1
    assert generations[1].kwargs["metadata"]["codex.step_index"] == 2
    assert generations[1].kwargs["input"][0]["type"] == "tool_result"
    assert generations[1].kwargs["input"][0]["name"] == "exec_command"
    assert generations[1].kwargs["input"][0]["output"] == "中文命令输出"
    assert generations[1].kwargs["usage_details"]["total"] == 20
    assert generations[1].kwargs["output"]["content"] == [
        {"phase": "final_answer", "text": "最终中文回答。"}
    ]
    assert generations[1].end_count == 1
    command_tool = generations[0].children[0]
    assert command_tool.kwargs["name"] == "exec_command"
    assert command_tool.parent is generations[0]
    assert command_tool.kwargs["output"] == "中文命令输出"
    assert command_tool.kwargs["metadata"]["codex.exit_code"] == 0
    assert command_tool.end_count == 1
    mcp_tool, dynamic_tool = generations[1].children
    assert mcp_tool.kwargs["name"] == "metric-mcp-remote.searchMetricApp"
    assert mcp_tool.parent is generations[1]
    assert mcp_tool.kwargs["level"] == "ERROR"
    assert mcp_tool.kwargs["status_message"] == "中文工具失败"
    assert mcp_tool.end_count == 1
    assert dynamic_tool.kwargs["name"] == "中文动态工具"
    assert dynamic_tool.kwargs["metadata"]["codex.synthetic_start"] is True
    assert dynamic_tool.kwargs["metadata"]["codex.orphan"] is True
    assert dynamic_tool.kwargs["input"]["value"] == "9007199254740993"
    assert dynamic_tool.kwargs["output"]["value"] == "9007199254740993"
    assert dynamic_tool.kwargs["output"]["text"].startswith("中" * 100)
    assert dynamic_tool.kwargs["output"]["text"].endswith(
        "...[truncated 10 chars]"
    )
    assert dynamic_tool.end_count == 1
    assert record is repeated_record
    assert len(mapping_lines) == 1
    assert json.loads(mapping_lines[0]) == record
    assert record["mapping_source"] == "codex_app_server_live_event"
    assert {
        "eval_run_id",
        "case_id",
        "dataset_name",
        "dataset_item_id",
        "trace_id",
        "root_observation_id",
        "tool_observation_ids",
        "generation_observation_ids",
        "token_usage_total",
        "turn_duration_ms",
        "turn_key",
        "thread_id",
        "turn_id",
        "session_id",
        "created_at",
    }.issubset(record)
    assert record["token_usage_total"]["totalTokens"] == 35
    assert record["turn_duration_ms"] == 321
    assert len(record["generation_observation_ids"]) == 2
    assert [
        item["parent_observation_id"] for item in record["tool_observation_ids"]
    ] == [generations[0].id, generations[1].id, generations[1].id]
    assert fake_lf.root.kwargs["output"] == "最终中文回答。"
    assert fake_lf.root.kwargs["metadata"]["codex.turn_started_at"] == (
        "2026-06-18T01:02:03Z"
    )
    assert fake_lf.root.kwargs["metadata"]["codex.turn_completed_at"] == (
        "2026-06-18T01:02:03.321Z"
    )
    assert fake_lf.root.kwargs["metadata"]["codex.turn_duration_ms"] == 321
    assert fake_lf.root.kwargs["metadata"]["codex.turn_status"] == "completed"
    assert "usage_details" not in fake_lf.root.kwargs
    assert fake_lf.root.end_count == 1
    assert fake_lf.flush_count == 1
    assert_degraded_finish()
    assert_mapping_retry_is_idempotent()
    assert_concurrent_finish_is_idempotent()
    assert_observation_end_failure_is_stable()
    assert_observation_update_failure_still_ends()
    assert_usage_updates_are_monotonic()
    assert_tool_without_id_is_correlated()
    assert_no_phase_agent_message_becomes_final_answer()
    assert_external_trace_context_is_used()
    assert_terminal_turn_ignores_late_events()
    assert_missing_model_marks_cost_unavailable()
    print(
        json.dumps(
            {
                "status": "PASS",
                "generation_count": 2,
                "tool_types": [
                    "commandExecution",
                    "mcpToolCall",
                    "dynamicToolCall",
                ],
                "token_usage_total": 35,
                "finish_idempotent": True,
                "mapping_retry": True,
                "concurrent_finish": True,
                "sdk_failure_latched": True,
                "usage_monotonic": True,
                "tool_without_id_correlated": True,
                "no_phase_final_answer_fallback": True,
                "external_trace_context": True,
                "terminal_late_events_ignored": True,
                "missing_model_cost_diagnostic": True,
                "usage_incomplete_fallback": True,
                "utf8_and_large_integer_clip": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
