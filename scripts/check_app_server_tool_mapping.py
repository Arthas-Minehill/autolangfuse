from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import aieval_runner.agent.app_server.protocol as app_server_protocol  # noqa: E402
from aieval_runner.agent.app_server.protocol import (  # noqa: E402
    AppServerClient,
    latest_final_agent_message,
)
from aieval_runner.agent.app_server.trace_mapping import (  # noqa: E402
    _clip,
    _completed_tool_items,
    _tool_name,
    _tool_output,
)


def event(
    method: str,
    *,
    thread_id: str,
    turn_id: str,
    item: dict[str, Any],
) -> dict[str, Any]:
    return {
        "method": method,
        "params": {
            "threadId": thread_id,
            "turnId": turn_id,
            "item": item,
        },
    }


def assert_notification_preserved(
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    assert actual["method"] == expected["method"]
    assert actual["params"] == expected["params"]
    received_at = actual.get("_received_at")
    assert isinstance(received_at, str)
    assert datetime.fromisoformat(received_at).utcoffset() == timedelta(0)


class FakeProcess:
    def __init__(self, stdout: Iterable[str]) -> None:
        self.stdout = stdout
        self.stderr: list[str] = []

    @staticmethod
    def poll() -> None:
        return None


class BlockingTerminateProcess(FakeProcess):
    def __init__(self) -> None:
        super().__init__([])
        self.terminate_entered = threading.Event()
        self.release_terminate = threading.Event()
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminate_entered.set()
        assert self.release_terminate.wait(timeout=1)
        self.terminated = True

    @staticmethod
    def wait(timeout: float) -> int:
        assert timeout == 5
        return 0

    def kill(self) -> None:
        self.killed = True


class FailingCloseHandle:
    def __init__(self) -> None:
        self.close_called = False

    def close(self) -> None:
        self.close_called = True
        raise RuntimeError("log close fixture error")


def main() -> int:
    assert latest_final_agent_message(
        [
            event(
                "item/started",
                thread_id="thread-protocol",
                turn_id="turn-protocol",
                item={
                    "type": "agentMessage",
                    "text": "draft answer",
                },
            ),
            event(
                "item/completed",
                thread_id="thread-protocol",
                turn_id="turn-protocol",
                item={
                    "type": "agentMessage",
                    "text": "completed fallback answer",
                },
            ),
            event(
                "item/completed",
                thread_id="thread-protocol",
                turn_id="turn-protocol",
                item={
                    "type": "agentMessage",
                    "phase": "final_answer",
                    "text": "explicit final answer",
                },
            ),
        ]
    ) == "explicit final answer"
    thread_id = "thread-utf8"
    turn_id = "turn-utf8"
    other_turn_id = "turn-other"
    history_ready = threading.Event()
    release_realtime = threading.Event()
    callback_entered = threading.Event()
    release_callback = threading.Event()
    historical_rows = [
        event(
            "item/started",
            thread_id=thread_id,
            turn_id=turn_id,
            item={"type": "mcpToolCall", "id": "history-1"},
        ),
        event(
            "item/completed",
            thread_id=thread_id,
            turn_id=other_turn_id,
            item={"type": "mcpToolCall", "id": "other-turn"},
        ),
    ]
    realtime_rows = [
        event(
            "item/completed",
            thread_id=thread_id,
            turn_id=turn_id,
            item={"type": "mcpToolCall", "id": "realtime-1"},
        ),
        event(
            "item/completed",
            thread_id=thread_id,
            turn_id=turn_id,
            item={"type": "agentMessage", "id": "realtime-2"},
        ),
    ]

    def stdout_lines() -> Iterable[str]:
        for row in historical_rows:
            yield json.dumps(row, ensure_ascii=False) + "\n"
        history_ready.set()
        assert release_realtime.wait(timeout=5)
        for row in realtime_rows:
            yield json.dumps(row, ensure_ascii=False) + "\n"

    with tempfile.TemporaryDirectory() as temp_dir:
        client = AppServerClient(cwd=Path(temp_dir), request_timeout_seconds=1)
        client._proc = FakeProcess(stdout_lines())  # type: ignore[assignment]
        reader = threading.Thread(target=client._read_stdout)
        reader.start()
        assert history_ready.wait(timeout=5)

        received: list[dict[str, Any]] = []

        def listener(message: dict[str, Any]) -> None:
            received.append(message)
            if message["params"]["item"]["id"] == "realtime-1":
                callback_entered.set()
                assert release_callback.wait(timeout=5)
                raise RuntimeError("listener fixture error")

        client.register_turn_event_listener(
            thread_id=thread_id,
            turn_id=turn_id,
            callback=listener,
        )
        release_realtime.set()
        assert callback_entered.wait(timeout=5)
        reader.join(timeout=5)
        assert not reader.is_alive()
        release_callback.set()
        errors = client.unregister_turn_event_listener(
            thread_id=thread_id,
            turn_id=turn_id,
        )
        closed_received: list[dict[str, Any]] = []
        client.register_turn_event_listener(
            thread_id=thread_id,
            turn_id=turn_id,
            callback=closed_received.append,
        )
        client._proc = None
        assert client.close() is None
        assert client.unregister_turn_event_listener(
            thread_id=thread_id,
            turn_id=turn_id,
        ) == []

    assert len(errors) == 1
    assert str(errors[0]) == "listener fixture error"
    received_ids = [message["params"]["item"]["id"] for message in received]
    assert received_ids == ["history-1", "realtime-1", "realtime-2"]
    assert [message["params"]["item"]["id"] for message in closed_received] == [
        "history-1",
        "realtime-1",
        "realtime-2",
    ]
    expected_rows = [historical_rows[0], *realtime_rows]
    for message, expected in zip(received, expected_rows):
        assert_notification_preserved(message, expected)

    isolation_row = event(
        "item/completed",
        thread_id=thread_id,
        turn_id=turn_id,
        item={
            "type": "mcpToolCall",
            "id": "isolation",
            "payload": {"value": "original"},
        },
    )
    mutation_done = threading.Event()

    def mutating_listener(message: dict[str, Any]) -> None:
        message["method"] = "mutated"
        message["params"]["item"]["payload"]["value"] = "mutated"
        mutation_done.set()

    with tempfile.TemporaryDirectory() as temp_dir:
        isolation_path = Path(temp_dir) / "isolation.jsonl"
        isolation_client = AppServerClient(
            cwd=Path(temp_dir),
            request_timeout_seconds=1,
        )
        isolation_client.register_turn_event_log(
            thread_id=thread_id,
            turn_id=turn_id,
            path=isolation_path,
        )
        isolation_client.register_turn_event_listener(
            thread_id=thread_id,
            turn_id=turn_id,
            callback=mutating_listener,
        )
        isolation_client._proc = FakeProcess(  # type: ignore[assignment]
            [json.dumps(isolation_row, ensure_ascii=False) + "\n"]
        )
        isolation_client._read_stdout()
        assert mutation_done.wait(timeout=1)
        assert isolation_client.unregister_turn_event_listener(
            thread_id=thread_id,
            turn_id=turn_id,
        ) == []
        isolation_client.unregister_turn_event_log(
            thread_id=thread_id,
            turn_id=turn_id,
        )

        snapshot_message = isolation_client.messages_snapshot()[0]
        assert_notification_preserved(snapshot_message, isolation_row)
        event_message = isolation_client.wait_for_event(
            "item/completed",
            timeout_seconds=1,
        )
        assert_notification_preserved(event_message, isolation_row)
        persisted_message = json.loads(
            isolation_path.read_text(encoding="utf-8").strip()
        )
        assert_notification_preserved(persisted_message, isolation_row)

        second_listener_received: list[dict[str, Any]] = []
        isolation_client.register_turn_event_listener(
            thread_id=thread_id,
            turn_id=turn_id,
            callback=second_listener_received.append,
        )
        assert isolation_client.unregister_turn_event_listener(
            thread_id=thread_id,
            turn_id=turn_id,
        ) == []
        assert len(second_listener_received) == 1
        assert_notification_preserved(second_listener_received[0], isolation_row)

    blocking_callback_entered = threading.Event()
    release_blocking_callback = threading.Event()
    blocking_callback_exited = threading.Event()

    def blocking_listener(_message: dict[str, Any]) -> None:
        blocking_callback_entered.set()
        release_blocking_callback.wait()
        blocking_callback_exited.set()
        raise RuntimeError("late listener error")

    original_listener_timeout = getattr(
        app_server_protocol,
        "TURN_EVENT_LISTENER_JOIN_TIMEOUT_SECONDS",
        None,
    )
    app_server_protocol.TURN_EVENT_LISTENER_JOIN_TIMEOUT_SECONDS = 0.05
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            blocking_client = AppServerClient(
                cwd=Path(temp_dir),
                request_timeout_seconds=1,
            )
            blocking_client.register_turn_event_listener(
                thread_id=thread_id,
                turn_id=turn_id,
                callback=blocking_listener,
            )
            blocking_subscription = blocking_client._turn_event_listeners[  # noqa: SLF001
                (thread_id, turn_id)
            ]
            blocking_client._proc = FakeProcess(  # type: ignore[assignment]
                [
                    json.dumps(realtime_rows[0], ensure_ascii=False) + "\n",
                ]
            )
            blocking_client._read_stdout()
            assert blocking_callback_entered.wait(timeout=1)

            unregister_result: list[list[Exception]] = []

            def unregister_blocking_listener() -> None:
                unregister_result.append(
                    blocking_client.unregister_turn_event_listener(
                        thread_id=thread_id,
                        turn_id=turn_id,
                    )
                )

            unregister_thread = threading.Thread(
                target=unregister_blocking_listener,
                daemon=True,
            )
            started_at = time.monotonic()
            unregister_thread.start()
            unregister_thread.join(timeout=0.5)
            returned_before_release = not unregister_thread.is_alive()
            elapsed = time.monotonic() - started_at
            release_blocking_callback.set()
            unregister_thread.join(timeout=1)
            assert not unregister_thread.is_alive()
            assert blocking_callback_exited.wait(timeout=1)
            terminal_errors = blocking_subscription.close()

        assert returned_before_release
        assert elapsed < 0.5
        assert len(unregister_result) == 1
        timeout_errors = [
            error for error in unregister_result[0] if isinstance(error, TimeoutError)
        ]
        assert len(timeout_errors) == 1
        assert str(timeout_errors[0]) == "turn event listener did not stop"
        assert len(terminal_errors) == 1
        assert isinstance(terminal_errors[0], TimeoutError)
        assert str(terminal_errors[0]) == "turn event listener did not stop"

        close_callback_entered = threading.Event()
        release_close_callback = threading.Event()
        close_callback_exited = threading.Event()
        close_error_callback_called = threading.Event()

        def close_blocking_listener(_message: dict[str, Any]) -> None:
            close_callback_entered.set()
            release_close_callback.wait()
            close_callback_exited.set()

        def close_error_listener(_message: dict[str, Any]) -> None:
            close_error_callback_called.set()
            raise RuntimeError("close fixture error")

        with tempfile.TemporaryDirectory() as temp_dir:
            close_client = AppServerClient(
                cwd=Path(temp_dir),
                request_timeout_seconds=1,
            )
            close_client.register_turn_event_listener(
                thread_id=thread_id,
                turn_id=turn_id,
                callback=close_blocking_listener,
            )
            close_client.register_turn_event_listener(
                thread_id=thread_id,
                turn_id="turn-close-error",
                callback=close_error_listener,
            )
            close_error_row = event(
                "item/completed",
                thread_id=thread_id,
                turn_id="turn-close-error",
                item={"type": "agentMessage", "id": "close-error"},
            )
            close_client._proc = FakeProcess(  # type: ignore[assignment]
                [
                    json.dumps(realtime_rows[0], ensure_ascii=False) + "\n",
                    json.dumps(close_error_row, ensure_ascii=False) + "\n",
                ]
            )
            close_client._read_stdout()
            assert close_callback_entered.wait(timeout=1)
            assert close_error_callback_called.wait(timeout=1)

            close_process = BlockingTerminateProcess()
            close_client._proc = close_process  # type: ignore[assignment]
            failing_log_handle = FailingCloseHandle()
            close_client._turn_event_logs[  # type: ignore[assignment]  # noqa: SLF001
                (thread_id, "turn-log-close-error")
            ] = failing_log_handle
            close_returned: list[None] = []
            close_exceptions: list[BaseException] = []

            def close_blocking_client() -> None:
                try:
                    close_returned.append(close_client.close())
                except BaseException as exc:
                    close_exceptions.append(exc)

            close_thread = threading.Thread(
                target=close_blocking_client,
                daemon=True,
            )
            close_thread.start()
            assert close_process.terminate_entered.wait(timeout=1)

            accepted_during_close = False
            try:
                close_client.register_turn_event_listener(
                    thread_id=thread_id,
                    turn_id="turn-during-close",
                    callback=lambda _message: None,
                )
                accepted_during_close = True
            except RuntimeError as exc:
                assert str(exc) == "app-server client is closed"

            close_process.release_terminate.set()
            close_thread.join(timeout=1)
            assert not close_thread.is_alive()
            if accepted_during_close:
                close_client.unregister_turn_event_listener(
                    thread_id=thread_id,
                    turn_id="turn-during-close",
                )

            accepted_after_close = False
            try:
                close_client.register_turn_event_listener(
                    thread_id=thread_id,
                    turn_id="turn-after-close",
                    callback=lambda _message: None,
                )
                accepted_after_close = True
            except RuntimeError as exc:
                assert str(exc) == "app-server client is closed"
            if accepted_after_close:
                close_client.unregister_turn_event_listener(
                    thread_id=thread_id,
                    turn_id="turn-after-close",
                )

            release_close_callback.set()
            assert close_callback_exited.wait(timeout=1)

        assert accepted_during_close is False
        assert accepted_after_close is False
        assert failing_log_handle.close_called is True
        assert close_process.terminated is True
        assert close_process.killed is False
        assert close_client._proc is None  # noqa: SLF001
        assert close_returned == []
        assert len(close_exceptions) == 1
        close_exception = close_exceptions[0]
        assert isinstance(close_exception, ExceptionGroup)
        assert str(close_exception) == (
            "app-server client shutdown failed (3 sub-exceptions)"
        )
        close_timeout_errors = [
            error for error in close_exception.exceptions if isinstance(error, TimeoutError)
        ]
        assert len(close_timeout_errors) == 1
        assert str(close_timeout_errors[0]) == "turn event listener did not stop"
        close_callback_errors = [
            error
            for error in close_exception.exceptions
            if isinstance(error, RuntimeError)
            and str(error) == "close fixture error"
        ]
        assert len(close_callback_errors) == 1
        close_log_errors = [
            error
            for error in close_exception.exceptions
            if isinstance(error, RuntimeError)
            and str(error) == "log close fixture error"
        ]
        assert len(close_log_errors) == 1

        original_popen = app_server_protocol.subprocess.Popen
        start_called = False

        def unexpected_popen(*_args: Any, **_kwargs: Any) -> None:
            nonlocal start_called
            start_called = True
            raise AssertionError("closed client must not start a process")

        app_server_protocol.subprocess.Popen = unexpected_popen  # type: ignore[assignment]
        try:
            closed_start_rejected = False
            try:
                close_client.start()
            except RuntimeError as exc:
                assert str(exc) == "app-server client is closed"
                closed_start_rejected = True
        finally:
            app_server_protocol.subprocess.Popen = original_popen

        closed_log_rejected = False
        closed_log_path = Path(temp_dir) / "closed.jsonl"
        try:
            close_client.register_turn_event_log(
                thread_id=thread_id,
                turn_id="turn-closed-log",
                path=closed_log_path,
            )
        except RuntimeError as exc:
            assert str(exc) == "app-server client is closed"
            closed_log_rejected = True

        closed_listener_rejected = False
        try:
            close_client.register_turn_event_listener(
                thread_id=thread_id,
                turn_id="turn-closed-listener",
                callback=lambda _message: None,
            )
        except RuntimeError as exc:
            assert str(exc) == "app-server client is closed"
            closed_listener_rejected = True

        assert closed_start_rejected is True
        assert start_called is False
        assert closed_log_rejected is True
        assert closed_log_path.exists() is False
        assert closed_listener_rejected is True
    finally:
        if original_listener_timeout is None:
            del app_server_protocol.TURN_EVENT_LISTENER_JOIN_TIMEOUT_SECONDS
        else:
            app_server_protocol.TURN_EVENT_LISTENER_JOIN_TIMEOUT_SECONDS = (
                original_listener_timeout
            )

    large_integer = 2013607075345784834
    success_arguments = {
        "query": "查询中文客户",
        "市场": "美国",
        "metric_app_id": large_integer,
    }
    success_result = {
        "content": [{"type": "text", "text": "工具结果正常：活跃客户 104 个"}],
    }
    failed_error = {"message": "工具错误：中文参数无效"}
    pre_hook_rows = [
        event(
            "item/started",
            thread_id=thread_id,
            turn_id=turn_id,
            item={
                "type": "mcpToolCall",
                "id": "call-success",
                "server": "metric-mcp-remote",
                "tool": "searchMetricApp",
                "status": "inProgress",
                "arguments": success_arguments,
            },
        ),
        event(
            "item/completed",
            thread_id=thread_id,
            turn_id=turn_id,
            item={
                "type": "mcpToolCall",
                "id": "call-success",
                "server": "metric-mcp-remote",
                "tool": "searchMetricApp",
                "status": "completed",
                "arguments": success_arguments,
                "result": success_result,
                "error": None,
                "durationMs": 123,
            },
        ),
        event(
            "item/completed",
            thread_id=thread_id,
            turn_id=turn_id,
            item={
                "type": "mcpToolCall",
                "id": "call-error",
                "server": "metric-mcp-remote",
                "tool": "searchBizMetric",
                "status": "failed",
                "arguments": {"name": "中文客户"},
                "result": None,
                "error": failed_error,
                "durationMs": 45,
            },
        ),
        event(
            "item/completed",
            thread_id=thread_id,
            turn_id=turn_id,
            item={
                "type": "agentMessage",
                "phase": "final_answer",
                "text": "最终中文回复",
            },
        ),
    ]
    hook_started = {
        "method": "hook/started",
        "params": {"threadId": thread_id, "turnId": turn_id},
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        incremental_path = Path(temp_dir) / "incremental.jsonl"
        persisted_before_hook: list[dict[str, Any]] = []

        def incremental_stdout_lines() -> Iterable[str]:
            for row in pre_hook_rows:
                yield json.dumps(row, ensure_ascii=False) + "\n"
            persisted_before_hook.extend(
                json.loads(line)
                for line in incremental_path.read_text(encoding="utf-8").splitlines()
            )
            assert len(persisted_before_hook) == len(pre_hook_rows)
            for actual, expected in zip(persisted_before_hook, pre_hook_rows):
                assert_notification_preserved(actual, expected)
            yield json.dumps(hook_started, ensure_ascii=False) + "\n"

        log_client = AppServerClient(cwd=Path(temp_dir), request_timeout_seconds=1)
        log_client._proc = FakeProcess(incremental_stdout_lines())  # type: ignore[assignment]
        log_client.register_turn_event_log(
            thread_id=thread_id,
            turn_id=turn_id,
            path=incremental_path,
        )
        log_client._read_stdout()
        log_client.unregister_turn_event_log(
            thread_id=thread_id,
            turn_id=turn_id,
        )

        raw_jsonl = incremental_path.read_bytes()
        assert "查询中文客户".encode("utf-8") in raw_jsonl
        assert b"\\u67e5\\u8be2" not in raw_jsonl
        persisted_rows = [
            json.loads(line)
            for line in raw_jsonl.decode("utf-8").splitlines()
        ]
        assert len(persisted_rows) == 5
        for actual, expected in zip(persisted_rows, [*pre_hook_rows, hook_started]):
            assert_notification_preserved(actual, expected)
        persisted_large_integer = persisted_rows[1]["params"]["item"]["arguments"][
            "metric_app_id"
        ]
        assert persisted_large_integer == large_integer
        assert isinstance(persisted_large_integer, int)
        tool_items = _completed_tool_items(persisted_rows)
        assert len(tool_items) == 2
        success, failed = tool_items
        assert _tool_name(success) == "metric-mcp-remote.searchMetricApp"
        assert _tool_name(failed) == "metric-mcp-remote.searchBizMetric"
        assert success["durationMs"] == 123
        assert failed["durationMs"] == 45
        bounded_arguments = _clip(success["arguments"])
        assert bounded_arguments["query"] == success_arguments["query"]
        assert bounded_arguments["市场"] == success_arguments["市场"]
        assert bounded_arguments["metric_app_id"] == str(large_integer)
        assert _clip(_tool_output(success)) == success_result
        assert _clip(_tool_output(failed)) == failed_error
        assert latest_final_agent_message(persisted_rows) == "最终中文回复"

    print(
        json.dumps(
            {
                "status": "PASS",
                "tool_count": 2,
                "utf8_fields": ["arguments", "result", "error", "assistant"],
                "ordered_listener": True,
                "received_at": True,
                "listener_error_isolated": True,
                "listener_message_isolated": True,
                "listener_stop_timeout": True,
                "listener_terminal_error_stable": True,
                "close_closes_listeners": True,
                "close_raises_listener_error_group": True,
                "close_cleanup_errors_aggregated": True,
                "register_rejected_when_closed": True,
                "closed_resource_creation_rejected": True,
                "process_cleanup_after_listener_timeout": True,
                "pre_hook_incremental_log": True,
                "utf8_jsonl": True,
                "large_integer_preserved": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
