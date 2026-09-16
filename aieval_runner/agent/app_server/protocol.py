from __future__ import annotations

import copy
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, TextIO


JsonObject = Dict[str, Any]
EventPredicate = Callable[[JsonObject], bool]
TurnEventCallback = Callable[[JsonObject], None]
TURN_EVENT_LISTENER_JOIN_TIMEOUT_SECONDS = 10.0


class _TurnEventSubscription:
    _STOP = object()

    def __init__(self, callback: TurnEventCallback) -> None:
        self._callback = callback
        self._queue: "queue.Queue[object]" = queue.Queue()
        self._errors: List[Exception] = []
        self._state_lock = threading.Lock()
        self._closed = False
        self._terminal = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def enqueue(self, message: JsonObject) -> None:
        copied_message = copy.deepcopy(message)
        with self._state_lock:
            if self._closed:
                return
            self._queue.put(copied_message)

    def close(self) -> List[Exception]:
        with self._state_lock:
            if not self._closed:
                self._closed = True
                self._queue.put(self._STOP)
        self._thread.join(timeout=TURN_EVENT_LISTENER_JOIN_TIMEOUT_SECONDS)
        with self._state_lock:
            if self._thread.is_alive() and not self._terminal:
                self._errors.append(TimeoutError("turn event listener did not stop"))
            self._terminal = True
            return list(self._errors)

    def _run(self) -> None:
        while True:
            message = self._queue.get()
            if message is self._STOP:
                return
            if not isinstance(message, dict):
                continue
            try:
                self._callback(message)
            except Exception as exc:
                with self._state_lock:
                    if not self._terminal:
                        self._errors.append(exc)


def sandbox_policy_from_mode(mode: str, cwd: Path) -> JsonObject:
    normalized = (mode or "workspace-write").strip()
    if normalized == "danger-full-access":
        return {"type": "dangerFullAccess"}
    if normalized == "read-only":
        return {"type": "readOnly", "networkAccess": True}
    if normalized == "workspace-write":
        return {
            "type": "workspaceWrite",
            "networkAccess": True,
            "writableRoots": [str(cwd.resolve())],
        }
    raise ValueError(f"unsupported codex sandbox for app-server: {mode}")


class AppServerClient:
    def __init__(
        self,
        *,
        cwd: Path,
        request_timeout_seconds: int,
        codex_bin: str = "",
    ) -> None:
        self.cwd = cwd.resolve()
        self.request_timeout_seconds = request_timeout_seconds
        self.codex_bin = codex_bin.strip()
        self._next_id = 1
        self._responses: "queue.Queue[JsonObject]" = queue.Queue()
        self._events: "queue.Queue[JsonObject]" = queue.Queue()
        self._stderr: "queue.Queue[str]" = queue.Queue()
        self._stderr_lines: List[str] = []
        self._all_messages: List[JsonObject] = []
        self._message_lock = threading.Lock()
        self._message_condition = threading.Condition(self._message_lock)
        self._send_lock = threading.Lock()
        self._turn_event_logs: Dict[tuple[str, str], TextIO] = {}
        self._turn_event_listeners: Dict[tuple[str, str], _TurnEventSubscription] = {}
        self._closed = False
        self._proc: Optional[subprocess.Popen[str]] = None

    def start(self) -> None:
        with self._message_lock:
            self._raise_if_closed_locked()
            if self._proc and self._proc.poll() is None:
                return
            codex_bin = self.codex_bin
            if not codex_bin:
                codex_bin = shutil.which("codex.cmd") if os.name == "nt" else None
                codex_bin = codex_bin or shutil.which("codex") or "codex"
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            self._proc = subprocess.Popen(
                [codex_bin, "app-server", "--stdio"],
                cwd=str(self.cwd),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            threading.Thread(target=self._read_stdout, daemon=True).start()
            threading.Thread(target=self._read_stderr, daemon=True).start()

    def initialize(self) -> JsonObject:
        result = self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "aieval_external_runner",
                    "title": "AI Eval External Runner",
                    "version": "0.1.0",
                },
                "capabilities": {"experimentalApi": True},
            },
        )
        self.notify("initialized", {})
        return result

    def start_thread(
        self,
        *,
        cwd: Path,
        sandbox: str,
        model: str,
        developer_instructions: str = "",
    ) -> JsonObject:
        params: JsonObject = {
            "cwd": str(cwd.resolve()),
            "runtimeWorkspaceRoots": [str(cwd.resolve())],
            "sandbox": sandbox or "workspace-write",
            "approvalPolicy": "never",
            "ephemeral": True,
        }
        if developer_instructions:
            params["developerInstructions"] = developer_instructions
        if model:
            params["model"] = model
        return self.request("thread/start", params)

    def start_turn(self, *, thread_id: str, cwd: Path, prompt: str, sandbox: str, model: str) -> JsonObject:
        params: JsonObject = {
            "threadId": thread_id,
            "cwd": str(cwd.resolve()),
            "runtimeWorkspaceRoots": [str(cwd.resolve())],
            "approvalPolicy": "never",
            "sandboxPolicy": sandbox_policy_from_mode(sandbox, cwd),
            "input": [{"type": "text", "text": prompt}],
        }
        if model:
            params["model"] = model
        return self.request("turn/start", params)

    def request(self, method: str, params: JsonObject | None = None) -> JsonObject:
        with self._message_lock:
            self._raise_if_closed_locked()
            req_id = self._next_id
            self._next_id += 1
        self._send({"method": method, "id": req_id, "params": params or {}})
        deadline = time.monotonic() + max(1, self.request_timeout_seconds)
        while time.monotonic() < deadline:
            with self._message_condition:
                message = self._response_for_request_locked(req_id)
                if message.get("id") == req_id:
                    if "error" in message:
                        raise RuntimeError(f"{method} failed: {json.dumps(message['error'], ensure_ascii=False)}")
                    result = message.get("result")
                    return dict(result) if isinstance(result, dict) else {"value": result}
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._message_condition.wait(timeout=min(1.0, max(0.1, remaining)))
            self._raise_if_dead()
        raise TimeoutError(f"waiting for app-server response timed out: method={method}, id={req_id}")

    def notify(self, method: str, params: JsonObject | None = None) -> None:
        self._send({"method": method, "params": params or {}})

    def wait_for_event(
        self,
        method: str,
        *,
        timeout_seconds: int,
        predicate: EventPredicate | None = None,
    ) -> JsonObject:
        deadline = time.monotonic() + max(1, timeout_seconds)
        with self._message_lock:
            error_start_index = len(self._all_messages)
        while time.monotonic() < deadline:
            with self._message_condition:
                message = self._event_for_wait_locked(method, predicate=predicate)
                if message is not None:
                    return copy.deepcopy(message)
                error = self._error_event_since_locked(error_start_index)
                if error is not None:
                    raise RuntimeError(f"app-server error event: {json.dumps(error, ensure_ascii=False)}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._message_condition.wait(timeout=min(1.0, max(0.1, remaining)))
            self._raise_if_dead()
        raise TimeoutError(f"waiting for app-server event timed out: method={method}")

    def wait_for_turn_completed(self, *, thread_id: str, turn_id: str, timeout_seconds: int) -> JsonObject:
        def matches(message: JsonObject) -> bool:
            params = message.get("params")
            if not isinstance(params, dict):
                return False
            turn = params.get("turn")
            return params.get("threadId") == thread_id and isinstance(turn, dict) and turn.get("id") == turn_id

        event = self.wait_for_event("turn/completed", timeout_seconds=timeout_seconds, predicate=matches)
        params = event.get("params")
        return dict(params) if isinstance(params, dict) else {}

    def messages_snapshot(self) -> List[JsonObject]:
        with self._message_lock:
            return list(self._all_messages)

    def stderr_snapshot(self) -> str:
        return "".join(self._stderr_lines)

    def register_turn_event_log(self, *, thread_id: str, turn_id: str, path: Path) -> None:
        key = (thread_id, turn_id)
        with self._message_lock:
            self._raise_if_closed_locked()
            path.parent.mkdir(parents=True, exist_ok=True)
            previous = self._turn_event_logs.pop(key, None)
            if previous is not None:
                previous.close()
            handle = path.open("w", encoding="utf-8", newline="\n")
            for message in self._all_messages:
                if message_matches_turn(message, thread_id=thread_id, turn_id=turn_id):
                    _write_jsonl_line(handle, message)
            handle.flush()
            self._turn_event_logs[key] = handle

    def unregister_turn_event_log(self, *, thread_id: str, turn_id: str) -> None:
        with self._message_lock:
            handle = self._turn_event_logs.pop((thread_id, turn_id), None)
            if handle is not None:
                handle.close()

    def register_turn_event_listener(
        self,
        *,
        thread_id: str,
        turn_id: str,
        callback: TurnEventCallback,
    ) -> None:
        key = (thread_id, turn_id)
        with self._message_lock:
            self._raise_if_closed_locked()
            if key in self._turn_event_listeners:
                raise ValueError(f"turn event listener already registered: {thread_id}/{turn_id}")
            subscription = _TurnEventSubscription(callback)
            for message in self._all_messages:
                if message_matches_turn(message, thread_id=thread_id, turn_id=turn_id):
                    subscription.enqueue(message)
            self._turn_event_listeners[key] = subscription
            subscription.start()

    def unregister_turn_event_listener(
        self,
        *,
        thread_id: str,
        turn_id: str,
    ) -> List[Exception]:
        with self._message_lock:
            subscription = self._turn_event_listeners.pop((thread_id, turn_id), None)
        return subscription.close() if subscription is not None else []

    def close(self) -> None:
        with self._message_lock:
            if self._closed:
                return
            self._closed = True
            subscriptions = list(self._turn_event_listeners.values())
            self._turn_event_listeners.clear()
            log_handles = list(self._turn_event_logs.values())
            self._turn_event_logs.clear()
            proc = self._proc
        errors: List[Exception] = []
        for handle in log_handles:
            try:
                handle.close()
            except Exception as exc:
                errors.append(exc)
        for subscription in subscriptions:
            try:
                errors.extend(subscription.close())
            except Exception as exc:
                errors.append(exc)
        try:
            if proc is not None:
                process_running = True
                try:
                    process_running = proc.poll() is None
                except Exception as exc:
                    errors.append(exc)
                if process_running:
                    try:
                        proc.terminate()
                    except Exception as exc:
                        errors.append(exc)
                    should_kill = False
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired as exc:
                        errors.append(exc)
                        should_kill = True
                    except Exception as exc:
                        errors.append(exc)
                        should_kill = True
                    if should_kill:
                        try:
                            proc.kill()
                        except Exception as exc:
                            errors.append(exc)
        finally:
            self._proc = None
        if errors:
            raise ExceptionGroup("app-server client shutdown failed", errors)

    def _send(self, message: JsonObject) -> None:
        with self._send_lock:
            if not self._proc or not self._proc.stdin:
                raise RuntimeError("app-server is not started")
            self._raise_if_dead()
            self._proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()

    def _response_for_request_locked(self, req_id: int) -> JsonObject:
        for message in self._all_messages:
            if message.get("id") == req_id and ("result" in message or "error" in message):
                return message
        return {}

    def _event_for_wait_locked(
        self,
        method: str,
        *,
        predicate: EventPredicate | None = None,
    ) -> JsonObject | None:
        for message in self._all_messages:
            if message.get("method") == method and (predicate is None or predicate(message)):
                return message
        return None

    def _error_event_since_locked(self, start_index: int) -> JsonObject | None:
        for message in self._all_messages[start_index:]:
            if message.get("method") == "error":
                return message
        return None

    def _record_internal_event(self, message: JsonObject) -> None:
        with self._message_condition:
            self._all_messages.append(message)
            self._message_condition.notify_all()
        self._events.put(message)

    def _read_stdout(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            raw = line.strip()
            if not raw:
                continue
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                message = {"method": "invalid-json", "params": {"raw": raw}}
            if isinstance(message, dict):
                if "id" not in message and isinstance(message.get("method"), str):
                    message["_received_at"] = datetime.now(timezone.utc).isoformat()
                with self._message_lock:
                    self._all_messages.append(message)
                    for (thread_id, turn_id), handle in self._turn_event_logs.items():
                        if message_matches_turn(message, thread_id=thread_id, turn_id=turn_id):
                            _write_jsonl_line(handle, message)
                            if should_flush_turn_event_log(message):
                                handle.flush()
                    for (thread_id, turn_id), subscription in self._turn_event_listeners.items():
                        if message_matches_turn(message, thread_id=thread_id, turn_id=turn_id):
                            subscription.enqueue(message)
                    self._message_condition.notify_all()
                if "id" in message and ("result" in message or "error" in message):
                    self._responses.put(message)
                else:
                    self._events.put(message)
                    self._auto_respond_to_elicitation(message)

    def _read_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        for line in self._proc.stderr:
            self._stderr_lines.append(line)
            self._stderr.put(line)

    def _auto_respond_to_elicitation(self, message: JsonObject) -> None:
        if message.get("method") != "mcpServer/elicitation/request":
            return
        request_id = message.get("id")
        if request_id is None:
            return
        params = message.get("params")
        if not isinstance(params, dict):
            return
        meta = params.get("_meta")
        if not isinstance(meta, dict) or meta.get("codex_approval_kind") != "mcp_tool_call":
            return
        try:
            self._send({"id": request_id, "result": {"action": "accept"}})
        except Exception as exc:
            self._record_internal_event(
                {
                    "method": "error",
                    "params": {
                        "message": f"auto approval response failed: {exc}",
                    },
                }
            )

    def _raise_if_dead(self) -> None:
        if self._proc and self._proc.poll() is not None:
            raise RuntimeError(f"app-server exited: returncode={self._proc.returncode}")

    def _raise_if_closed_locked(self) -> None:
        if self._closed:
            raise RuntimeError("app-server client is closed")


def extract_thread_id(response: JsonObject) -> str:
    thread = response.get("thread")
    if isinstance(thread, dict):
        thread_id = thread.get("id") or thread.get("threadId")
        if thread_id:
            return str(thread_id)
    for key in ("threadId", "id"):
        if response.get(key):
            return str(response[key])
    raise RuntimeError(f"unable to parse thread id from response: {json.dumps(response, ensure_ascii=False)}")


def extract_session_id(response: JsonObject) -> str:
    thread = response.get("thread")
    if isinstance(thread, dict):
        session_id = thread.get("sessionId") or thread.get("session_id") or thread.get("id")
        if session_id:
            return str(session_id)
    return extract_thread_id(response)


def extract_turn_id(response: JsonObject) -> str:
    turn = response.get("turn")
    if isinstance(turn, dict) and turn.get("id"):
        return str(turn["id"])
    for key in ("turnId", "id"):
        if response.get(key):
            return str(response[key])
    raise RuntimeError(f"unable to parse turn id from response: {json.dumps(response, ensure_ascii=False)}")


def messages_for_turn(messages: Iterable[JsonObject], *, thread_id: str, turn_id: str) -> List[JsonObject]:
    return [
        message
        for message in messages
        if message_matches_turn(message, thread_id=thread_id, turn_id=turn_id)
    ]


def message_matches_turn(message: JsonObject, *, thread_id: str, turn_id: str) -> bool:
    params = message.get("params")
    if not isinstance(params, dict) or params.get("threadId") != thread_id:
        return False
    if params.get("turnId") == turn_id:
        return True
    turn = params.get("turn")
    return isinstance(turn, dict) and turn.get("id") == turn_id


def should_flush_turn_event_log(message: JsonObject) -> bool:
    if message.get("method") in {"hook/started", "turn/completed"}:
        return True
    params = message.get("params")
    item = params.get("item") if isinstance(params, dict) else None
    if not isinstance(item, dict) or message.get("method") != "item/completed":
        return False
    if item.get("type") == "mcpToolCall":
        return True
    return item.get("type") == "agentMessage"


def latest_final_agent_message(messages: Iterable[JsonObject]) -> str:
    final = ""
    fallback = ""
    for message in messages:
        if message.get("method") != "item/completed":
            continue
        params = message.get("params")
        item = params.get("item") if isinstance(params, dict) else None
        if not isinstance(item, dict) or item.get("type") != "agentMessage":
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text:
            continue
        fallback = text
        if item.get("phase") == "final_answer":
            final = text
    return final or fallback


def mcp_tool_events(messages: Iterable[JsonObject]) -> List[JsonObject]:
    events: List[JsonObject] = []
    for message in messages:
        params = message.get("params")
        item = params.get("item") if isinstance(params, dict) else None
        if isinstance(item, dict) and item.get("type") == "mcpToolCall":
            events.append(message)
    return events


def write_jsonl(path: Path, messages: Iterable[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for message in messages:
            _write_jsonl_line(fh, message)


def _write_jsonl_line(handle: TextIO, message: JsonObject) -> None:
    handle.write(json.dumps(message, ensure_ascii=False, default=str, separators=(",", ":")) + "\n")
