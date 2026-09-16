from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".aieval" / "app-server-mcp-check"


class AppServerClient:
    def __init__(self, *, cwd: Path, output_dir: Path, timeout_seconds: int) -> None:
        self.cwd = cwd
        self.output_dir = output_dir
        self.timeout_seconds = timeout_seconds
        self._next_id = 1
        self._responses: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._events: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._stderr: "queue.Queue[str]" = queue.Queue()
        self._all_messages: List[Dict[str, Any]] = []
        self._proc: Optional[subprocess.Popen[str]] = None

    def start(self) -> None:
        codex_bin = shutil.which("codex.cmd") if os.name == "nt" else None
        codex_bin = codex_bin or shutil.which("codex") or "codex"
        cmd = [codex_bin, "app-server", "--stdio"]
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self.cwd),
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

    def close(self) -> None:
        if not self._proc:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._write_jsonl("messages.jsonl", self._all_messages)
        stderr_lines: List[str] = []
        while True:
            try:
                stderr_lines.append(self._stderr.get_nowait())
            except queue.Empty:
                break
        if stderr_lines:
            (self.output_dir / "stderr.txt").write_text("".join(stderr_lines), encoding="utf-8")

    def initialize(self) -> Dict[str, Any]:
        response = self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "aieval_app_server_mcp_check",
                    "title": "AI Eval App Server MCP Check",
                    "version": "0.1.0",
                },
                "capabilities": {"experimentalApi": True},
            },
        )
        self.notify("initialized", {})
        return response

    def request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1
        self._send({"method": method, "id": req_id, "params": params})
        deadline = time.monotonic() + self.timeout_seconds
        deferred: List[Dict[str, Any]] = []
        try:
            while time.monotonic() < deadline:
                remaining = max(0.1, deadline - time.monotonic())
                try:
                    message = self._responses.get(timeout=min(remaining, 1.0))
                except queue.Empty:
                    self._raise_if_dead()
                    continue
                if message.get("id") == req_id:
                    if "error" in message:
                        raise RuntimeError(f"{method} failed: {json.dumps(message['error'], ensure_ascii=False)}")
                    return dict(message.get("result") or {})
                deferred.append(message)
            raise TimeoutError(f"等待 app-server 响应超时: method={method}, id={req_id}")
        finally:
            for message in deferred:
                self._responses.put(message)

    def notify(self, method: str, params: Dict[str, Any]) -> None:
        self._send({"method": method, "params": params})

    def wait_for_turn_completed(self, *, timeout_seconds: int) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                message = self._events.get(timeout=1.0)
            except queue.Empty:
                self._raise_if_dead()
                continue
            if message.get("method") == "turn/completed":
                return dict(message.get("params") or {})
            if message.get("method") == "error":
                raise RuntimeError(f"app-server error event: {json.dumps(message, ensure_ascii=False)}")
        raise TimeoutError("等待 turn/completed 超时")

    def drain_events(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                break
        return events

    def messages_snapshot(self) -> List[Dict[str, Any]]:
        return list(self._all_messages)

    def _send(self, message: Dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin:
            raise RuntimeError("app-server 未启动")
        self._proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

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
            self._all_messages.append(message)
            if "id" in message and ("result" in message or "error" in message):
                self._responses.put(message)
            else:
                self._events.put(message)

    def _read_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        for line in self._proc.stderr:
            self._stderr.put(line)

    def _raise_if_dead(self) -> None:
        if self._proc and self._proc.poll() is not None:
            raise RuntimeError(f"app-server 已退出: returncode={self._proc.returncode}")

    def _write_jsonl(self, filename: str, messages: Iterable[Dict[str, Any]]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as fh:
            for message in messages:
                fh.write(json.dumps(message, ensure_ascii=False) + "\n")


def compact_status(status: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for server in status.get("data") or []:
        if not isinstance(server, dict):
            continue
        tools = server.get("tools") or {}
        rows.append(
            {
                "name": server.get("name"),
                "authStatus": server.get("authStatus"),
                "toolCount": len(tools) if isinstance(tools, dict) else 0,
                "tools": sorted(tools.keys()) if isinstance(tools, dict) else [],
            }
        )
    return rows


def find_server(status: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for server in status.get("data") or []:
        if isinstance(server, dict) and server.get("name") == name:
            return server
    return None


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def start_thread(client: AppServerClient, args: argparse.Namespace) -> Dict[str, Any]:
    thread = client.request(
        "thread/start",
        {
            "cwd": str(Path(args.cwd).resolve()),
            "runtimeWorkspaceRoots": [str(Path(args.cwd).resolve())],
            "sandbox": args.sandbox,
            "approvalPolicy": args.approval_policy,
            "ephemeral": True,
            "model": args.model or None,
        },
    )
    thread_id = thread.get("thread", {}).get("id") or thread.get("threadId") or thread.get("id")
    if not thread_id:
        raise RuntimeError(f"无法从 thread/start 响应中解析 thread id: {json.dumps(thread, ensure_ascii=False)}")
    return {"thread": thread, "threadId": thread_id}


def run_agent_turn(client: AppServerClient, args: argparse.Namespace) -> Dict[str, Any]:
    started = start_thread(client, args)
    thread_id = started["threadId"]

    prompt = (
        f"请使用 MCP 服务器 `{args.server}` 的 `{args.tool}` 工具验证可用性。"
        "只输出是否成功、工具名和返回内容的一句话摘要。"
    )
    client.request(
        "turn/start",
        {
            "threadId": thread_id,
            "cwd": str(Path(args.cwd).resolve()),
            "runtimeWorkspaceRoots": [str(Path(args.cwd).resolve())],
            "approvalPolicy": args.approval_policy,
            "sandboxPolicy": {"type": "dangerFullAccess"},
            "input": [{"type": "text", "text": prompt}],
        },
    )
    completed = client.wait_for_turn_completed(timeout_seconds=args.turn_timeout_seconds)
    messages = client.messages_snapshot()
    mcp_tool_events = []
    final_agent_message = None
    for message in messages:
        params = message.get("params")
        item = params.get("item") if isinstance(params, dict) else None
        if isinstance(item, dict) and item.get("type") == "mcpToolCall":
            mcp_tool_events.append(message)
        if isinstance(item, dict) and item.get("type") == "agentMessage" and item.get("phase") == "final_answer":
            final_agent_message = item.get("text")
    return {
        "thread": started["thread"],
        "turnCompleted": completed,
        "mcpToolCallEvents": mcp_tool_events,
        "finalAgentMessage": final_agent_message,
        "eventCount": sum(1 for message in messages if "method" in message),
    }


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 codex app-server 是否能调用当前配置的 MCP 工具")
    parser.add_argument("--server", default="metric-mcp-remote", help="MCP server name")
    parser.add_argument("--tool", default="metricMcpInfo", help="MCP tool name")
    parser.add_argument("--arguments-json", default="{}", help="MCP tool arguments JSON")
    parser.add_argument("--cwd", default=str(PROJECT_ROOT), help="app-server/thread 工作目录")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="诊断输出目录")
    parser.add_argument("--timeout-seconds", type=int, default=60, help="单个 JSON-RPC 请求超时")
    parser.add_argument("--agent-turn", action="store_true", help="额外跑一次模型 turn，验证 agent 是否会调用 MCP")
    parser.add_argument("--turn-timeout-seconds", type=int, default=180, help="模型 turn 完成超时")
    parser.add_argument("--model", default="", help="可选模型覆盖")
    parser.add_argument("--approval-policy", default="never", help="thread/turn approvalPolicy")
    parser.add_argument(
        "--sandbox",
        default="danger-full-access",
        choices=["read-only", "workspace-write", "danger-full-access"],
        help="thread/start sandbox",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        tool_arguments = json.loads(args.arguments_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--arguments-json 不是合法 JSON: {exc}") from exc
    if not isinstance(tool_arguments, dict):
        raise SystemExit("--arguments-json 必须是 JSON object")

    client = AppServerClient(cwd=Path(args.cwd).resolve(), output_dir=output_dir, timeout_seconds=args.timeout_seconds)
    summary: Dict[str, Any] = {
        "server": args.server,
        "tool": args.tool,
        "cwd": str(Path(args.cwd).resolve()),
        "outputDir": str(output_dir),
        "checks": {},
    }
    try:
        client.start()
        init_result = client.initialize()
        write_json(output_dir / "initialize.json", init_result)

        status = client.request("mcpServerStatus/list", {"detail": "full", "limit": 100})
        write_json(output_dir / "mcp_status.json", status)
        status_rows = compact_status(status)
        write_json(output_dir / "mcp_status_compact.json", status_rows)
        target_server = find_server(status, args.server)
        summary["checks"]["server_visible"] = target_server is not None

        if not target_server:
            available = [row["name"] for row in status_rows]
            summary["availableServers"] = available
            write_json(output_dir / "summary.json", summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 2

        tools = target_server.get("tools") or {}
        summary["checks"]["tool_visible"] = isinstance(tools, dict) and args.tool in tools
        summary["targetServer"] = {
            "name": target_server.get("name"),
            "authStatus": target_server.get("authStatus"),
            "toolCount": len(tools) if isinstance(tools, dict) else 0,
            "availableTools": sorted(tools.keys()) if isinstance(tools, dict) else [],
        }

        if not summary["checks"]["tool_visible"]:
            write_json(output_dir / "summary.json", summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 3

        direct_thread = start_thread(client, args)
        write_json(output_dir / "direct_thread.json", direct_thread["thread"])

        direct_result = client.request(
            "mcpServer/tool/call",
            {
                "server": args.server,
                "tool": args.tool,
                "arguments": tool_arguments,
                "threadId": direct_thread["threadId"],
            },
        )
        write_json(output_dir / "direct_tool_call.json", direct_result)
        summary["checks"]["direct_tool_call_succeeded"] = not bool(direct_result.get("isError"))
        summary["directToolCall"] = {
            "isError": direct_result.get("isError"),
            "contentPreview": json.dumps(direct_result.get("content"), ensure_ascii=False)[:800],
        }

        if args.agent_turn:
            agent_result = run_agent_turn(client, args)
            write_json(output_dir / "agent_turn.json", agent_result)
            agent_events = agent_result.get("mcpToolCallEvents") or []
            summary["checks"]["agent_turn_completed"] = True
            summary["checks"]["agent_turn_had_mcp_event"] = any(
                event.get("params", {}).get("item", {}).get("server") == args.server
                and event.get("params", {}).get("item", {}).get("tool") == args.tool
                for event in agent_events
            )
            summary["agentTurn"] = {
                "mcpToolCallEventCount": len(agent_events),
                "finalAgentMessage": agent_result.get("finalAgentMessage"),
            }

        write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if all(summary["checks"].values()) else 4
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
