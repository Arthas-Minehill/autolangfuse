from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aieval_runner.agent.app_server.protocol import AppServerClient, extract_thread_id, write_jsonl  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".aieval" / "app-server-extension-check"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def count_skills(value: Dict[str, Any]) -> int:
    data = value.get("data")
    if isinstance(data, list):
        return sum(len(row.get("skills") or []) for row in data if isinstance(row, dict))
    skills = value.get("skills")
    if isinstance(skills, list):
        return len(skills)
    result = value.get("result")
    if isinstance(result, dict):
        return count_skills(result)
    return 0


def count_plugins(value: Dict[str, Any]) -> int:
    marketplaces = value.get("marketplaces")
    if isinstance(marketplaces, list):
        return sum(len(row.get("plugins") or []) for row in marketplaces if isinstance(row, dict))
    plugins = value.get("plugins")
    if isinstance(plugins, list):
        return len(plugins)
    result = value.get("result")
    if isinstance(result, dict):
        return count_plugins(result)
    return 0


def parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify codex app-server skills and plugins")
    parser.add_argument("--cwd", default=str(PROJECT_ROOT), help="app-server/thread working directory")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="diagnostic output directory")
    parser.add_argument("--timeout-seconds", type=int, default=60, help="single JSON-RPC request timeout")
    parser.add_argument("--sandbox", default="danger-full-access", help="thread sandbox mode")
    parser.add_argument("--model", default="", help="optional model override")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    cwd = Path(args.cwd).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    client = AppServerClient(cwd=cwd, request_timeout_seconds=args.timeout_seconds)
    summary: Dict[str, Any] = {
        "cwd": str(cwd),
        "outputDir": str(output_dir),
        "checks": {},
    }
    try:
        client.start()
        initialize = client.initialize()
        write_json(output_dir / "initialize.json", initialize)

        skills = client.request("skills/list", {"cwds": [str(cwd)], "forceReload": False})
        write_json(output_dir / "skills_list.json", skills)
        summary["checks"]["skills_list_succeeded"] = True
        summary["skillCount"] = count_skills(skills)
        summary["checks"]["skills_visible"] = summary["skillCount"] > 0

        plugins = client.request("plugin/list", {"cwds": [str(cwd)]})
        write_json(output_dir / "plugin_list.json", plugins)
        summary["checks"]["plugin_list_succeeded"] = True
        summary["pluginCount"] = count_plugins(plugins)
        summary["checks"]["plugins_visible"] = summary["pluginCount"] > 0

        thread = client.start_thread(cwd=cwd, sandbox=args.sandbox, model=args.model)
        write_json(output_dir / "direct_thread.json", thread)
        thread_id = extract_thread_id(thread)
        plugin_tool = client.request(
            "mcpServer/tool/call",
            {
                "server": "codex_apps",
                "tool": "sites_list_sites",
                "arguments": {"limit": 1},
                "threadId": thread_id,
            },
        )
        write_json(output_dir / "direct_plugin_tool_call.json", plugin_tool)
        summary["checks"]["readonly_plugin_tool_succeeded"] = not bool(plugin_tool.get("isError"))
        summary["pluginTool"] = {
            "server": "codex_apps",
            "tool": "sites_list_sites",
            "isError": plugin_tool.get("isError"),
            "contentPreview": json.dumps(plugin_tool.get("content"), ensure_ascii=False, default=str)[:800],
        }

        write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if all(summary["checks"].values()) else 4
    finally:
        write_jsonl(output_dir / "messages.jsonl", client.messages_snapshot())
        stderr = client.stderr_snapshot()
        if stderr:
            (output_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
