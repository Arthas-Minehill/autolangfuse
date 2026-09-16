from __future__ import annotations

from typing import Any, Dict, Optional

from aieval_runner.core.models import EvalCase
from aieval_runner.evaluation.events import expected_tool_leaf


DATA_QUERY_DEVELOPER_INSTRUCTIONS = """
仅使用当前会话直接暴露的企业数据工具完成数据发现与查询。可以读取系统明确列出的
SKILL.md；除此之外，不得搜索或读取本地仓库、题库、Gold、评测产物、进程信息、
环境变量、连接配置或凭证。不得通过 shell、SDK、CLI、直接 HTTP 请求或手写协议
连接企业数据服务。会话未暴露可信取数工具时，应说明证据边界并给出查询计划，
不得从本地文件寻找答案或旁路取数。
""".strip()


def build_user_prompt(case: EvalCase) -> str:
    return case.original_request


def select_target_observation_id(mapping: Dict[str, Any], case: EvalCase) -> Optional[str]:
    tools = mapping.get("tool_observation_ids")
    if not isinstance(tools, list):
        return None

    def tool_name(tool: Any) -> str:
        return str(tool.get("name") or "") if isinstance(tool, dict) else ""

    expected_names = [
        expected_tool_leaf(name)
        for name in case.tool_contract.get("expected_tools", [])
    ]
    for expected_name in expected_names:
        for tool in tools:
            name = expected_tool_leaf(tool_name(tool))
            if expected_name and name == expected_name:
                return str(tool.get("id") or "") if isinstance(tool, dict) and tool.get("id") else None

    for tool in tools:
        if isinstance(tool, dict) and tool.get("id"):
            return str(tool["id"])
    return None
