"""幂等创建 real_cases 三个专项 LLM Judge 及 experiment 绑定。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.upload_dataitems import _api_request, _load_env_file


DATASET_NAME = "aieval/real_cases"
OBSOLETE_RULE_NAMES = {"eval-prompt-real-cases"}
OUTPUT_DEFINITION = {
    "dataType": "NUMERIC",
    "reasoning": {"description": "用一句中文说明首要得分或失分依据"},
    "score": {"description": "0 到 1 之间的数值分数"},
}


JUDGES: Dict[str, Dict[str, Any]] = {
    "real_cases_answer_correctness": {
        "prompt": """你是严谨、保守的数据分析回答正确性评估器。

输入：
- 用户问题：{{input}}
- 整体结果要求：{{expected_result}}
- 本次真实运行输出：{{output}}

本次真实运行输出是 JSON；只评估其中 answer 字段，不要把 tool_evidence 当作回答内容。
只检查回答是否满足整体结果要求：对象、时间范围、筛选条件、统计口径、结论与完整性。
不要因为语言流畅、提供 SQL 草案或声称无法查询而给高分。

评分：
- 1.0：整体结论、范围、口径和必要结果完整正确。
- 0.75：主要结论正确，仅有轻微遗漏，不影响使用。
- 0.5：部分正确，但缺少一个重要结果或口径。
- 0.25：仅提供相关线索、计划或 SQL，未完成主要结果。
- 0.0：无答案、拒答、答非所问或关键结论冲突。

只输出 0～1 分数和一句中文理由。""",
        "mapping": [
            {"variable": "input", "source": "input", "jsonPath": "$.input"},
            {
                "variable": "expected_result",
                "source": "expected_output",
                "jsonPath": "$.expected_result",
            },
            {"variable": "output", "source": "output"},
        ],
    },
    "real_cases_answer_values": {
        "prompt": """你是严谨、保守的数据分析字段与数值评估器。

输入：
- 用户问题：{{input}}
- 字段和数值契约：{{expected_values}}
- 本次真实运行输出：{{output}}

本次真实运行输出是 JSON；只评估其中 answer 字段。
只校验契约明确要求的字段、数值、行数、单位、粒度、排序和容差，不扩展新的要求。
SQL 草案、查询计划和字段口径说明不能替代最终字段与数值结果。

评分：
- 1.0：要求的字段与数值全部存在且正确。
- 0.75：主体数据正确，仅有非关键字段、格式或舍入偏差。
- 0.5：约一半关键字段或数值满足。
- 0.25：只提供字段/公式/SQL，没有足够最终数值。
- 0.0：无结果，或关键字段和数值缺失、错误、冲突。

只输出 0～1 分数和一句中文理由，理由必须点出首要缺失或错误字段。""",
        "mapping": [
            {"variable": "input", "source": "input", "jsonPath": "$.input"},
            {
                "variable": "expected_values",
                "source": "expected_output",
                "jsonPath": "$.expected_values",
            },
            {"variable": "output", "source": "output"},
        ],
    },
    "real_cases_execution_quality": {
        "prompt": """你是严谨、保守的数据分析工具证据评估器。

输入：
- 用户问题：{{input}}
- 预期工具能力：{{expected_tools}}
- 工具匹配模式：{{tool_match_mode}}
- 本次真实运行输出：{{output}}

本次真实运行输出 JSON 包含 answer 与 tool_evidence。你只评估：answer 中的关键结论、字段和数值是否被同次运行 tool_evidence 的真实结果支撑。
completed 只代表调用结束，不自动代表答案有证据；仅调用工具、提供 SQL 草案或引用未执行查询不能高分。
若证据包含 error、失败状态、空结果、与答案冲突，或答案给出证据中不存在的具体数值，必须显著降分。

评分：
- 1.0：关键结论和数值均可由成功工具结果直接核验。
- 0.75：主要结论有证据，仅少量次要内容无法核验。
- 0.5：工具成功且证据相关，但只支撑部分结论或结果被截断。
- 0.25：调用了相关工具，但只有定义、SQL、空结果或间接线索。
- 0.0：没有可用证据、工具失败，或证据与回答关键内容冲突。

只输出 0～1 分数和一句中文理由，理由必须说明证据是否真正支撑答案。""",
        "mapping": [
            {"variable": "input", "source": "input", "jsonPath": "$.input"},
            {
                "variable": "expected_tools",
                "source": "metadata",
                "jsonPath": "$.expected_tools",
            },
            {
                "variable": "tool_match_mode",
                "source": "metadata",
                "jsonPath": "$.tool_match_mode",
            },
            {"variable": "output", "source": "output"},
        ],
    },
}


def _latest(items: list[Dict[str, Any]], name: str) -> Dict[str, Any] | None:
    matches = [
        item
        for item in items
        if item.get("name") == name and item.get("scope") == "project"
    ]
    return max(matches, key=lambda item: int(item.get("version") or 0), default=None)


def _variables(prompt: str) -> list[str]:
    import re

    return list(dict.fromkeys(re.findall(r"{{\s*([A-Za-z0-9_]+)\s*}}", prompt)))


def _ensure_evaluator(
    evaluators: list[Dict[str, Any]],
    *,
    name: str,
    prompt: str,
) -> Dict[str, Any]:
    expected_variables = set(_variables(prompt))
    current = _latest(evaluators, name)
    if (
        current
        and current.get("prompt") == prompt
        and set(current.get("variables") or []) == expected_variables
        and current.get("outputDefinition") == OUTPUT_DEFINITION
    ):
        return current
    created = _api_request(
        "POST",
        "/unstable/evaluators",
        body={
            "name": name,
            "type": "llm_as_judge",
            "prompt": prompt,
            "outputDefinition": OUTPUT_DEFINITION,
            "modelConfig": None,
        },
    )
    if set(created.get("variables") or []) != expected_variables:
        raise RuntimeError(f"{name} 变量创建异常：{created.get('variables')}")
    return created


def _rule_body(
    *, dataset_id: str, name: str, mapping: list[Dict[str, Any]]
) -> Dict[str, Any]:
    return {
        "name": name,
        "evaluator": {
            "name": name,
            "scope": "project",
            "type": "llm_as_judge",
        },
        "target": "experiment",
        "enabled": True,
        "sampling": 1,
        "filter": [
            {
                "type": "stringOptions",
                "column": "datasetId",
                "operator": "any of",
                "value": [dataset_id],
            }
        ],
        "mapping": mapping,
    }


def main() -> int:
    _load_env_file(ROOT / ".env")
    datasets = list(_api_request("GET", "/datasets?limit=100").get("data") or [])
    dataset = next((item for item in datasets if item.get("name") == DATASET_NAME), None)
    if dataset is None:
        raise RuntimeError(f"Langfuse dataset 不存在：{DATASET_NAME}")

    evaluators = list(
        _api_request("GET", "/unstable/evaluators?limit=100").get("data") or []
    )
    rules = list(
        _api_request("GET", "/unstable/evaluation-rules?limit=100").get("data")
        or []
    )
    provisioned = []
    for name, config in JUDGES.items():
        evaluator = _ensure_evaluator(
            evaluators,
            name=name,
            prompt=str(config["prompt"]),
        )
        existing_rule = next((rule for rule in rules if rule.get("name") == name), None)
        body = _rule_body(
            dataset_id=str(dataset["id"]),
            name=name,
            mapping=list(config["mapping"]),
        )
        if existing_rule:
            rule = _api_request(
                "PATCH",
                f"/unstable/evaluation-rules/{existing_rule['id']}",
                body=body,
            )
        else:
            rule = _api_request("POST", "/unstable/evaluation-rules", body=body)
        provisioned.append(
            {
                "name": name,
                "evaluator_id": evaluator.get("id"),
                "version": evaluator.get("version"),
                "variables": evaluator.get("variables"),
                "rule_id": rule.get("id"),
                "mapping": rule.get("mapping"),
            }
        )

    deleted_rules = []
    for old_rule in rules:
        if old_rule.get("name") in OBSOLETE_RULE_NAMES:
            _api_request("DELETE", f"/unstable/evaluation-rules/{old_rule['id']}")
            deleted_rules.append(
                {"id": old_rule.get("id"), "name": old_rule.get("name")}
            )

    print(
        json.dumps(
            {
                "dataset": {"id": dataset.get("id"), "name": dataset.get("name")},
                "judges": provisioned,
                "deleted_rules": deleted_rules,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
