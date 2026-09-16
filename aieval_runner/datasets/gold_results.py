from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from aieval_runner.evaluation.tables.contract import normalize_table


def stable_dataset_item_id(dataset: str, case_id: str) -> str:
    digest = hashlib.sha256(f"{dataset}\0{case_id}".encode("utf-8")).hexdigest()
    return f"aieval-{digest[:32]}"


def _parse_jsonish(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def _split_list(value: Any) -> List[str]:
    parsed = _parse_jsonish(value, [])
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if isinstance(parsed, str):
        return [part.strip() for part in parsed.split(",") if part.strip()]
    return []


def _normalize_identifier(value: Any) -> str:
    return "".join(character for character in str(value or "").lower() if character.isalnum())


def _canonical_tool_name(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if "." in raw:
        server, tool = raw.rsplit(".", 1)
    else:
        server, tool = "", raw
    server = {
        "middle-mcp-remote": "metric-mcp-remote",
    }.get(server, server)
    tool = {
        "getMetricAppPage": "searchMetricApp",
        "getMetricAppQueryResult": "searchMetricAppQueryResult",
        "getSqlByMetricTypeAndNameExact": "searchSqlByMetricTypeAndNameExact",
    }.get(tool, tool)
    return f"{server}.{tool}" if server else tool


def _parse_expected_tool_calls(value: Any) -> List[Dict[str, Any]]:
    parsed = _parse_jsonish(value, [])
    if not isinstance(parsed, list):
        return []
    calls: List[Dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, dict):
            calls.append(item)
        elif isinstance(item, str) and item.strip():
            calls.append({"tool_name": item.strip(), "arguments": {}})
    return calls


def _tool_names(calls: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for call in calls:
        name = _canonical_tool_name(call.get("tool_name") or call.get("name"))
        if name and name not in names:
            names.append(name)
    return names


def _main_query_from_calls(calls: List[Dict[str, Any]]) -> str:
    target_names = {
        "getmetricappqueryresult",
        "getsqlbymetrictypeandnameexact",
        "searchmetricappqueryresult",
        "searchsqlbymetrictypeandnameexact",
    }
    for call in reversed(calls):
        tool_name = str(call.get("tool_name") or call.get("name") or "")
        if _normalize_identifier(tool_name.rsplit(".", 1)[-1]) not in target_names:
            continue
        arguments = call.get("arguments")
        if not isinstance(arguments, dict):
            continue
        req = arguments.get("req")
        if isinstance(req, dict) and isinstance(req.get("name"), str) and req["name"].strip():
            return req["name"].strip()
        name = arguments.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return ""


def _parse_schema_columns(value: Any) -> List[str]:
    parsed = _parse_jsonish(value, {})
    if not isinstance(parsed, dict):
        return []
    columns = parsed.get("columns")
    if not isinstance(columns, list):
        return []
    return [str(column).strip() for column in columns if str(column).strip()]


def _case_id_from_raw(raw: Dict[str, Any]) -> str:
    input_obj = raw.get("input") if isinstance(raw.get("input"), dict) else {}
    expected = raw.get("expectedOutput", raw.get("expected_output"))
    expected_obj = expected if isinstance(expected, dict) else {}
    return str(
        raw.get("case_id")
        or input_obj.get("case_id")
        or expected_obj.get("case_id")
        or raw.get("id")
        or ""
    ).strip()


def _normalize_gold_entry(value: Any, *, case_id: str) -> Dict[str, Any]:
    parsed = _parse_jsonish(value, value)
    contract: Dict[str, Any] = {}
    table_value = parsed
    if isinstance(parsed, dict) and "expected_result" in parsed:
        table_value = parsed.get("expected_result")
        raw_contract = parsed.get("contract")
        if isinstance(raw_contract, dict):
            contract = dict(raw_contract)
    table = normalize_table(
        _parse_jsonish(table_value, None),
        field_name=f"case_id={case_id}.expected_result",
    )
    return {"expected_result": table, "contract": contract}


def load_gold_results(path: str) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    parsed = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    source = parsed
    if isinstance(parsed, dict):
        source = parsed.get("items") or parsed.get("data") or parsed

    results: Dict[str, Dict[str, Any]] = {}
    if isinstance(source, dict):
        for case_id, entry in source.items():
            clean_case_id = str(case_id).strip()
            if clean_case_id:
                results[clean_case_id] = _normalize_gold_entry(entry, case_id=clean_case_id)
        return results

    if isinstance(source, list):
        for item in source:
            if not isinstance(item, dict):
                continue
            case_id = _case_id_from_raw(item)
            if not case_id or "expected_result" not in item:
                continue
            results[case_id] = _normalize_gold_entry(
                {"expected_result": item.get("expected_result"), "contract": item.get("contract")},
                case_id=case_id,
            )
        return results

    raise ValueError("--gold-results-json must be an object or a list of case objects")


def _apply_gold(
    expected_output: Dict[str, Any],
    *,
    case_id: str,
    gold_results: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    if not gold_results or case_id not in gold_results:
        return expected_output
    gold_entry = gold_results[case_id]
    expected_output = {
        **expected_output,
        "expected_result": copy.deepcopy(gold_entry["expected_result"]),
    }
    contract = gold_entry.get("contract")
    if isinstance(contract, dict) and contract:
        expected_output["expected_result_contract"] = copy.deepcopy(contract)
    return expected_output


def normalize_eval_raw_item(
    raw: Dict[str, Any],
    *,
    dataset_name: str,
    gold_results: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if raw.get("input") is not None and raw.get("expectedOutput", raw.get("expected_output")) is not None:
        item = copy.deepcopy(raw)
        input_obj = _parse_jsonish(item.get("input"), None)
        expected_output = _parse_jsonish(item.get("expectedOutput", item.get("expected_output")), None)
        metadata = _parse_jsonish(item.get("metadata"), {})
        if not isinstance(input_obj, dict) or not isinstance(expected_output, dict):
            return item
        if not isinstance(metadata, dict):
            metadata = {"metadata": metadata}
        case_id = str(input_obj.get("case_id") or expected_output.get("case_id") or "").strip()
        if "expected_result" not in expected_output and metadata.get("expected_result") not in (None, ""):
            expected_output = {
                **expected_output,
                "expected_result": normalize_table(
                    _parse_jsonish(metadata.get("expected_result"), None),
                    field_name=f"case_id={case_id}.metadata.expected_result",
                ),
            }
        if (
            "expected_result_contract" not in expected_output
            and isinstance(metadata.get("expected_result_contract"), dict)
        ):
            expected_output["expected_result_contract"] = dict(metadata["expected_result_contract"])
        if case_id:
            expected_output = _apply_gold(
                expected_output,
                case_id=case_id,
                gold_results=gold_results,
            )
        item["input"] = input_obj
        item["expectedOutput"] = expected_output
        item["metadata"] = metadata
        if not str(item.get("id") or "").strip() and case_id:
            item["id"] = stable_dataset_item_id(dataset_name, case_id)
        return item

    case_id = _case_id_from_raw(raw)
    original_request = str(
        raw.get("original_request")
        or raw.get("user_query")
        or raw.get("question")
        or ""
    ).strip()
    tool_calls = _parse_expected_tool_calls(raw.get("expected_tools"))
    schema_columns = _parse_schema_columns(raw.get("expected_result_schema"))
    evaluation_focus = _split_list(raw.get("evaluation_focus"))
    if not evaluation_focus and schema_columns:
        evaluation_focus = [f"field:{column}" for column in schema_columns]
    expected_output: Dict[str, Any] = {
        "case_id": case_id,
        "main_query": str(raw.get("main_query") or "").strip() or _main_query_from_calls(tool_calls),
        "expected_tools": _tool_names(tool_calls) or _split_list(raw.get("expected_tools")),
        "evaluation_focus": evaluation_focus,
    }
    source_sql = str(raw.get("expected_sql") or raw.get("source_sql") or "").strip()
    if source_sql:
        expected_output["source_sql"] = source_sql
    inline_result = raw.get("expected_result")
    if inline_result not in (None, ""):
        expected_output["expected_result"] = normalize_table(
            _parse_jsonish(inline_result, None),
            field_name=f"case_id={case_id}.expected_result",
        )
    expected_output = _apply_gold(
        expected_output,
        case_id=case_id,
        gold_results=gold_results,
    )

    metadata = _parse_jsonish(raw.get("metadata"), {})
    if not isinstance(metadata, dict):
        metadata = {"metadata": metadata}
    if case_id:
        metadata.setdefault("case_id", case_id)
    for key in (
        "task_level",
        "task_type",
        "business_domain",
        "clarification_context",
        "difficulty",
        "deterministic_evaluators",
        "evaluation_profile",
        "expected_answer_contract",
        "source",
        "created_at",
        "last_verified_at",
    ):
        value = raw.get(key)
        if isinstance(value, str):
            value = value.strip()
        if value not in (None, ""):
            metadata.setdefault(key, value)

    return {
        "id": str(raw.get("dataset_item_id") or raw.get("langfuse_dataset_item_id") or "").strip()
        or stable_dataset_item_id(dataset_name, case_id),
        "input": {
            "case_id": case_id,
            "original_request": original_request,
        },
        "expectedOutput": expected_output,
        "metadata": metadata,
    }
