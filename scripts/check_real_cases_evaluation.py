"""验证 real_cases 数据契约、完整事件解析与两项确定性评分。"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_NAME = "aieval/real_cases"
SECTIONS = ("dataset", "events", "scoring", "provisioning", "workspace")


def _real_case_item(
    *,
    mode: str = "any",
    tools: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": "item-real-001",
        "input": {"input": "查询最近一个月的业务指标。"},
        "expectedOutput": {
            "expected_result": "返回最近一个月的指标结果。",
            "expected_values": {
                "columns": ["指标值"],
                "row_count": 1,
            },
        },
        "metadata": {
            "case_id": "real-001",
            "source": "回归样例",
            "expected_tools": (
                ["mcp__metric_mcp_remote__searchBizMetric"]
                if tools is None
                else tools
            ),
            "tool_match_mode": mode,
        },
        "status": "ACTIVE",
    }


def _completed_tool_event(
    *,
    call_id: str,
    server: str,
    tool: str,
    result: Any,
    error: Any = None,
) -> dict[str, Any]:
    return {
        "method": "item/completed",
        "params": {
            "item": {
                "type": "mcpToolCall",
                "id": call_id,
                "server": server,
                "tool": tool,
                "status": "completed",
                "arguments": {},
                "result": result,
                "error": error,
            }
        },
    }


def _started_tool_event(*, call_id: str, server: str, tool: str) -> dict[str, Any]:
    return {
        "method": "item/started",
        "params": {
            "item": {
                "type": "mcpToolCall",
                "id": call_id,
                "server": server,
                "tool": tool,
                "status": "inProgress",
            }
        },
    }


def _write_events(path: Path, events: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )


def _case_prompt(case: Any) -> str:
    value = getattr(case, "original_request", None)
    if value is None:
        value = getattr(case, "request", None)
    assert isinstance(value, str), "EvalCase 必须保留真实用户问题"
    return value


def check_dataset() -> None:
    from aieval_runner.core.constants import (
        EXPECTED_OUTPUT_SCHEMA,
        INPUT_SCHEMA,
        METADATA_SCHEMA,
    )
    from aieval_runner.core.models import EvalRunContext
    from aieval_runner.datasets.cases import build_trace_metadata, parse_eval_case
    from aieval_runner.runner.config import normalize_config
    from scripts.upload_dataitems import to_dataset_item_payload

    any_case = parse_eval_case(_real_case_item(), dataset_name=DATASET_NAME)
    assert any_case.case_id == "real-001"
    assert _case_prompt(any_case) == "查询最近一个月的业务指标。"
    assert any_case.item_metadata["source"] == "回归样例"
    assert any_case.expected_output == {
        "expected_result": "返回最近一个月的指标结果。",
        "expected_values": {"columns": ["指标值"], "row_count": 1},
    }
    assert any_case.tool_contract == {
        "expected_tools": ["mcp__metric_mcp_remote__searchBizMetric"],
        "tool_match_mode": "any",
    }
    trace_metadata = build_trace_metadata(
        any_case,
        EvalRunContext(
            eval_run_id="run-profile",
            dataset_name=DATASET_NAME,
            agent_version="回归",
            client="回归",
            skill_version="回归",
            db_snapshot_id="回归",
            response_profile="complete_data_result_v1",
            response_instructions_sha256="a" * 64,
        ),
    )
    assert trace_metadata["response_profile"] == "complete_data_result_v1"
    assert trace_metadata["response_instructions_sha256"] == "a" * 64
    assert "tool_result_status" not in EXPECTED_OUTPUT_SCHEMA["properties"]
    assert EXPECTED_OUTPUT_SCHEMA["required"] == ["expected_result", "expected_values"]
    assert INPUT_SCHEMA["properties"]["input"]["maxLength"] == 100_000
    assert METADATA_SCHEMA["properties"]["case_id"]["maxLength"] == 500
    assert METADATA_SCHEMA["properties"]["expected_tools"]["maxItems"] == 50
    assert METADATA_SCHEMA["required"] == [
        "case_id",
        "expected_tools",
        "tool_match_mode",
    ]

    all_case = parse_eval_case(
        _real_case_item(
            mode="all",
            tools=[
                "mcp__metric_mcp_remote__searchBizMetric",
                "mcp__metabase__query",
            ],
        ),
        dataset_name=DATASET_NAME,
    )
    assert all_case.tool_contract["tool_match_mode"] == "all"
    assert len(all_case.tool_contract["expected_tools"]) == 2

    upload_payload = to_dataset_item_payload(
        DATASET_NAME,
        _real_case_item(),
        allow_update=True,
    )
    assert upload_payload["datasetName"] == DATASET_NAME
    assert upload_payload["metadata"]["case_id"] == "real-001"
    assert upload_payload["expectedOutput"] == any_case.expected_output
    assert upload_payload["metadata"]["expected_tools"] == any_case.tool_contract["expected_tools"]
    assert upload_payload["metadata"]["tool_match_mode"] == "any"
    legacy_payload = _real_case_item()
    legacy_payload["expectedOutput"]["expected_tools"] = ["query"]
    legacy_payload["expectedOutput"]["tool_match_mode"] = "any"
    legacy_payload["metadata"]["expected_result"] = {"rows": [[1]]}
    legacy_payload["metadata"]["expected_values"] = {"value": 1}
    try:
        to_dataset_item_payload(DATASET_NAME, legacy_payload, allow_update=True)
    except ValueError:
        pass
    else:
        raise AssertionError("上传器必须拒绝四个契约字段的旧位置")
    try:
        to_dataset_item_payload(
            "aieval/data_analysis",
            _real_case_item(),
            allow_update=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("上传器必须拒绝非 real_cases dataset")

    invalid_items = [
        _real_case_item(tools=[]),
        _real_case_item(mode="some"),
        _real_case_item(mode="ANY"),
        _real_case_item(mode=" all "),
        {
            **_real_case_item(),
            "expected_output": _real_case_item()["expectedOutput"],
            "expectedOutput": None,
        },
        {**_real_case_item(), "input": json.dumps({"input": "字符串对象"})},
        {**_real_case_item(), "metadata": json.dumps({"case_id": "bad"})},
        {
            **_real_case_item(),
            "metadata": {
                **_real_case_item()["metadata"],
                "expected_tools": "toolA,toolB",
            },
        },
        _real_case_item(
            tools=["searchBizMetric", "mcp__srv__search_biz_metric"]
        ),
        _real_case_item(tools=["---"]),
        {**_real_case_item(), "id": "x" * 501},
        {
            **_real_case_item(),
            "input": {"input": "x" * 100_001},
        },
        {
            **_real_case_item(),
            "metadata": {"case_id": "x" * 501},
        },
        {
            "id": "legacy-item",
            "input": {
                "case_id": "legacy-001",
                "original_request": "旧格式问题",
            },
            "expectedOutput": {
                "case_id": "legacy-001",
                "main_query": "旧指标",
                "expected_result": {"columns": ["value"], "rows": [[1]]},
            },
            "metadata": {},
            "status": "ACTIVE",
        },
    ]
    for invalid in invalid_items:
        try:
            parse_eval_case(invalid, dataset_name=DATASET_NAME)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("非法或旧版 DatasetItem 必须在启动 Agent 前被拒绝")

    try:
        parse_eval_case(_real_case_item(), dataset_name="aieval/data_analysis")
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("parser 必须拒绝 real_cases 之外的数据集")

    normalized = normalize_config(
        {
            "dataset": DATASET_NAME,
            "run_id": "run-test",
            "codex_cwd": "",
            "limit": 0,
            "item_id": "",
            "app_server_request_timeout_seconds": 90,
            "app_server_turn_timeout_seconds": 1800,
            "max_concurrency": 1,
            "print_schemas": False,
        }
    )
    assert normalized["dataset"] == DATASET_NAME
    from aieval_runner.core.constants import AGENT_WORKSPACE_ROOT

    normalized_cwd = Path(normalized["codex_cwd"])
    assert normalized_cwd.parent == AGENT_WORKSPACE_ROOT.resolve()
    assert normalized_cwd.name == "run-test"
    with tempfile.TemporaryDirectory() as custom_dir:
        custom = normalize_config(
            {
                **normalized,
                "run_id": "run-custom",
                "codex_cwd": custom_dir,
            }
        )
        assert Path(custom["codex_cwd"]) == Path(custom_dir).resolve()
    invalid_config = dict(normalized, dataset="aieval/data_analysis")
    try:
        normalize_config(invalid_config)
    except ValueError:
        pass
    else:
        raise AssertionError("配置层必须在 Agent 启动前拒绝其他 dataset")


def check_events() -> None:
    import aieval_runner.evaluation.events as events_module
    from aieval_runner.evaluation.events import load_event_summary

    with tempfile.TemporaryDirectory() as temp_dir:
        event_path = Path(temp_dir) / "完整事件.jsonl"
        events = [
            _started_tool_event(
                call_id="call-search",
                server="metric-mcp-remote",
                tool="searchBizMetric",
            ),
            _completed_tool_event(
                call_id="call-search",
                server="metric-mcp-remote",
                tool="searchBizMetric",
                result={"status": "success", "data": []},
            ),
            _completed_tool_event(
                call_id="call-query",
                server="metabase",
                tool="query",
                result={"content": [{"type": "text", "text": '{"data":[]}'}]},
            ),
            _completed_tool_event(
                call_id="call-extra",
                server="other-service",
                tool="discoverDefinition",
                result={"status": "success"},
            ),
            _started_tool_event(
                call_id="call-incomplete",
                server="metric-mcp-remote",
                tool="constructQuery",
            ),
        ]
        _write_events(event_path, events)

        summary = load_event_summary(event_path)
        assert len(summary.tool_calls) == 3, "必须保留全部已结束 MCP 调用"
        assert [call["tool"] for call in summary.tool_calls] == [
            "searchBizMetric",
            "query",
            "discoverDefinition",
        ]
        assert all(call["status"] == "completed" for call in summary.tool_calls)
        assert all("result_statuses" not in call for call in summary.tool_calls)
        assert all("result" in call for call in summary.tool_calls)
        assert all("arguments" in call for call in summary.tool_calls)
        assert all("duration_ms" in call for call in summary.tool_calls)
        assert summary.started_execution_ids == {"call-incomplete"}
        assert "incomplete_execution_count=1" in (summary.error or "")

        invalid_path = Path(temp_dir) / "非法事件.jsonl"
        invalid_path.write_text("{not-json}\n", encoding="utf-8")
        try:
            load_event_summary(invalid_path)
        except ValueError as exc:
            assert "不是合法 JSON" in str(exc)
        else:
            raise AssertionError("非法 JSON 事件必须被拒绝")

        limits = {
            "MAX_EVENT_FILE_BYTES": events_module.MAX_EVENT_FILE_BYTES,
            "MAX_EVENT_LINE_CHARS": events_module.MAX_EVENT_LINE_CHARS,
            "MAX_EVENT_COUNT": events_module.MAX_EVENT_COUNT,
        }
        try:
            events_module.MAX_EVENT_FILE_BYTES = 1
            try:
                load_event_summary(event_path)
            except ValueError as exc:
                assert "文件超过" in str(exc)
            else:
                raise AssertionError("超大事件文件必须被拒绝")
            events_module.MAX_EVENT_FILE_BYTES = limits["MAX_EVENT_FILE_BYTES"]
            events_module.MAX_EVENT_LINE_CHARS = 10
            try:
                load_event_summary(event_path)
            except ValueError as exc:
                assert "行超过" in str(exc)
            else:
                raise AssertionError("超长事件行必须被拒绝")
            events_module.MAX_EVENT_LINE_CHARS = limits["MAX_EVENT_LINE_CHARS"]
            events_module.MAX_EVENT_COUNT = 1
            try:
                load_event_summary(event_path)
            except ValueError as exc:
                assert "事件数量" in str(exc)
            else:
                raise AssertionError("过多事件必须被拒绝")
        finally:
            for name, value in limits.items():
                setattr(events_module, name, value)

        deeply_nested_json = "[" * 10_000 + "0" + "]" * 10_000
        deeply_nested_event_path = Path(temp_dir) / "深层事件.jsonl"
        deeply_nested_event_path.write_text(deeply_nested_json + "\n", encoding="utf-8")
        try:
            load_event_summary(deeply_nested_event_path)
        except ValueError as exc:
            assert "不是合法 JSON" in str(exc)
        else:
            raise AssertionError("深层顶层 JSON 必须受控失败")


def _evaluate(
    event_path: Path,
    *,
    mode: str,
    tools: list[str],
) -> dict[str, Any]:
    from aieval_runner.core.models import AgentExecution, EvalRunContext
    from aieval_runner.datasets.cases import parse_eval_case
    from aieval_runner.evaluation import evaluate_case

    case = parse_eval_case(
        _real_case_item(mode=mode, tools=tools),
        dataset_name=DATASET_NAME,
    )
    execution = AgentExecution(
        trace_id="trace-real-cases",
        observation_id="observation-root",
        output={"answer": "不参与评分", "event_path": str(event_path)},
        trace_metadata={},
        mapping={"root_observation_id": "observation-root"},
    )
    context = EvalRunContext(
        eval_run_id="run-real-cases",
        dataset_name=DATASET_NAME,
        agent_version="回归",
        client="回归脚本",
        skill_version="回归",
        db_snapshot_id="不适用",
    )
    scores = evaluate_case(case, execution, context)
    assert {score.name for score in scores} == {
        "expected_tools_match",
        "tool_result_status",
    }, "只能生成两项工具契约评分"
    assert all(score.data_type == "BOOLEAN" for score in scores)
    return {score.name: score for score in scores}


def check_scoring() -> None:
    from aieval_runner.core.models import AgentExecution, EvalRunContext
    from aieval_runner.datasets.cases import parse_eval_case
    from aieval_runner.evaluation import evaluate_case
    from aieval_runner.evaluation.flow.hosted_experiment import run_hosted_dataset_experiment
    from aieval_runner.evaluation.flow.local_run import process_case
    from aieval_runner.evaluation.flow.records import judge_output_from_execution
    from aieval_runner.evaluation.flow.scoring import score_observation_id
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        any_path = root / "any.jsonl"
        _write_events(
            any_path,
            [
                _completed_tool_event(
                    call_id="call-any",
                    server="metric-mcp-remote",
                    tool="SEARCH-BIZ-METRIC",
                    result={"records": [], "total": 0},
                ),
                _completed_tool_event(
                    call_id="call-extra",
                    server="other",
                    tool="extraTool",
                    result={"status": "success"},
                ),
            ],
        )
        scores = _evaluate(
            any_path,
            mode="any",
            tools=["mcp__metric_mcp_remote__searchBizMetric", "missingTool"],
        )
        assert scores["expected_tools_match"].value is True
        assert scores["tool_result_status"].value is True
        assert all(
            score_observation_id(score, type("Execution", (), {"mapping": {"root_observation_id": "root"}})())
            == "root"
            for score in scores.values()
        )

        all_path = root / "all.jsonl"
        _write_events(
            all_path,
            [
                _completed_tool_event(
                    call_id="call-search-failed",
                    server="metric.mcp.remote",
                    tool="search_biz_metric",
                    result={"status": "failed"},
                ),
                _completed_tool_event(
                    call_id="call-search-retry",
                    server="metric.mcp.remote",
                    tool="search_biz_metric",
                    result={"status": "completed"},
                ),
                _completed_tool_event(
                    call_id="call-query",
                    server="metabase",
                    tool="query",
                    result={
                        "content": [
                            {
                                "type": "text",
                                "text": '{"result":{"status":"COMPLETED"}}',
                            }
                        ]
                    },
                ),
            ],
        )
        scores = _evaluate(
            all_path,
            mode="all",
            tools=["searchBizMetric", "mcp__metabase__query"],
        )
        assert scores["expected_tools_match"].value is True
        assert scores["tool_result_status"].value is True

        mixed_status_path = root / "mixed-status.jsonl"
        _write_events(
            mixed_status_path,
            [
                _completed_tool_event(
                    call_id="call-with-status",
                    server="metabase",
                    tool="query",
                    result={"status": "failed"},
                ),
                _completed_tool_event(
                    call_id="call-without-status",
                    server="metabase",
                    tool="query",
                    result={"data": []},
                ),
            ],
        )
        scores = _evaluate(
            mixed_status_path,
            mode="any",
            tools=["query"],
        )
        assert scores["tool_result_status"].value is True
        assert scores["tool_result_status"].metadata["observed_statuses"][
            "query"
        ] == ["completed"]

        failure_path = root / "failure.jsonl"
        _write_events(
            failure_path,
            [
                _completed_tool_event(
                    call_id="call-prefix-only",
                    server="metric-mcp-remote",
                    tool="searchBizMetricDetail",
                    result={"status": "success"},
                ),
                _completed_tool_event(
                    call_id="call-query-no-status",
                    server="metabase",
                    tool="query",
                    result={"data": []},
                ),
                _completed_tool_event(
                    call_id="call-query-error",
                    server="metabase",
                    tool="query",
                    result={"status": "success"},
                    error={"message": "调用失败"},
                ),
            ],
        )
        scores = _evaluate(
            failure_path,
            mode="all",
            tools=["searchBizMetric", "query"],
        )
        assert scores["expected_tools_match"].value is False
        assert "searchBizMetric" in scores["expected_tools_match"].metadata.get(
            "missing_tools", []
        )
        assert scores["tool_result_status"].value is False

        failed_lifecycle_path = root / "failed-lifecycle.jsonl"
        failed_lifecycle = _completed_tool_event(
            call_id="call-failed-lifecycle",
            server="metabase",
            tool="query",
            result={"status": "success"},
        )
        failed_lifecycle["params"]["item"]["status"] = "failed"
        _write_events(failed_lifecycle_path, [failed_lifecycle])
        scores = _evaluate(
            failed_lifecycle_path,
            mode="any",
            tools=["query"],
        )
        assert scores["expected_tools_match"].value is True
        assert scores["tool_result_status"].value is False
        for lifecycle_status in ("aborted", "timedOut", "interrupted", "inProgress"):
            unknown_lifecycle_path = root / f"lifecycle-{lifecycle_status}.jsonl"
            unknown_lifecycle = _completed_tool_event(
                call_id=f"call-{lifecycle_status}",
                server="metabase",
                tool="query",
                result={"status": "success"},
            )
            unknown_lifecycle["params"]["item"]["status"] = lifecycle_status
            _write_events(unknown_lifecycle_path, [unknown_lifecycle])
            scores = _evaluate(
                unknown_lifecycle_path,
                mode="any",
                tools=["query"],
            )
            assert scores["tool_result_status"].value is False

        for error_key in ("Error", "ERROR", "is_error", "is-error"):
            error_value: Any = True if "is" in error_key.lower() else "failed"
            error_variant_path = root / f"error-{error_key}.jsonl"
            _write_events(
                error_variant_path,
                [
                    _completed_tool_event(
                        call_id=f"call-{error_key}",
                        server="metabase",
                        tool="query",
                        result={"status": "success", error_key: error_value},
                    )
                ],
            )
            scores = _evaluate(
                error_variant_path,
                mode="any",
                tools=["query"],
            )
            assert scores["tool_result_status"].value is True
        assert "query" in json.dumps(
            scores["tool_result_status"].metadata, ensure_ascii=False
        )

        result_error_path = root / "result-error.jsonl"
        _write_events(
            result_error_path,
            [
                _completed_tool_event(
                    call_id="call-result-error",
                    server="metabase",
                    tool="query",
                    result={"status": "success", "error": "query failed"},
                )
            ],
        )
        scores = _evaluate(
            result_error_path,
            mode="any",
            tools=["query"],
        )
        assert scores["expected_tools_match"].value is True
        assert scores["tool_result_status"].value is True

        completed_with_error_path = root / "completed-with-error.jsonl"
        _write_events(
            completed_with_error_path,
            [
                _completed_tool_event(
                    call_id="call-completed-with-error",
                    server="metabase",
                    tool="query",
                    result={"error": "business payload"},
                    error={"message": "inconsistent protocol payload"},
                )
            ],
        )
        scores = _evaluate(
            completed_with_error_path,
            mode="any",
            tools=["query"],
        )
        assert scores["tool_result_status"].value is True

        all_with_failed_tool_path = root / "all-with-failed-tool.jsonl"
        failed_tool = _completed_tool_event(
            call_id="call-failed-tool",
            server="metric-mcp-remote",
            tool="searchBizMetric",
            result={"status": "success"},
        )
        failed_tool["params"]["item"]["status"] = "failed"
        _write_events(
            all_with_failed_tool_path,
            [
                failed_tool,
                _completed_tool_event(
                    call_id="call-completed-tool",
                    server="metabase",
                    tool="query",
                    result={"data": []},
                ),
            ],
        )
        scores = _evaluate(
            all_with_failed_tool_path,
            mode="all",
            tools=["searchBizMetric", "query"],
        )
        assert scores["expected_tools_match"].value is True
        assert scores["tool_result_status"].value is False

        conflicting_status_path = root / "conflicting-status.jsonl"
        _write_events(
            conflicting_status_path,
            [
                _completed_tool_event(
                    call_id="call-conflicting-status",
                    server="metabase",
                    tool="query",
                    result={
                        "status": "success",
                        "result": {"status": "failed"},
                    },
                )
            ],
        )
        scores = _evaluate(
            conflicting_status_path,
            mode="any",
            tools=["query"],
        )
        assert scores["tool_result_status"].value is True

        missing_lifecycle_status_path = root / "missing-lifecycle-status.jsonl"
        event = _completed_tool_event(
            call_id="call-with-result-status",
            server="metric-mcp-remote",
            tool="searchBizMetric",
            result={"status": "success"},
        )
        del event["params"]["item"]["status"]
        _write_events(missing_lifecycle_status_path, [event])
        scores = _evaluate(
            missing_lifecycle_status_path,
            mode="any",
            tools=["searchBizMetric"],
        )
        assert scores["tool_result_status"].value is False

        flow_case = parse_eval_case(_real_case_item(), dataset_name=DATASET_NAME)
        flow_context = EvalRunContext(
            eval_run_id="run-flow",
            dataset_name=DATASET_NAME,
            agent_version="回归",
            client="回归脚本",
            skill_version="回归",
            db_snapshot_id="不适用",
        )
        flow_execution = AgentExecution(
            trace_id="trace-flow",
            observation_id="tool-observation",
            output={"answer": "不参与评分", "event_path": str(any_path)},
            trace_metadata={},
            mapping={"root_observation_id": "root-observation"},
        )

        many_calls_path = root / "many-calls.jsonl"
        many_calls = [
            _completed_tool_event(
                call_id=f"call-extra-{index}",
                server="other",
                tool=f"extra{index}",
                result={"record": index},
            )
            for index in range(17)
        ]
        many_calls[8] = _completed_tool_event(
            call_id="call-critical-middle",
            server="metric-mcp-remote",
            tool="searchBizMetric",
            result={"records": ["关键结果"], "payload": "x" * 5_000},
        )
        _write_events(many_calls_path, many_calls)
        bounded_output = judge_output_from_execution(
            AgentExecution(
                trace_id="trace-many",
                observation_id=None,
                output={"answer": "关键结果", "event_path": str(many_calls_path)},
                trace_metadata={},
                mapping={},
            ),
            flow_case,
        )
        assert len(bounded_output["tool_evidence"]) == 16
        critical = next(
            evidence
            for evidence in bounded_output["tool_evidence"]
            if evidence["tool"] == "searchBizMetric"
        )
        assert "关键结果" in critical["result_excerpt"]
        assert critical["result_truncated"] is True
        assert bounded_output["tool_evidence_total"] == 17
        assert bounded_output["tool_evidence_truncated"] is True

        class FakeApi:
            def __init__(self) -> None:
                self.requests: list[tuple[str, str, Any]] = []

            def request(self, method: str, path: str, *, body: Any = None) -> dict[str, Any]:
                self.requests.append((method, path, body))
                return {"id": "dataset-run-item"}

        local_writes: list[tuple[Any, Any]] = []
        local_api = FakeApi()
        local_result = process_case(
            flow_case,
            flow_context,
            api=local_api,
            config=SimpleNamespace(),
            run_agent=lambda *_args, **_kwargs: flow_execution,
            evaluate=evaluate_case,
            write_score_fn=lambda score, **kwargs: local_writes.append(
                (score, kwargs.get("observation_id"))
            ),
        )
        assert len(local_api.requests) == 1
        assert {score.name for score, _ in local_writes} == {
            "expected_tools_match",
            "tool_result_status",
        }
        assert all(observation_id == "root-observation" for _, observation_id in local_writes)
        assert len(local_result["scores"]) == 2

        class FakeDataset:
            def __init__(self, item: dict[str, Any]) -> None:
                self.items = [item]

            def run_experiment(self, *, task: Any, **_kwargs: Any) -> Any:
                item_results = []
                for item in self.items:
                    output = asyncio.run(task(item=item))
                    item_results.append(
                        SimpleNamespace(
                            item=item,
                            trace_id="experiment-trace",
                            dataset_run_id="dataset-run",
                            output=output,
                        )
                    )
                return SimpleNamespace(
                    name="experiment",
                    run_name="run-flow",
                    experiment_id="experiment-id",
                    dataset_run_id="dataset-run",
                    dataset_run_url="https://example.invalid/run",
                    item_results=item_results,
                )

        class FakeLangfuse:
            def __init__(self, item: dict[str, Any]) -> None:
                self.dataset = FakeDataset(item)

            def get_dataset(self, _name: str) -> FakeDataset:
                return self.dataset

            def get_current_trace_id(self) -> str:
                return "experiment-trace"

            def get_current_observation_id(self) -> str:
                return "experiment-root"

        hosted_writes: list[tuple[Any, Any]] = []
        hosted_results = run_hosted_dataset_experiment(
            flow_context,
            SimpleNamespace(
                dataset=DATASET_NAME,
                item_id="",
                limit=0,
                max_concurrency=1,
            ),
            get_langfuse_sdk_fn=lambda _config: FakeLangfuse(_real_case_item()),
            run_agent=lambda *_args, **_kwargs: flow_execution,
            evaluate=evaluate_case,
            write_score_fn=lambda score, **kwargs: hosted_writes.append(
                (score, kwargs.get("observation_id"))
            ),
        )
        assert hosted_results[0]["status"] == "success"
        judge_output = hosted_results[0]["experiment"]["item_result"]["output"]
        assert judge_output["answer"] == "不参与评分"
        assert judge_output["tool_evidence"][0]["tool"] == "SEARCH-BIZ-METRIC"
        assert judge_output["tool_evidence"][0]["status"] == "completed"
        assert "records" in judge_output["tool_evidence"][0]["result_excerpt"]
        assert judge_output["tool_evidence_truncated"] is False
        assert {score.name for score, _ in hosted_writes} == {
            "expected_tools_match",
            "tool_result_status",
        }
        assert all(observation_id == "root-observation" for _, observation_id in hosted_writes)

        degraded_results = run_hosted_dataset_experiment(
            flow_context,
            SimpleNamespace(
                dataset=DATASET_NAME,
                item_id="",
                limit=0,
                max_concurrency=1,
            ),
            get_langfuse_sdk_fn=lambda _config: FakeLangfuse(_real_case_item()),
            run_agent=lambda *_args, **_kwargs: flow_execution,
            evaluate=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("评分器故障")
            ),
            write_score_fn=lambda *_args, **_kwargs: None,
        )
        assert degraded_results[0]["status"] == "success"
        assert degraded_results[0]["deterministic_score_status"] == "failed"
        assert degraded_results[0]["experiment"]["item_result"]["output"]["answer"] == "不参与评分"


def check_provisioning() -> None:
    import scripts.provision_real_cases_judges as provision

    state: dict[str, Any] = {
        "evaluators": [],
        "rules": [{"id": "old-rule", "name": "eval-prompt-real-cases"}],
        "calls": [],
    }

    def fake_request(
        method: str, path: str, *, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        state["calls"].append((method, path, body))
        if method == "GET" and path.startswith("/datasets"):
            return {"data": [{"id": "dataset-real", "name": provision.DATASET_NAME}]}
        if method == "GET" and path.startswith("/unstable/evaluators"):
            return {"data": list(state["evaluators"])}
        if method == "GET" and path.startswith("/unstable/evaluation-rules"):
            return {"data": list(state["rules"])}
        if method == "POST" and path == "/unstable/evaluators":
            item = {
                **(body or {}),
                "id": f"evaluator-{len(state['evaluators']) + 1}",
                "scope": "project",
                "version": 1,
                "variables": provision._variables(str((body or {}).get("prompt") or "")),
            }
            state["evaluators"].append(item)
            return item
        if method == "POST" and path == "/unstable/evaluation-rules":
            item = {**(body or {}), "id": f"rule-{len(state['rules']) + 1}"}
            state["rules"].append(item)
            return item
        if method == "PATCH" and path.startswith("/unstable/evaluation-rules/"):
            rule_id = path.rsplit("/", 1)[-1]
            item = {**(body or {}), "id": rule_id}
            state["rules"] = [item if rule.get("id") == rule_id else rule for rule in state["rules"]]
            return item
        if method == "DELETE" and path.startswith("/unstable/evaluation-rules/"):
            rule_id = path.rsplit("/", 1)[-1]
            state["rules"] = [rule for rule in state["rules"] if rule.get("id") != rule_id]
            return {}
        raise AssertionError(f"未预期请求：{method} {path}")

    original_request = provision._api_request
    original_load_env = provision._load_env_file
    try:
        provision._api_request = fake_request
        provision._load_env_file = lambda _path: None
        provision.main()
        provision.main()
    finally:
        provision._api_request = original_request
        provision._load_env_file = original_load_env

    evaluator_posts = [
        call for call in state["calls"]
        if call[0] == "POST" and call[1] == "/unstable/evaluators"
    ]
    assert len(evaluator_posts) == 3, "连续执行不得重复创建 evaluator"
    assert {rule["name"] for rule in state["rules"]} == set(provision.JUDGES)
    for rule in state["rules"]:
        assert rule["target"] == "experiment"
        assert rule["sampling"] == 1
        assert rule["filter"][0]["value"] == ["dataset-real"]
    mappings = {rule["name"]: rule["mapping"] for rule in state["rules"]}
    assert mappings["real_cases_answer_correctness"][1] == {
        "variable": "expected_result",
        "source": "expected_output",
        "jsonPath": "$.expected_result",
    }
    assert mappings["real_cases_answer_values"][1] == {
        "variable": "expected_values",
        "source": "expected_output",
        "jsonPath": "$.expected_values",
    }
    assert mappings["real_cases_execution_quality"][1]["source"] == "metadata"
    assert mappings["real_cases_execution_quality"][2]["source"] == "metadata"


def check_workspace() -> None:
    import os
    from unittest.mock import patch

    from aieval_runner.agent.workspace import prepare_agent_workspace
    from aieval_runner.core.constants import AGENT_WORKSPACE_ROOT
    from aieval_runner.runner.config import load_config

    with patch.dict(
        os.environ,
        {
            "AIEVAL_CODEX_CWD": "",
            "LANGFUSE_DATASET_NAME": DATASET_NAME,
        },
        clear=False,
    ):
        config = load_config(["--run-id", "workspace-profile-test"])
    assert Path(config.codex_cwd).parent == AGENT_WORKSPACE_ROOT.resolve()
    assert config.response_profile == "complete_data_result_v1"
    assert len(config.response_instructions_sha256) == 64

    explicit_codex = Path(sys.executable).resolve()
    with patch.dict(
        os.environ,
        {
            "AIEVAL_CODEX_BIN": str(explicit_codex),
            "AIEVAL_CODEX_CWD": "",
            "LANGFUSE_DATASET_NAME": DATASET_NAME,
        },
        clear=False,
    ):
        explicit_config = load_config(["--run-id", "workspace-explicit-codex-test"])
    assert Path(explicit_config.codex_bin) == explicit_codex

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        template = root / "project" / "AGENTS.override.md"
        template.parent.mkdir(parents=True)
        template.write_text("# 数据查询回答规范\n\n必须完整返回结果。\n", encoding="utf-8")
        target = root / "runtime"

        state = prepare_agent_workspace(
            target,
            template_path=template,
            response_profile="complete_data_result_v1",
        )

        assert state.cwd == target.resolve()
        assert state.response_profile == "complete_data_result_v1"
        assert len(state.instructions_sha256) == 64
        assert (target / "AGENTS.override.md").read_bytes() == template.read_bytes()

        conflicting = root / "custom-runtime"
        conflicting.mkdir()
        existing = conflicting / "AGENTS.override.md"
        existing.write_text("# 用户自己的覆盖\n", encoding="utf-8")
        try:
            prepare_agent_workspace(
                conflicting,
                template_path=template,
                response_profile="complete_data_result_v1",
            )
        except FileExistsError as exc:
            assert "AGENTS.override.md" in str(exc)
        else:
            raise AssertionError("不同内容的覆盖文件必须拒绝同步")
        assert existing.read_text(encoding="utf-8") == "# 用户自己的覆盖\n"

        leaking_template = root / "leaking" / "AGENTS.override.md"
        leaking_template.parent.mkdir()
        leaking_template.write_text("请参考 expected_values 完成回答。", encoding="utf-8")
        try:
            prepare_agent_workspace(
                root / "leaking-runtime",
                template_path=leaking_template,
            )
        except ValueError as exc:
            assert "expected_values" in str(exc)
        else:
            raise AssertionError("覆盖模板不得包含评测契约")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        choices=SECTIONS,
        action="append",
        help="只运行指定部分；可重复传入，默认运行全部部分",
    )
    args = parser.parse_args()
    selected = args.section or list(SECTIONS)
    checks: dict[str, Callable[[], None]] = {
        "dataset": check_dataset,
        "events": check_events,
        "scoring": check_scoring,
        "provisioning": check_provisioning,
        "workspace": check_workspace,
    }
    for section in selected:
        checks[section]()
        print(f"通过：{section}")
    print("real_cases 回归检查通过")


if __name__ == "__main__":
    main()
