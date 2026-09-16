"""一次性迁移有效评测数据：移除 StarRocks MCP 依赖并稳定题目与 Gold。"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]

DIRECT_QUERY_TOOLS = [
    "openmetadata.search_metadata",
    "metabase.search",
    "metabase.get_table",
    "metabase.query",
]
METRIC_QUERY_TOOLS = [
    "metric-mcp-remote.searchMetricApp",
    "metric-mcp-remote.searchMetricAppQueryResult",
    "metric-mcp-remote.searchSqlByMetricTypeAndNameExact",
]

MAIN_QUERIES = {
    "L1-REAL-01": "dws_tra_warehouse_good_daily_hi turnover_by_warehouse",
    "L1-REAL-02": "dws_tra_warehouse_good_daily_hi turnover_by_warehouse",
    "L1-REAL-04": "dws_tra_warehouse_good_daily_hi inventory_volume_3pl_fs_split",
    "L1-REAL-09": "dws_tra_warehouse_good_daily_hi inventory_volume_3pl_fs_split",
    "L1-REAL-05": "dws_tra_warehouse_good_daily_hi inventory_volume_by_warehouse",
    "L1-REAL-06": "platform_data_quality_by_create_month",
    "L1-REAL-07": "ads_client_good_overall_mv month_end_inventory_valuation_2024",
    "L1-REAL-10": "tms_quote_template_usps_ground_advantage1_customer_binding",
    "L1-REAL-12": "ods_loctek_crm_custom_approved_after_2025_10_16_type_distribution",
    "L1-REAL-13": "ods_oms_inventory_flow_sku_flow_summary_2024_12_31_to_2026_07_08",
    "L1-REAL-14": "ods_oms_toc_order_express_info_usps_tracking_184_label_url",
    "L1-REAL-15": "oms_sales_back_order barcode compare with original order goods code by tracking_no",
    "L1-REAL-16": "HOU07_missing_first_scan_report_2026_06_30",
    "L1-METRIC-CUSTOMER-ACTIVE-002": "海外仓全局经营分析",
    "L1-METRIC-CUSTOMER-ACTIVE-003": "海外仓全局经营分析",
}

REQUESTS = {
    "L1-REAL-01": (
        "请从 dws_tra_warehouse_good_daily_hi 查询数据日期为 2026-06-19、"
        "SKU 为 LT-DHW019BN014C 的各仓库30日周转率，并汇总整体加权周转率。"
    ),
    "L1-REAL-02": (
        "请从 dws_tra_warehouse_good_daily_hi 查询数据日期为 2026-06-19、"
        "SKU 为 LT-DHW019BN014C 的各仓库30日周转率，并汇总整体加权周转率。"
    ),
    "L1-REAL-04": (
        "查询 dws_tra_warehouse_good_daily_hi 在 2026-06-29 的数据："
        "inventory_good_volume_d 和 outbound_good_volume_d 单位均为立方米。"
        "按客户代码关联 dim_customer_hf，business_type_name=海外仓归为3PL，其他归为FS；"
        "分别统计客户数、库存体积、出库体积，并按64立方米/柜折算库存柜数和出库柜数。"
    ),
    "L1-REAL-09": (
        "查询 dws_tra_warehouse_good_daily_hi 在 2026-06-29 的数据，"
        "按客户代码关联 dim_customer_hf；business_type_name=海外仓归为3PL，其他归为FS。"
        "分别统计客户数、库存数量、库存体积，并按64立方米/柜折算柜数。"
    ),
    "L1-REAL-05": (
        "查询 dws_tra_warehouse_good_daily_hi 在 2026-06-29 每个仓库的在库库存体积；"
        "inventory_good_volume_d 单位为立方米，并按64立方米/柜计算柜数。"
    ),
    "L1-REAL-06": (
        "分析 ods.ods_oms_toc_order 与 ods.ods_tms_external_order 的 platform 字段数据质量。"
        "按创建月份倒序输出每月总行数、空值率和缺失率，统计范围截止到 2026-06-30；"
        "先确认两张表真实的创建时间字段和 platform 字段类型。"
    ),
    "L1-REAL-10": (
        "说明如何基于截至 2026-07-08 23:59:59 的历史绑定记录，精确判断曾绑定产品 "
        "usps_ground_advantage1、但尚未绑定价卡模版“USPS 2026.7客户VIP报价（3lbs内）”"
        "的客户；模版生效时间为 2026-07-12 00:00:00 至 2027-01-31 23:59:59。"
        "请给出真实表结构确认要点、筛选与去重口径，以及应输出的字段，不返回客户数量。"
    ),
    "L1-REAL-11": (
        "分析 2026-06-29 至 2026-07-05 UPS 的整体发货情况，按天输出包裹量，"
        "并汇总服务类型和币种。"
    ),
    "L1-REAL-12": (
        "说明如何精确查询 crm_custom 中创建时间晚于 2025-10-16 00:00:00、"
        "不晚于 2026-07-08 23:59:59、审批通过且未删除的客户，并按 custom_type "
        "和 business_type 分组。请给出真实字段确认要点、筛选与分组口径，以及应输出的"
        "字段，不返回会随数据修订变化的客户数量。"
    ),
    "L1-REAL-13": (
        "按SKU汇总 2024-12-31 至 2026-07-08 的库存流水，输出流水行数、数量合计、"
        "最早和最晚库存日期。SKU为：GEWX-ET3-SL、GEWX-ET3-WH、GEWX-ET3-BK、"
        "GEWX-ET2-WH、GEWX-ET2-BK、GEWX-ET3T-BK-2、GEWX-ET3T-BK-1、"
        "GEWX-ET223H(IB)(2/2)-WHT。"
    ),
    "L1-REAL-14": (
        "说明如何精确查询 oms_toc_order_express_info 中截至 2026-07-08 23:59:59 创建、"
        "未删除、承运商为 USPS、tracking_no 包含 184 且 label_url 非空的订单。"
        "请给出真实字段确认要点、筛选与去重口径，以及记录数、订单数、去重跟踪号数和"
        "最新跟踪号对应的输出字段，不返回会随数据修订变化的具体数值。"
    ),
    "L1-REAL-16": (
        "使用 Metabase 查询 HOU07 仓库在 2026-06-30 已发货的有效订单，"
        "以 delivery_time 为发货时间，a_scan_time 为空表示没有 instant/第一枪物流轨迹；"
        "输出订单总数、缺失第一枪数量、缺失比例并给出简短报告。"
    ),
    "L1-METRIC-CUSTOMER-ACTIVE-005": (
        "查询“客户实时余额_额度_库存体积与估值”的指标定义和返回字段结构，"
        "列出16个原始字段及其业务含义；不要查询或引用会随运行时间变化的实时数值，"
        "也不要把 create_volume_sum、standard_create_volume_sum 或 "
        "create_volume_value_sum 解释成在途体积。"
    ),
    "L1-METRIC-CUSTOMER-ACTIVE-006": (
        "查询“统计时间内的去重活跃客户数”的指标定义，说明默认统计周期是否在 SQL 中"
        "显式限定为2026年，并说明全局去重应采用 overall distinct custom_code，"
        "不能把市场或业务类型分组结果直接相加。不要返回运行时动态计数。"
    ),
    "L1-METRIC-CUSTOMER-ACTIVE-008": (
        "查询 2026-06-01 华北业务部韩国市场的客户行为。必须按完整日期、区域和市场条件"
        "执行精确查询；若结果为零，需要用完整查询结果证明，不能根据分页或缺少目标行推断。"
    ),
    "L1-METRIC-CUSTOMER-ACTIVE-009": (
        "查询 2026-06-01 华北业务部韩国市场的去重活跃客户数。必须按 distinct custom_code"
        " 的口径执行精确查询；若结果为零，需要用完整查询结果证明，不能根据分页或缺少目标行推断。"
    ),
    "L1-METRIC-CUSTOMER-ACTIVE-010": (
        "查询“多维快递财务收入成本”的指标定义和明细字段结构，说明稳定获取前50行时"
        "必须固定哪些日期、批次和排序条件；不要返回会随运行时间变化的当前前50行数值。"
    ),
    "L1-META-CONSTRAINED-001": (
        "在企业数据目录的 dws 数据中查找客户行为相关、owner 为 HAYLEYHU/hurunli 的表，"
        "列出2个候选并说明日粒度和月粒度差异。"
    ),
}

SCHEMA_ONLY_NO_ROWS = {
    "L1-REAL-10",
    "L1-REAL-12",
    "L1-REAL-14",
    "L1-METRIC-CUSTOMER-ACTIVE-005",
    "L1-METRIC-CUSTOMER-ACTIVE-006",
    "L1-METRIC-CUSTOMER-ACTIVE-010",
}

FOCUS_OVERRIDES = {
    "L1-REAL-10": [
        "answer:确认真实表结构",
        "filter:历史绑定截止2026-07-08 23:59:59",
        "answer:筛选与去重口径及输出字段",
        "rule:不返回动态客户数量",
    ],
    "L1-REAL-12": [
        "answer:确认真实字段",
        "filter:create_time>2025-10-16且<=2026-07-08 23:59:59",
        "answer:custom_type,business_type分组口径及输出字段",
        "rule:不返回动态客户数量",
    ],
    "L1-REAL-14": [
        "answer:确认真实字段",
        "filter:USPS,tracking_no包含184,label_url非空,deleted=0",
        "answer:四个汇总输出字段及去重口径",
        "rule:不返回动态数值",
    ],
    "L1-METRIC-CUSTOMER-ACTIVE-005": [
        "answer:列出16个原始字段及业务含义",
        "rule:不把create_volume相关字段解释为在途体积",
        "rule:不返回运行时动态数值",
    ],
    "L1-METRIC-CUSTOMER-ACTIVE-006": [
        "answer:说明默认统计周期是否显式限定2026年",
        "answer:overall distinct custom_code",
        "rule:不相加分组distinct结果",
    ],
    "L1-METRIC-CUSTOMER-ACTIVE-008": [
        "filter:stats_dt=2026-06-01",
        "filter:effective_area_name=华北",
        "filter:market=韩国/Korea",
        "rule:零值必须由完整精确查询证明",
    ],
    "L1-METRIC-CUSTOMER-ACTIVE-009": [
        "filter:stats_dt=2026-06-01",
        "filter:effective_area_name=华北",
        "filter:market=韩国/Korea",
        "measure:distinct custom_code",
        "rule:零值必须由完整精确查询证明",
    ],
    "L1-METRIC-CUSTOMER-ACTIVE-010": [
        "answer:明细字段结构",
        "answer:稳定查询需要固定日期、批次和排序",
        "rule:不返回运行时动态数值",
    ],
}


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _case_id(item: Dict[str, Any]) -> str:
    expected = item.get("expectedOutput")
    if isinstance(expected, dict) and expected.get("case_id"):
        return str(expected["case_id"])
    return str(item.get("case_id") or "")


def _tool_dicts(names: Iterable[str]) -> List[Dict[str, Any]]:
    return [{"tool_name": name, "arguments": {}} for name in names]


def _raw_tool_names(value: Any) -> List[str]:
    """兼容原始 JSON 字符串，并修复旧迁移产生的单字符工具数组。"""

    if isinstance(value, str):
        value = json.loads(value)
    elif (
        isinstance(value, list)
        and value
        and all(
            isinstance(item, dict)
            and len(str(item.get("tool_name") or item.get("name") or "")) <= 1
            for item in value
        )
    ):
        serialized = "".join(
            str(item.get("tool_name") or item.get("name") or "") for item in value
        )
        value = json.loads(serialized)
    if not isinstance(value, list):
        raise ValueError("expected_tools 必须是数组或可解析为数组的 JSON 字符串")
    names = [
        str(tool.get("tool_name") or tool.get("name") or "").strip()
        if isinstance(tool, dict)
        else str(tool).strip()
        for tool in value
    ]
    if any(not name or "." not in name for name in names):
        raise ValueError(f"expected_tools 包含非法工具名: {names!r}")
    return names


def _tools_for(case_id: str, current: Iterable[str]) -> List[str]:
    current_list = [str(name) for name in current]
    if case_id == "L1-REAL-15":
        return DIRECT_QUERY_TOOLS[:-1]
    if case_id.startswith("L1-REAL-") and any(
        "starrocks" in name.lower() for name in current_list
    ):
        return DIRECT_QUERY_TOOLS
    if case_id in {
        "L1-METRIC-CUSTOMER-ACTIVE-002",
        "L1-METRIC-CUSTOMER-ACTIVE-003",
        "L1-METRIC-CUSTOMER-ACTIVE-008",
        "L1-METRIC-CUSTOMER-ACTIVE-009",
    }:
        return METRIC_QUERY_TOOLS + DIRECT_QUERY_TOOLS
    return [name for name in current_list if "starrocks" not in name.lower()]


def _migrate_active_item(item: Dict[str, Any]) -> None:
    case_id = _case_id(item)
    expected = item.get("expectedOutput")
    if not isinstance(expected, dict):
        return
    if case_id in REQUESTS:
        item.setdefault("input", {})["original_request"] = REQUESTS[case_id]
    if case_id in MAIN_QUERIES:
        expected["main_query"] = MAIN_QUERIES[case_id]
    expected["expected_tools"] = _tools_for(
        case_id,
        expected.get("expected_tools") or [],
    )
    if case_id in FOCUS_OVERRIDES:
        expected["evaluation_focus"] = FOCUS_OVERRIDES[case_id]
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("clarification_context", None)


def _migrate_raw_item(item: Dict[str, Any]) -> None:
    case_id = _case_id(item)
    if case_id in REQUESTS:
        item["user_query"] = REQUESTS[case_id]
        item["original_request"] = REQUESTS[case_id]
    if case_id in MAIN_QUERIES:
        item["main_query"] = MAIN_QUERIES[case_id]
    current_names = _raw_tool_names(item.get("expected_tools") or [])
    item["expected_tools"] = _tool_dicts(_tools_for(case_id, current_names))
    if case_id in FOCUS_OVERRIDES:
        item["evaluation_focus"] = FOCUS_OVERRIDES[case_id]
    elif isinstance(item.get("evaluation_focus"), list):
        item["evaluation_focus"] = [
            focus.replace("query:StarRocks ", "query:")
            if isinstance(focus, str)
            else focus
            for focus in item["evaluation_focus"]
        ]
    item.pop("clarification_context", None)
    if case_id in SCHEMA_ONLY_NO_ROWS:
        result = item.get("expected_result")
        if isinstance(result, dict):
            result["rows"] = []


def _remove_removed_tool_provenance(value: Any) -> Any:
    if isinstance(value, list):
        cleaned = []
        for item in value:
            serialized = json.dumps(item, ensure_ascii=False).lower()
            if (
                "mcp-server-starrocks" in serialized
                or "mcp_server_starrocks" in serialized
                or (
                    isinstance(item, str)
                    and (
                        "starrocks:" in item.lower()
                        or "query:starrocks" in item.lower()
                    )
                )
            ):
                continue
            cleaned.append(_remove_removed_tool_provenance(item))
        return cleaned
    if isinstance(value, dict):
        return {
            key: _remove_removed_tool_provenance(nested)
            for key, nested in value.items()
            if key not in {"gold_fix_reason"}
        }
    if isinstance(value, str) and "starrocks_" in value.lower():
        return value.replace("starrocks_", "query_").replace("StarRocks_", "query_")
    return value


def _migrate_gold(items: Dict[str, Dict[str, Any]]) -> None:
    if "L1-REAL-01" in items and "L1-REAL-02" in items:
        items["L1-REAL-01"] = copy.deepcopy(items["L1-REAL-02"])

    for case_id, gold in items.items():
        contract = gold.get("contract")
        if not isinstance(contract, dict):
            continue
        contract.pop("alternative_expected_results", None)
        contract.pop("accepted_main_queries", None)
        contract.pop("gold_fix_reason", None)
        if "judge_guidance" in contract and "starrocks" in str(
            contract["judge_guidance"]
        ).lower():
            contract.pop("judge_guidance", None)
        cleaned = _remove_removed_tool_provenance(contract)
        contract.clear()
        contract.update(cleaned)

        if case_id in SCHEMA_ONLY_NO_ROWS:
            result = gold.get("expected_result")
            if isinstance(result, dict):
                result["rows"] = []
            contract["score_mode"] = "schema_only"
            contract["row_count"] = 0
            contract.pop("must_include", None)
            contract.pop("expected_answer_key_points", None)

        filters = contract.setdefault("filters", {})
        if not isinstance(filters, dict):
            filters = {}
            contract["filters"] = filters
        if case_id == "L1-REAL-10":
            filters["snapshot_cutoff"] = "2026-07-08 23:59:59"
            filters.pop("accepted_active_subset", None)
        elif case_id == "L1-REAL-12":
            filters["create_time_lte"] = "2026-07-08 23:59:59"
        elif case_id == "L1-REAL-14":
            filters["create_time_lte"] = "2026-07-08 23:59:59"
        elif case_id == "L1-METRIC-CUSTOMER-ACTIVE-009":
            filters.update(
                {
                    "effective_area_name": "华北",
                    "market_cn_name": "韩国",
                    "distinct_key": "custom_code",
                    "active_customer_rule": "指标应用原始活跃客户口径",
                    "zero_requires_complete_query_result": True,
                }
            )


def main() -> int:
    active_paths = [
        ROOT / "docs" / "l1_real_completed_items.json",
        ROOT / "docs" / "optimized_dataset.json",
    ]
    for path in active_paths:
        document = _read(path)
        for item in document.get("items") or []:
            _migrate_active_item(item)
        _write(path, document)

    raw_path = ROOT / "docs" / "optimized_dataset_raw.json"
    raw_document = _read(raw_path)
    for item in raw_document.get("items") or []:
        _migrate_raw_item(item)
    _write(raw_path, raw_document)

    for path in [
        ROOT / "docs" / "l1_real_gold_results.json",
        ROOT / "docs" / "gold_results.json",
    ]:
        document = _read(path)
        _migrate_gold(document.get("items") or {})
        _write(path, document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
