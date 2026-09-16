# `real_cases` DatasetItem 格式

业务人员只需要填写问题和结果要求；工具调用、调用状态以及 Metabase 原始结果由运行时自动采集。

```json
{
  "input": {
    "input": "查询 2025 年 EUWE 和 EUKL 的 DPD 发货量"
  },
  "expectedOutput": {
    "expected_result": "返回 EUWE 和 EUKL 两个仓库 2025 年的 DPD 发货量，结果应按仓库区分。",
    "expected_values": {
      "row_count": 2,
      "columns": ["warehouse", "dpd_shipped_qty"]
    }
  },
  "metadata": {
    "case_id": "L1-META-QUERY-001",
    "type": "查询取数",
    "expected_tools": [
      "mcp__metabase__execute_query"
    ],
    "tool_match_mode": "all"
  }
}
```

字段说明：

- `input.input`：真实业务问题。
- `metadata.case_id`：样本唯一标识。
- `metadata.type`：样本业务类型。
- `expectedOutput.expected_result`：整体结果要求，使用自然语言描述。
- `expectedOutput.expected_values`：字段、数值和取数口径要求；没有精确值时填写空对象。
- `metadata.expected_tools` 与 `tool_match_mode`：用于确定性工具契约评分。

Langfuse 为本数据集绑定三个 experiment evaluator：

- `real_cases_answer_correctness`：比较问题、`expectedOutput.expected_result` 与最终回答；
- `real_cases_answer_values`：比较 `expectedOutput.expected_values` 与最终回答中的字段、数值；
- `real_cases_execution_quality`：比较 `metadata.expected_tools` 与运行输出中的 `tool_evidence`，判断工具结果能否支撑回答。

工具完整性与工具完成状态继续由 runner 写入 `expected_tools_match` 和 `tool_result_status` 两项确定性分数。LLM-as-judge 只负责比较问题、结果要求、精确值约束和整体回答。

运行 `python -X utf8 scripts/provision_real_cases_judges.py` 可幂等更新三个 evaluator 及其 experiment rule，并移除旧的 `eval-prompt-real-cases` 规则。

生命周期字段 `mcpToolCall.status=completed` 只表示调用结束，不代表 Metabase 业务执行成功。
