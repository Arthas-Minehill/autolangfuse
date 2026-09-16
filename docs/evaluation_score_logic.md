# real_cases 评测指标逻辑

## 数据来源

两项评分都读取本地完整 app-server 事件：

```text
.aieval/runs/<run_id>/app-server-events/<case_id>[__<dataset_item_id>].jsonl
```

系统收集所有 `item/completed` 的 `mcpToolCall`。额外工具不扣分；只有 `item/started` 而没有完成事件的调用不算命中，并记录为未结束执行。

预期工具契约只从 `metadata.expected_tools` 与 `metadata.tool_match_mode` 读取；结果契约只从 `expectedOutput.expected_result` 与 `expectedOutput.expected_values` 读取。旧位置字段会被 parser 拒绝。

## expected_tools_match

类型：`BOOLEAN`。

预期工具使用 MCP 全名，实际事件使用 `server + tool`。匹配时提取工具末段，移除非字母数字分隔符并忽略大小写，然后完整比较：

- `mcp__metric_mcp_remote__searchBizMetric` 匹配 `metric-mcp-remote / SEARCH-BIZ-METRIC`；
- `searchBizMetric` 不匹配 `searchBizMetricDetail`。

聚合规则：

- `any`：至少一个预期工具有完成调用；
- `all`：每个预期工具都有至少一个完成调用。

## tool_result_status

类型：`BOOLEAN`。

系统只检查与预期工具匹配的调用。单次调用通过必须同时满足：

1. 事件类型为 `item/completed`；
2. `mcpToolCall.status` 为 `completed`。

期望生命周期固定为 `completed`，不依赖 `expectedOutput` 中的状态字段。业务
`result` 内的同名字段也不参与评分。

同一预期工具多次调用时，至少一次符合状态即可。最终仍按 `tool_match_mode` 聚合：`any` 至少一个工具状态通过，`all` 每个工具状态都通过。

## 不参与评分的内容

- 最终自然语言回答；
- SQL 内容；
- 返回表格、数值与业务口径；
- `metadata.validation_rules`；
- Langfuse Tool Observation preview。

## 诊断 metadata

`expected_tools_match` 记录 `expected_tools`、`actual_tools`、`matched_tools` 和 `missing_tools`。

`tool_result_status` 记录固定的 `expected_lifecycle_status`、`successful_status_tools` 和 `observed_statuses`。

## MCP 统一执行状态（2026-08-10）

`tool_result_status` 只读取 Codex app-server 的 `mcpToolCall.status`。这是所有 MCP
经过 app-server 归一化后共有的生命周期字段：

- `completed`：工具调用正常完成；
- `failed`、`aborted`、`timedOut`、`interrupted`、`inProgress` 或缺失：不通过。

评分器直接固定检查生命周期 `completed`，不读取 Dataset output 中的状态字段。
评分也不读取 MCP 业务 `result` 内的 `status`、`success`、`error`、`code` 等字段，
因为这些字段在不同 MCP 中含义不一致，且可能只是业务记录自身的状态。

## LLM-as-a-judge

Langfuse experiment 额外运行三项 0～1 数值评分：

- `real_cases_answer_correctness`：整体回答是否满足 `expected_result`；
- `real_cases_answer_values`：最终回答中的字段、数值和口径是否满足 `expected_values`；
- `real_cases_execution_quality`：实际 MCP 工具结果是否足以支撑最终回答。

runner 写入 Langfuse 的 experiment output 是有界结构：`answer` 保存最终回答，`tool_evidence` 保存已完成 MCP 调用的工具名、生命周期、错误摘要和结果摘要。Judge 不依赖不可见的子 observation。
