# 确定性评测分数实施计划

对应 OpenSpec：`openspec/changes/add-deterministic-eval-scores/`

## 任务 1：扩展 DatasetItem 模型

1. 修改 `aieval_runner/constants.py`：
   - 将 schema 版本升级到 v2。
   - 在 `EXPECTED_OUTPUT_SCHEMA` 增加 `expected_result.columns` 与 `expected_result.rows`。
2. 修改 `aieval_runner/cases.py`：
   - 校验可选 `expected_result`。
   - 保留兼容字段 `expected_tools`、`evaluation_focus`。
3. 运行：

```powershell
python external_eval_runner.py --print-schemas
```

## 任务 2：实现事件与表格解析

1. 新增 `aieval_runner/eval_events.py`：
   - `load_completed_tool_calls(event_path)` 只返回完成事件。
   - `is_sql_tool_call(call)` 识别指标查询、指标 SQL 与 StarRocks。
   - `extract_metric_name(call)` 读取 `arguments.req.name` 或 `arguments.name`。
   - `extract_data_table(call)` 递归解析 `result.content[].text`、`preview` 和 JSON 字符串，返回 `{"columns": ..., "rows": ...}`。
2. 使用真实事件：

```text
.aieval/runs/eval-20260615-063117/app-server-events/L1-METRIC-CUSTOMER-ACTIVE-011.jsonl
```

验证最后一次结果能提取 10000 行查询中的 `data`，且指标名为 `月x市场x客户业务类型_客户行为`。

## 任务 3：实现列内容评分

1. 新增 `aieval_runner/table_scoring.py`。
2. 从 `scripts/evaluate.py` 等价迁移以下逻辑到内存表格：
   - Null、数值、日期、日期时间、字符串标准化。
   - 每列排序后生成 column signature。
   - signature 多重集匹配。
   - `score = max(0, recall - 0.01 * extra_columns / predicted_columns)`。
3. 保持行顺序和列名不影响普通列匹配。

## 任务 4：实现四个评测器

1. 新增 `aieval_runner/evaluators.py`，实现：

```python
def evaluate_sql_execution_success(events: list[dict]) -> EvaluatorResult: ...
def evaluate_numeric_accuracy(events: list[dict], expected_result: dict | None) -> EvaluatorResult: ...
def evaluate_metric_definition_consistency(events: list[dict], main_query: str) -> EvaluatorResult: ...
def evaluate_task_completed(execution: AgentExecution, events: list[dict]) -> EvaluatorResult: ...
```

2. 修改 `aieval_runner/models.py` 增加内部 `EvaluatorResult`。
3. 修改 `aieval_runner/evaluation.py`：
   - 删除 `evaluate_placeholder`。
   - 加载 `execution.output.event_path`。
   - 组合四个 `EvalScore`。
   - 在 score metadata 中记录使用的 tool、metric、表格规模和失败原因。

## 任务 5：扩展上传转换

1. 修改 `scripts/upload_dataitems.py`：
   - 从 `expected_tools` JSON 中提取工具名和最后一个数据查询工具的 `req.name`/`name`。
   - 解析 `expected_result_schema.columns`。
   - 保存 `source_sql` 和质量提示。
   - 新增 `--gold-results-json`。
2. 标准结果 JSON 格式：

```json
{
  "L1-METRIC-CUSTOMER-ACTIVE-011": {
    "columns": ["is_new_sum", "is_active_sum", "last_month_is_active_sum"],
    "rows": [[38, 1043, 1020]]
  }
}
```

3. 未传 `--commit` 时保持纯 dry-run。

## 任务 6：检查脚本与文档

1. 新增 `scripts/check_deterministic_evaluation.py`，使用临时事件 JSONL 和内存 DatasetItem 覆盖：
   - SQL 成功/失败。
   - 双层 JSON 文本中的 `data`。
   - 完全匹配、部分匹配、缺少 gold。
   - 指标匹配 `main_query`，或匹配 contract 中显式声明的可接受候选。
   - task 完成/失败。
   - `update.xlsx` 单行转换与 gold 注入。
2. 更新 `README.md` 与 `docs/external_eval_runner_design.md`。
3. 执行：

```powershell
python scripts/check_deterministic_evaluation.py
python external_eval_runner.py --print-schemas
python scripts/upload_dataitems.py --dataset aieval/data_analysis --file docs/update.xlsx --limit 1
openspec validate add-deterministic-eval-scores --strict
```

## 任务 7：评审与修复

1. 运行 `/multi-review` 对事件解析、评分正确性、兼容性和敏感数据处理做宏观评审。
2. 修复全部 P0/P1，并重新执行任务 6 的命令。
