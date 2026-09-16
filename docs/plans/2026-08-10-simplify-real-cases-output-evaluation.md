# `aieval/real_cases` 工具契约评测重构战术计划

## 目标

保持 `external_eval_runner.py -> aieval_runner.runner.cli` 与 app-server/Langfuse experiment 主链路不变，将运行时数据模型和确定性评分收缩为 `aieval/real_cases` 的两项工具契约：`expected_tools_match` 与 `tool_result_status`。

## 约束

- 当前工作区已有用户未提交修改；每一步只编辑列出的文件，不重置、不覆盖无关差异。
- 不使用 `/opsx:apply`。
- 先写失败回归，再实现最小代码使其通过。
- 不删除含用户未提交内容的历史文件；先解除运行时引用，最终根据引用扫描决定是否安全删除。

## 1. 建立新的失败回归（约 5 分钟）

编辑 `scripts/check_real_cases_evaluation.py`，加入三个真实结构 fixture：

- `input.input + metadata.case_id + expectedOutput(any/success)`；
- `input.input + metadata.case_id + expectedOutput(all/completed)`；
- 缺少字段或旧格式 item。

加入直接 status、`content[].text` JSON status、缺失 status、error、多次调用、额外工具、started 未 completed 的事件 fixture。

运行：

```powershell
python scripts/check_real_cases_evaluation.py
```

预期：因 parser/evaluator 尚不存在而失败。

## 2. 收缩配置与数据模型（每步约 2–5 分钟）

1. 编辑 `aieval_runner/core/constants.py`：增加唯一数据集常量，默认 dataset 改为 `aieval/real_cases`，schema 移除 Gold/表格字段并加入三个工具契约字段。
2. 编辑 `aieval_runner/core/models.py`：从 `RunnerConfig` 移除 `gold_results_json_path`。
3. 编辑 `aieval_runner/runner/config.py`：删除 `--gold-results-json` 和环境映射，在配置归一化阶段拒绝非 `aieval/real_cases`。
4. 编辑 `aieval_runner/datasets/validation.py`：改成 real_cases 严格结构验证，不再执行 StarRocks、相对时间和 Gold 校验。
5. 编辑 `aieval_runner/datasets/cases.py`：从 `input.input`、`metadata.case_id` 解析，保留且只保留三个 expectedOutput 字段。
6. 编辑 `aieval_runner/datasets/loading.py`：删除 Gold 加载/注入，直接返回经验证的远端或本地 raw item。
7. 编辑 `aieval_runner/evaluation/flow/dataset_items.py`、`hosted_experiment.py` 和 `aieval_runner/evaluation/__init__.py`：删除 experiment item 的 Gold 参数和兼容包装。

运行：

```powershell
python scripts/check_real_cases_evaluation.py --section dataset
python external_eval_runner.py --print-schemas
```

## 3. 重构完整事件解析（每步约 2–5 分钟）

1. 重写 `aieval_runner/evaluation/events.py` 的运行时职责：保留安全读取限制、全部 completed MCP calls、started 未完成集合和基础 error 诊断。
2. 增加 `canonical_tool_name`、expected tool 末段归一化与 `tool_name_matches`。
3. 后续决策已取代业务 result 状态解析：事件摘要只保留 app-server 统一生命周期 `status`，固定以 `completed` 表示成功。
4. completed call 不保留原始 `result`，避免不同 MCP 的业务字段语义和大型结果污染确定性评分。

运行：

```powershell
python scripts/check_real_cases_evaluation.py --section events
```

## 4. 替换评分器（每步约 2–5 分钟）

1. 新建 `aieval_runner/evaluation/evaluators/tool_contract.py`，实现 expected tool 到实际 calls 的映射及 any/all 聚合。
2. 实现 `evaluate_expected_tools_match`，返回缺失工具、实际工具和匹配关系 metadata。
3. 实现 `evaluate_tool_result_status`，固定要求完成事件的 `mcpToolCall.status=completed`；不读取业务 result/error，同工具多次调用允许一次成功。
4. 编辑 `aieval_runner/evaluation/evaluators/base.py`：删除 main query、Gold 和 contract 属性。
5. 重写 `aieval_runner/evaluation/evaluators/registry.py` 和 `__init__.py`：只注册两个 evaluator。
6. 编辑 `aieval_runner/evaluation/flow/scoring.py` 与 `aieval_runner/evaluation/__init__.py`：两个分数均写 experiment root observation，删除旧 score 分流。
7. 编辑 `aieval_runner/evaluation/flow/records.py`：结果摘要改为工具契约，不再读取 `main_query`。

运行：

```powershell
python scripts/check_real_cases_evaluation.py --section scoring
```

## 5. 清理运行路径与文档（每步约 2–5 分钟）

1. 使用 `rg` 检查 runner 运行路径是否仍引用 `gold_results`、表格评分和旧 evaluator；只解除运行时引用，不删除包含用户差异的遗留文件。
2. 编辑 `README.md`：默认 dataset、样例结构、命令和两个分数。
3. 重写 `docs/eval_case.md` 与 `docs/evaluation_score_logic.md`。
4. 编辑 `docs/external_eval_runner_design.md`：更新数据流与评分边界。
5. 更新 OpenSpec `tasks.md` 完成项。

## 6. 验证与审查

依次运行：

```powershell
python -m compileall external_eval_runner.py aieval_runner scripts/check_real_cases_evaluation.py
python scripts/check_real_cases_evaluation.py
python external_eval_runner.py --print-schemas
rg -n "sql_execution_success|numeric_accuracy|metric_definition_consistency|task_completed" aieval_runner/runner aieval_runner/datasets aieval_runner/evaluation/flow aieval_runner/evaluation/evaluators/registry.py
git diff --check
```

最后运行 `/multi-review`，修复 P0/P1 后重跑上述命令。
