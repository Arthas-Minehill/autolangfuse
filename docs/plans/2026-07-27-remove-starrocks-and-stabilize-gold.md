# 移除 StarRocks 评测依赖与稳定 Gold 实施计划

## 1. 评测器

- 修改 `scripts/check_deterministic_evaluation.py`，先断言旧 StarRocks 事件不再获得执行或口径分。
- 修改 `aieval_runner/evaluation/events.py`，删除 StarRocks server/tool 分类。
- 修改 `aieval_runner/evaluation/evaluators/metric_definition.py` 和 `sql_execution.py`，删除专用通过路径与文案。
- 运行 `python scripts/check_deterministic_evaluation.py`。

## 2. 数据校验

- 在 `aieval_runner/datasets/validation.py` 增加已移除工具、专用主查询前缀、相对时间和 Gold 行数校验。
- 在 `aieval_runner/datasets/cases.py` 接入校验。
- 增加 `scripts/validate_active_dataset.py`，覆盖本地两组有效数据及 Gold。
- 先构造失败样本，再运行确定性检查。

## 3. 题目与 Gold

- 迁移版本化的 `docs/l1_real_completed_items.json`、`docs/optimized_dataset.json` 和原始构建数据。
- 补全固定日期、单位、分组、关联、去重和计算参数。
- 将动态数值题固定截止时间；无法稳定复现的题改为 `schema_only` 且清空动态行值。
- 删除 Gold 中 StarRocks MCP 工具、专用前缀、provenance 和替代结果。
- 运行 `python scripts/validate_active_dataset.py`。

## 4. Skill

- 保留旧版 `lecangs-data-collection` 快照。
- 建立三类行为样本：过滤能力不足后继续取数、结构确认后写 SQL、缺行不等于零。
- 仅增加抽象原则，运行新版与旧版对照并生成评审结果。

## 5. 验收

- 更新说明文档并扫描有效路径中的失效依赖。
- 上传更新后的 31 条 DatasetItem。
- 回放重点失败项，随后执行全量。
- 等待异步处理并从 Langfuse 查询运行、trace 和 scores。
- 执行多角色评审，修复 P0/P1。
