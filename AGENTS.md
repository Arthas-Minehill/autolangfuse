# AGENTS.md

## 语言

所有对话、说明、提交信息、评审结论和文档输出都使用简体中文。代码标识符、命令和文件路径保持英文。

## 项目定位

本仓库是轻量化脚本项目，不再是 FastAPI 服务，也不再包含前端页面。核心入口是 `external_eval_runner.py`，辅助上传工具是 `scripts/upload_dataitems.py`。

## 维护规则

- 开始实现前先读取 `knowledge/index.md`；只按相关性读取具体知识文件，不要全量加载 `knowledge/`。
- 评测prompt不显式告诉ai这是一个评测，只模拟用户真实的输入
- 新逻辑优先保持在单文件脚本内，只有确实复用时才抽取模块。
- `.env` 只能保留占位配置，不要提交真实密钥。
- 运行输出、trace mapping、本地数据文件和缓存不得入库。


## 评测指标优化规则
- 所有线上数据都代表着从langfuse 的aieval/data_analysis 中的experiment 实例
- 线上适用LLM-as-a-judge 评测器，主要对应eval-prompt这个指标，其逻辑主要是使用metadata.expected_result metadata.expected_result_contract 中的数据和真实回答进行比较给分，可以在scores中筛选eval-prompt 获取其判断依据
- 本项目与langfuse高度结合，必要时可在langfuse中修改item
- 所有数据读取查询分析skill相关，直接优化本地的lecangs-data-collection 这个skill
- 如果需要对于线上数据、评测结果进行优化，按照 SKILL - Item数据 - 评测指标逻辑 这个顺序依次进行排查，优先级从高到低，最先优化SKILL
- SKILL中应该记录抽象的原则，而非具体的实现细节，原则是对所有数据都适用的
- 评测指标格式见eval_case.md
