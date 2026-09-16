# 评测 Agent 指令隔离实施计划

## 1. 模板同步 tracer bullet

1. 在 `scripts/check_real_cases_evaluation.py` 增加临时工作区同步测试，断言项目模板原样生成 `AGENTS.override.md`。
2. 运行 `--section workspace`，确认因接口缺失而失败。
3. 在 `aieval_runner/agent/workspace.py` 实现单一公开接口 `prepare_agent_workspace()`。
4. 新增 `config/eval-agent/AGENTS.override.md`，运行测试确认转绿。

## 2. 冲突与安全边界

1. 增加已有不同覆盖文件不得覆盖、模板缺失必须失败的测试。
2. 增加模板不得包含 expected 契约词和具体 case 的测试。
3. 最小实现冲突保护、模板校验与 SHA-256 返回值，逐项转绿。

## 3. run 级 cwd 与 runtime 接入

1. 增加配置测试：默认 cwd 包含安全化 `run_id`，显式 `--codex-cwd` 保持精确路径。
2. 调整 `runner/config.py` 默认 cwd 归一化。
3. 在 `aieval_runner/agent/app_server/backend.py` 创建 app-server runtime 前调用同步接口。
4. 增加回归断言：同步发生在 thread 启动前，用户 Prompt 仍只有原始问题。

## 4. 可追溯性

1. 增加 trace metadata 红灯测试，要求 `response_profile` 和模板 SHA-256。
2. 扩展 `RunnerConfig`、schema、sidecar 和 trace metadata 传递。
3. 运行 dataset/scoring/live trace 回归确认转绿。

## 5. 运行验证

1. 运行 `scripts/check_real_cases_evaluation.py`、`scripts/check_app_server_live_trace.py`、`compileall`、`git diff --check`。
2. 运行一条 Metabase 小结果 canary，检查回答字段和行数覆盖。
3. 运行全量 13 条、收集五项指标及 OAuth/504 错误。
4. 执行 `/multi-review` 并修复 P0/P1。
