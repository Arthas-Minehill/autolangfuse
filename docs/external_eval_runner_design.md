# external_eval_runner 工程设计

## 目标

`external_eval_runner.py` 保持薄入口，只运行 Langfuse Dataset `aieval/real_cases`，通过 Codex app-server 模拟真实会话，并以完整本地事件评估工具调用契约。

## 主流程

```text
Langfuse aieval/real_cases / --items-json
  -> 严格解析 input.input、metadata.case_id、expectedOutput
  -> codex app-server thread/turn
  -> 实时写 Agent / Generation / Tool trace
  -> 落盘完整 event JSONL
  -> expected_tools_match + tool_result_status
  -> DatasetRunItem + 两个 BOOLEAN Score
```

Agent prompt 只包含 `input.input`。预期工具、状态和 metadata 不会进入会话。

## 模块边界

- `aieval_runner/runner/`：配置、CLI 和 schema；配置层拒绝其他 dataset。
- `aieval_runner/datasets/`：real_cases 严格验证、解析和远端/本地加载。
- `aieval_runner/agent/app_server/`：进程、thread/turn、事件落盘与实时 Langfuse trace。
- `aieval_runner/evaluation/events.py`：安全读取完整事件、收集全部完成 MCP 调用，并保留统一生命周期 status。
- `aieval_runner/evaluation/evaluators/tool_contract.py`：两项独立 BOOLEAN evaluator。
- `aieval_runner/evaluation/flow/`：local/hosted 共用评分、DatasetRunItem 和结果摘要。

## 事件与状态

事件文件限制为 64MB、单行 16MB、最多 10,000 个事件。工具标识和生命周期 status 采用有界字符串保留，不解析业务 result。

所有完成 MCP 调用都会进入摘要，不设指标平台、Metabase 或 SQL 白名单。工具出现与工具成功分开评分：失败调用仍能证明“工具被调用”，但不能通过状态评分。

## Trace ownership

runner 内 live emitter 持有 app-server eval trace，层级为 `Agent -> Generation -> Tool`。全局 Stop hook 可以运行，但不是该 trace owner。sidecar 继续记录 run、case、dataset item、thread、turn、event 和 mapping 的关联。

两个确定性 Score 都写 experiment root observation；若 root observation 缺失则退化为 trace-level score，并在基础 metadata 中保留 mapping 诊断。

## 验证

```powershell
python scripts/check_real_cases_evaluation.py
python external_eval_runner.py --print-schemas
python -m compileall external_eval_runner.py aieval_runner scripts/check_real_cases_evaluation.py
```
