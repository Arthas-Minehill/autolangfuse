# App-server MCP 事件适配为 Langfuse Tool Observation

## 元数据

- 日期：2026-06-15
- 更新：2026-06-19
- 类型：pattern
- 标签：app-server, mcp, langfuse, tool-observation, utf-8

> 状态：其中“由 Stop hook worker 事后创建 app-server eval trace”的方案已被 runner 内 `AppServerLiveTrace` 取代。本文仅保留 event JSONL 的 UTF-8、增量落盘和工具字段保真经验；当前 Tool 必须实时创建在 Generation 下。

## 背景

Codex Desktop 的 Langfuse worker 可以从 rollout transcript 中读取 `function_call` 和 `function_call_output`。外部评测的 app-server Stop payload 没有 transcript path，只提供最终 assistant message，因此旧 direct-event 路径无法上传 MCP 工具调用。

## 关键时序

app-server 的事件顺序是：

```text
mcpToolCall item/completed
agentMessage item/completed
hook/started
hook/completed
turn/completed
```

如果等 `turn/completed` 后才写 event JSONL，Stop hook 启动时 worker 看不到工具事件。正确做法是在 `turn/start` 返回后注册增量日志，并在每个 MCP 完成事件和最终 assistant message 后 flush。

## 历史适配方法

1. sidecar 显式记录 `event_path`、`thread_id`、`turn_id` 和 `trace_owner`。
2. runner 用严格 UTF-8 记录事件，并按 thread/turn 过滤。
3. 按 Tool call id 合并 started/completed，实时维护 Tool 生命周期。
4. 工具名称使用 `<server>.<tool>`，arguments 作为 input，result 作为 output，error/status/duration 写入 metadata。
5. mapping 记录实际 Tool Observation ID，且父 observation 必须是对应 Generation。

## 数据保真

- hook stdin 使用 `utf-8-sig`。
- event JSONL 使用 UTF-8 和 `ensure_ascii=False`。
- arguments/result/error 只做长度裁剪，不自动识别或脱敏秘密；输入与工具参数不得包含密钥，凭据应留在环境变量或 MCP 配置中。
- 绝对值超过 `9_007_199_254_740_991` 的整数转为十进制字符串，避免 JavaScript JSON 精度丢失。

## 验证

本地 fixture 应覆盖中文参数、中文结果、中文错误、started/completed 去重、session 与 thread ID 不同以及增量落盘。端到端验证通过真实 app-server turn 和 Langfuse Public API 回读 trace，比较远端 Tool Observation 的父子关系、input 和 output。
