# App-server 实时 Langfuse Generation Trace

## 元数据

- 日期：2026-06-19
- 类型：pattern
- 标签：app-server, langfuse, agent, generation, tool, token-usage, live-trace

## 背景

外部评测通过 `codex app-server --stdio` 执行真实 thread/turn。旧方案在 turn 结束后由 Stop hook worker 根据 sidecar 和 event JSONL 补写 trace，只能稳定得到单层 Agent/Tool，无法还原每次模型调用的 Generation、真实 token usage 和 observation 生命周期。

## 根因

事后补写拿到的是完成后的事件快照，而不是 Langfuse observation 的实时上下文：

- 无法在模型调用开始时建立 Generation。
- Tool 只能挂到事后创建的 root，丢失“哪次模型调用发起工具”的关系。
- 累计 token usage 容易被当成每次调用 usage，造成重复计费。
- 事后创建并立即结束的 root 常表现为 0 token、约 1ms latency。
- Stop hook 可能与 runner 同时创建同一 app-server turn 的 trace，产生重复所有权。

## 正确模式

runner 在 `turn/start` 返回后创建 `AppServerLiveTrace`，注册 thread/turn listener，并按事件顺序实时维护：

```text
Codex Turn Agent
└── 每次模型调用 Generation
    └── 该次调用发起的 MCP / command / dynamic Tool
```

sidecar 保留本地关联信息，并显式写入：

```json
{"trace_owner": "aieval_app_server_live_event"}
```

全局 Stop hook 仍可运行，但不再拥有 app-server eval trace。

## 事件到 observation 的映射

| app-server 事件 | Langfuse 动作 |
|---|---|
| `turn/started` | 更新 Agent 的 turn 起始元数据 |
| reasoning / assistant `item/started`、`item/completed` | 创建或补充当前 Generation 的 input/output |
| Tool `item/started` | 在当前 Generation 下创建 Tool |
| Tool `item/completed` | 更新 Tool output/status/duration 并结束 |
| `thread/tokenUsage/updated` | 用 `tokenUsage.last` 更新并结束当前 Generation |
| `turn/completed` / `failed` / `cancelled` | 更新 Agent 状态和 `turn.durationMs` |
| listener 注销后 `finish()` | 幂等结束残留 observation、flush、写 mapping |

缺失 Tool started 时允许补建 synthetic Tool，并在 metadata 标记 `codex.synthetic_start=true`；其父节点仍必须是当前 Generation。

## Usage 去重

`thread/tokenUsage/updated.tokenUsage.last` 表示本次模型调用，写入对应 Generation 的 `usage_details`。`tokenUsage.total` 是 turn 累计值，只写入 mapping 和 Agent metadata 作为诊断信息。

不得把 `tokenUsage.total` 上传到每个 Generation，否则多次更新会重复累计 token 和费用。重复或倒退的累计 total 也应忽略，防止重放事件污染诊断数据。

## 异常与幂等边界

- listener 异常不能破坏 app-server stdout reader，但必须在注销或 `finish()` 时传播为 tracing 基础设施失败。
- turn 事件和本地 event JSONL 优先保留；tracing 失败后仍等待 turn 完成并执行可安全的收尾。
- `finish()` 可重复调用；mapping 以 `turn_key + trace_id + root_observation_id` 去重。
- observation update/end、Langfuse flush、mapping 写入和 runtime shutdown 的错误分别收集，不能用清理异常覆盖主异常。
- 失败时不回退创建 0-token、约 1ms 的 direct-event 伪 trace。

## 验证命令

```powershell
.venv\Scripts\python.exe scripts\check_app_server_live_trace.py
.venv\Scripts\python.exe scripts\check_app_server_tool_mapping.py
.venv\Scripts\python.exe scripts\check_deterministic_evaluation.py
.venv\Scripts\python.exe scripts\check_app_server_hook_mapping.py --turn-timeout-seconds 300
```

真实 canary 还应通过 Langfuse Public API 回读 `/traces/{trace_id}`，确认 Agent latency 大于 1 秒、Generation usage 大于 0，以及 Tool 的 `parentObservationId` 属于 mapping 中的 Generation IDs。

2026-06-22 的真实 Tool canary 已验证该链路：run `app-server-live-trace-1782090128`、trace `80282d7b8cecff75e75f21478a9e0daf` 包含 2 个 Generation，token 总计 53230（26592 + 26638）；1 个 Tool 的 remote/mapping parent 精确一致。`turnDurationMs=32742`，Agent latency 为 31.207 秒，`totalCost=0`。

自定义模型没有 Langfuse 价格映射时，token 仍可见而 `totalCost` 可能为 `0`，这不是 usage 丢失。

`_clip` 只负责长度裁剪与大整数转换，不会自动识别或脱敏秘密。输入数据和 Tool 参数不得包含密钥，凭据必须留在环境变量或 MCP 配置中。

## 适用场景

适用于需要把流式 Agent 事件还原为 Langfuse Agent/Generation/Tool 层级，并要求准确 token、时长、工具父子关系、异常传播和重复事件幂等的 app-server 评测或类似外部 runner。
