# External runner thin entry modularization

## Metadata

- Date: 2026-06-12
- Updated: 2026-06-19
- Type: decision
- Tags: external-runner, langfuse, cli, modularization, codex-app-server

## Context

`external_eval_runner.py` 需要保持轻量脚本入口，同时评测执行要尽量接近真实 Codex Desktop 会话。验证表明 app-server 可以加载当前 Codex 配置中的 MCP、skills 和 plugins，并能触发 Stop hook；因此 runner 的 Agent 执行路径固定为 `codex app-server --stdio`。

## Decision

保留 `external_eval_runner.py` 作为薄入口，只导入 `aieval_runner.cli.main`。

Agent 执行固定走 app-server：

- `agent.py` 只调用 `codex_app_server_backend.run_codex_app_server()`。
- `app_server_protocol.py` 封装 JSON-RPC、进程生命周期、请求超时和事件读取。
- `codex_app_server_backend.py` 管理 eval run 级 app-server 进程，每个 case 创建独立 ephemeral thread。
- 不再保留本项目内的命令式 exec 后端、后端选择参数、exec timeout 或 exec output 文件逻辑。

## Mapping 与 trace ownership

runner 使用 sidecar 记录 app-server turn 与本地产物的关联：

```text
.aieval/runs/<run_id>/app-server-eval-context.jsonl
```

sidecar 包含 `trace_owner=aieval_app_server_live_event`。app-server eval trace 由 runner 内 `AppServerLiveTrace` 实时创建，层级为 `Agent -> Generation -> Tool`；全局 Stop hook 仍可运行，但不是该 trace 的 owner。

Generation usage 来自每次 `thread/tokenUsage/updated.tokenUsage.last`，累计 `tokenUsage.total` 只用于 mapping 和 Agent metadata 诊断。event JSONL 继续作为确定性评分和故障诊断的数据源。

## Preserved Behavior

- Codex prompt 仍只包含 `case.original_request`。
- `expectedOutput`、`eval_run_id`、`case_id` 不进入用户 prompt。
- DatasetRunItem 和 Score 写回仍在 `evaluation.py` 与 `langfuse_adapter.py`。
- 运行产物仍写在 `.aieval/runs/`，不入库。

## Validation

已验证：

- `python external_eval_runner.py --print-schemas`
- `python scripts/check_app_server_mcp.py --server metric-mcp-remote --tool metricMcpInfo --agent-turn`
- `python scripts/check_app_server_extensions.py`
- `python scripts/check_app_server_hook_mapping.py`

未完全验证：

- 真实 Dataset 单 case `L1-METRIC-CUSTOMER-ACTIVE-011` 曾使用 app-server 执行 600 秒未收到 `turn/completed`。提升到 1800 秒后，`app-server-runner-canary-long-1781523110` 成功完成，真实调用了 `searchMetricApp`、`searchBizMetric`、多次 `searchMetricAppQueryResult` 和 `searchSqlByMetricTypeAndNameExact`，并写回 DatasetRunItem/Score。当前结论是：timeout 不是指标平台 MCP 主链路稳定不可用，更像偶发 app-server/model turn 未收尾或非关键 MCP stream 抖动。默认 turn timeout 已提升到 1800 秒。

## Consequences

- runner 行为更接近 Codex Desktop thread/turn 会话。
- 不再维护两套 Agent 执行逻辑。
- app-server 后端对真实业务 case 的耗时和 MCP 远端稳定性更敏感，需要通过 canary 和 per-case event JSONL 定位具体瓶颈。
- tracing 失败时保留本地事件并报告基础设施失败，不再降级创建 0-token、近零时长的 direct-event 伪 trace。
