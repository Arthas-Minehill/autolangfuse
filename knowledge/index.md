# Knowledge Index

## Patterns

- [App Server MCP Timeout Diagnosis](patterns/app-server-mcp-timeout-diagnosis.md) - 区分 app-server timeout、MCP stream 抖动和指标平台主链路失败。
- [Windows Stop Hook 标准输入必须显式使用 UTF-8](patterns/windows-hook-stdin-utf8.md) - 避免 app-server direct-event 中文被 Windows 代码页误解码。
- [App-server MCP 事件适配为 Langfuse Tool Observation](patterns/app-server-tool-observation-upload.md) - 历史 direct-event 方案，已被 live emitter 取代。
- [App-server 实时 Langfuse Generation Trace](patterns/app-server-live-langfuse-generation-trace.md) - runner 实时维护 Agent、Generation、Tool、usage 与时长。
- [Langfuse Experiment Runner Managed Evaluators](patterns/langfuse-experiment-runner-managed-evaluators.md) - experiment judge 必须走 SDK root span 并复用 trace context。

- [Langfuse 低分项契约排查模式](patterns/langfuse-low-score-contract-triage.md) - 区分 judge 表达问题、远端 contract 漏同步和指标别名问题。

## Anti-Patterns

## Decisions

- [External runner thin entry modularization](decisions/external-runner-thin-entry-modularization.md) - 外部 runner 固定走 app-server；runner live emitter 拥有 Langfuse trace。
- [Real Cases Tool Contract Evaluation](decisions/real-cases-tool-contract-evaluation.md) - real_cases 仅以完整事件校验工具匹配和显式返回状态。

## Subsystem Specs
- [Evaluation Score Logic](../docs/evaluation_score_logic.md) - real_cases 两项工具契约评分的来源、聚合与诊断规则。
