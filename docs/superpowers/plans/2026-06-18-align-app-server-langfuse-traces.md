# App-server Langfuse Trace 对齐实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `codex app-server` 评测 trace 从事后单层 Agent/Tool 补写，升级为实时 `Agent -> Generation -> Tool`，正确上传模型调用 token 和 turn 时长。

**Architecture:** `AppServerClient` 为每个 thread/turn 提供有序异步事件订阅；新的 `AppServerLiveTrace` 状态机实时维护 Langfuse observation 生命周期，并在 turn 结束后写回兼容 mapping。backend 负责注册/关闭订阅并读取 mapping，旧 direct-event 函数保留为兼容代码但不再作为正常 fallback。

**Tech Stack:** Python 3.12、Codex app-server JSON-RPC、Langfuse Python SDK 4.x、PowerShell/脚本式确定性检查。

---

## 文件结构

- Create: `aieval_runner/app_server_live_trace.py` — app-server 事件到 Langfuse Agent/Generation/Tool 的实时状态机。
- Modify: `aieval_runner/app_server_protocol.py` — UTC 接收时间、有序 turn listener 队列和错误回收。
- Modify: `aieval_runner/codex_app_server_backend.py` — 创建 live tracer、注册 listener、关闭并读取 mapping。
- Modify: `aieval_runner/app_server_trace_mapping.py` — 复用 clipping、mapping 写入和工具字段归一化帮助函数。
- Create: `scripts/check_app_server_live_trace.py` — 不访问网络的完整状态机 fixture。
- Modify: `scripts/check_app_server_tool_mapping.py` — 移除对旧全局 Python hook 的依赖，改为验证本项目事件持久化与 listener。
- Modify: `scripts/check_deterministic_evaluation.py` — 断言新 mapping 字段和 observation 父子关系。
- Modify: `README.md`、`docs/external_eval_runner_design.md` — 记录 token、时长、模型定价和降级语义。

### Task 1: 有序 app-server turn 事件订阅

**Files:**
- Modify: `aieval_runner/app_server_protocol.py`
- Test: `scripts/check_app_server_tool_mapping.py`

- [x] **Step 1: 写失败检查，要求事件带接收时间且 listener 按顺序收到历史和实时事件**

在 `scripts/check_app_server_tool_mapping.py` 中直接构造 `AppServerClient` fake stdout，注册 listener 后断言：

```python
received: list[dict[str, Any]] = []
client.register_turn_event_listener(
    thread_id=thread_id,
    turn_id=turn_id,
    callback=received.append,
)
client._read_stdout()
errors = client.unregister_turn_event_listener(thread_id=thread_id, turn_id=turn_id)
assert errors == []
assert [row["method"] for row in received] == [
    "item/completed",
    "item/completed",
    "item/completed",
    "hook/started",
]
assert all(isinstance(row.get("_received_at"), str) for row in received)
```

- [x] **Step 2: 运行检查并确认失败**

Run: `.venv\Scripts\python.exe scripts\check_app_server_tool_mapping.py`

Expected: FAIL，原因是 `register_turn_event_listener` 尚不存在，且当前脚本仍依赖旧 hook。

- [x] **Step 3: 实现订阅对象和 UTC 接收时间**

在 `aieval_runner/app_server_protocol.py` 增加：

```python
TurnEventCallback = Callable[[JsonObject], None]
_TURN_EVENT_STOP = object()


class _TurnEventSubscription:
    def __init__(self, callback: TurnEventCallback) -> None:
        self.callback = callback
        self.queue: "queue.Queue[object]" = queue.Queue()
        self.errors: List[BaseException] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def put(self, message: JsonObject) -> None:
        self.queue.put(message)

    def close(self) -> List[BaseException]:
        self.queue.put(_TURN_EVENT_STOP)
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            self.errors.append(TimeoutError("turn event listener did not stop"))
        return list(self.errors)

    def _run(self) -> None:
        while True:
            message = self.queue.get()
            if message is _TURN_EVENT_STOP:
                return
            try:
                self.callback(dict(message))
            except BaseException as exc:
                self.errors.append(exc)
```

在 client 中增加 `_turn_event_listeners`，注册时在 `_message_lock` 内先排入历史消息再安装订阅，后续 `_read_stdout` 只做 `subscription.put(message)`。解析通知后写入：

```python
message.setdefault("_received_at", datetime.now(timezone.utc).isoformat())
```

`unregister_turn_event_listener()` 返回 listener errors；`close()` 关闭所有订阅。

- [x] **Step 4: 运行检查并确认通过**

Run: `.venv\Scripts\python.exe scripts\check_app_server_tool_mapping.py`

Expected: PASS，输出包含 `ordered_listener=true` 和 `received_at=true`。

### Task 2: 实时 Agent/Generation/Tool 状态机

**Files:**
- Create: `aieval_runner/app_server_live_trace.py`
- Modify: `aieval_runner/app_server_trace_mapping.py`
- Create: `scripts/check_app_server_live_trace.py`

- [x] **Step 1: 写一个完整失败 fixture**

`scripts/check_app_server_live_trace.py` 定义 `FakeLangfuse`/`FakeObservation`，事件顺序至少包含：

```python
events = [
    turn_started(started_at=1_781_679_704),
    item_completed("reasoning", summary=["先查指标"]),
    item_completed("agentMessage", phase="commentary", text="正在查询"),
    item_started("commandExecution", id="cmd-1", command="echo ok"),
    item_completed("commandExecution", id="cmd-1", output="ok", duration_ms=20),
    token_usage(last={"inputTokens": 10, "cachedInputTokens": 4, "outputTokens": 2,
                      "reasoningOutputTokens": 1, "totalTokens": 12},
                total={"inputTokens": 10, "cachedInputTokens": 4, "outputTokens": 2,
                       "reasoningOutputTokens": 1, "totalTokens": 12}),
    item_started("mcpToolCall", id="mcp-1", server="metric-mcp-remote",
                 tool="searchMetricApp", arguments={"name": "客户"}),
    item_completed("mcpToolCall", id="mcp-1", result={"data": []}, duration_ms=30),
    item_completed("agentMessage", phase="final_answer", text="查询完成"),
    token_usage(last={"inputTokens": 20, "cachedInputTokens": 8, "outputTokens": 3,
                      "reasoningOutputTokens": 1, "totalTokens": 23},
                total={"inputTokens": 30, "cachedInputTokens": 12, "outputTokens": 5,
                       "reasoningOutputTokens": 2, "totalTokens": 35}),
    turn_completed(started_at=1_781_679_704, completed_at=1_781_679_809, duration_ms=105_433),
]
```

断言根节点为 Agent、两个 Generation、Tool 父节点为 Generation、usage 映射完整、最终 mapping 包含 generation/tool ids、累计 usage 和 `turn_duration_ms`。

- [x] **Step 2: 运行 fixture 并确认失败**

Run: `.venv\Scripts\python.exe scripts\check_app_server_live_trace.py`

Expected: FAIL，原因是 `AppServerLiveTrace` 尚不存在。

- [x] **Step 3: 实现 `AppServerLiveTrace` 公共接口**

新模块导出：

```python
class AppServerLiveTrace:
    def __init__(
        self,
        *,
        case: EvalCase,
        ctx: EvalRunContext,
        config: RunnerConfig,
        mapping_path: Path,
        thread_id: str,
        turn_id: str,
        session_id: str,
        user_prompt: str,
        event_path: Path,
        model: str,
    ) -> None: ...

    def handle_event(self, message: Dict[str, Any]) -> None: ...

    def finish(self) -> Dict[str, Any]: ...
```

构造时通过 `get_langfuse_sdk(config)` 和 `create_trace_id()` 创建根 Agent；`handle_event()` 按 method 分派；`finish()` 幂等结束残留 observation、flush 并写 mapping。

- [x] **Step 4: 实现 Generation 切分和 usage 映射**

核心映射函数：

```python
def _usage_details(raw: Any) -> Optional[Dict[str, int]]:
    if not isinstance(raw, dict):
        return None
    fields = {
        "inputTokens": "input",
        "cachedInputTokens": "cache_read_input_tokens",
        "outputTokens": "output",
        "reasoningOutputTokens": "reasoning_tokens",
        "totalTokens": "total",
    }
    details = {
        target: int(raw[source])
        for source, target in fields.items()
        if isinstance(raw.get(source), int) and not isinstance(raw.get(source), bool)
    }
    return details or None
```

收到 `thread/tokenUsage/updated` 时 update/end active Generation；`tokenUsage.total` 仅保存到 mapping/根 metadata。

- [x] **Step 5: 实现 Tool 生命周期**

支持 `mcpToolCall`、`commandExecution`、`dynamicToolCall`。started 时创建 Tool；completed 时 update output/error/status/duration 后 end。缺失 started 时补建，并写：

```python
metadata["codex.synthetic_start"] = True
```

所有 Tool 都通过 `active_generation.start_observation(..., as_type="tool")` 创建。

- [x] **Step 6: 运行 fixture 并确认通过**

Run: `.venv\Scripts\python.exe scripts\check_app_server_live_trace.py`

Expected: PASS，输出包含 `generation_count=2`、`tool_count=2`、`total_tokens=35`、`turn_duration_ms=105433`。

### Task 3: backend 接入与 mapping 兼容

**Files:**
- Modify: `aieval_runner/codex_app_server_backend.py`
- Modify: `scripts/check_deterministic_evaluation.py`
- Test: `scripts/check_app_server_live_trace.py`

- [x] **Step 1: 写失败断言验证 backend 不再调用 direct-event fallback**

在确定性检查中 monkeypatch `AppServerLiveTrace`，断言调用顺序是创建 tracer、注册 listener、等待 turn、注销 listener、finish，并断言：

```python
assert execution.output["mapping_source"] == "codex_app_server_live_event"
assert execution.mapping["generation_observation_ids"] == ["generation-1"]
assert execution.mapping["token_usage_total"]["totalTokens"] == 35
```

- [x] **Step 2: 运行确定性检查并确认失败**

Run: `.venv\Scripts\python.exe scripts\check_deterministic_evaluation.py`

Expected: FAIL，backend 仍调用 `create_direct_event_mapping()`。

- [x] **Step 3: 接入 live tracer**

`run_codex_app_server()` 在取得 turn id 后：

```python
live_trace = AppServerLiveTrace(
    case=case,
    ctx=ctx,
    config=config,
    mapping_path=target_mapping_path,
    thread_id=thread_id,
    turn_id=turn_id,
    session_id=session_id,
    user_prompt=user_prompt,
    event_path=event_path,
    model=config.codex_model,
)
runtime.client.register_turn_event_listener(
    thread_id=thread_id,
    turn_id=turn_id,
    callback=live_trace.handle_event,
)
```

finally 中先关闭 event log，再注销 listener并收集 errors。正常情况下调用 `live_trace.finish()` 获取 mapping；listener/tracer 失败时抛出明确基础设施错误，不再创建伪 trace。

- [x] **Step 4: 保持 mapping 和 observation 选择兼容**

保留 `trace_id`、`root_observation_id`、`tool_observation_ids` 和 `select_target_observation_id()` 行为；新增字段只做扩展。sidecar 增加：

```python
"trace_owner": "aieval_app_server_live_event"
```

- [x] **Step 5: 运行确定性检查和 schema 检查**

Run:

```powershell
.venv\Scripts\python.exe scripts\check_deterministic_evaluation.py
.venv\Scripts\python.exe external_eval_runner.py --print-schemas
```

Expected: 两条命令均退出 0。

### Task 4: canary、文档和完整验证

**Files:**
- Modify: `scripts/check_app_server_tool_mapping.py`
- Modify: `README.md`
- Modify: `docs/external_eval_runner_design.md`
- Modify: `openspec/changes/align-app-server-langfuse-traces/tasks.md`

- [x] **Step 1: 更新 canary 为项目内实现**

移除 `codex_langfuse_worker.py`/`codex_langfuse_stop_hook.py` 动态加载。canary 只验证：

- UTF-8 事件增量落盘
- `_received_at`
- listener 顺序与错误隔离
- 中文、大整数、失败工具字段保真

- [x] **Step 2: 更新文档**

README 和设计文档明确：

```text
app-server trace 由 runner 内实时 emitter 创建：
Codex Turn Agent -> 每次模型调用 Generation -> 对应 Tool。
Generation usage 来自 thread/tokenUsage/updated.tokenUsage.last。
turn duration 来自真实 observation 生命周期，并用 Codex durationMs 校验。
若模型未配置 Langfuse 定价，token 仍可见但 totalCost 可能为 0。
```

- [x] **Step 3: 运行本地完整检查**

Run:

```powershell
.venv\Scripts\python.exe scripts\check_app_server_live_trace.py
.venv\Scripts\python.exe scripts\check_app_server_tool_mapping.py
.venv\Scripts\python.exe scripts\check_deterministic_evaluation.py
.venv\Scripts\python.exe external_eval_runner.py --print-schemas
```

Expected: 全部退出 0。

- [x] **Step 4: 运行真实 app-server Tool canary**

Run:

```powershell
.venv\Scripts\python.exe scripts\check_app_server_hook_mapping.py `
  --prompt "Use the shell command tool to run Write-Output app-server-live-tool-canary, then reply exactly: app-server live tool canary ok" `
  --expected-tool-count 1 `
  --turn-timeout-seconds 300
```

Expected: 创建非空 mapping；远端 Langfuse trace 至少包含一个 Agent、一个 Generation 和一个 Tool，Generation usage total 大于 0，Agent latency 大于 1 秒；全部 mapping Tool 都能在远端找到，且远端父节点精确等于 mapping 的 `parent_observation_id` 并属于 Generation IDs。

2026-06-22 已成功验证：run `app-server-live-trace-1782090128`，trace `80282d7b8cecff75e75f21478a9e0daf`。远端包含 2 个 Generation，token 分别为 26592 和 26638，总计 53230；包含 1 个 Tool，remote 与 mapping 的 parent observation id 精确一致并指向对应 Generation。`turnDurationMs=32742`，Agent latency 为 31.207 秒；自定义模型未配置价格映射，`totalCost=0`。

- [x] **Step 5: 更新 OpenSpec task 状态**

完成并验证的条目由 `- [ ]` 改为 `- [x]`；无法执行的真实远端验证必须保留未勾选并写明原因，禁止伪造完成。

---

## 自检

- Spec coverage：Agent 生命周期、Generation token、输入输出、Tool 父子关系、mapping 扩展、listener 错误隔离和降级均有对应任务。
- Placeholder scan：计划不含 TBD/TODO/“类似上一步”等占位描述。
- Type consistency：统一使用 `AppServerLiveTrace.handle_event()`、`finish()`、`register_turn_event_listener()` 和 `unregister_turn_event_listener()`。
