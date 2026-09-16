# AI Eval External Runner

## Session 数据回收一键入口

运行 `python 一键启动数据回收.py`，可按时间范围（或全部 Session）从 Langfuse 读取会话、
调用大模型做第一遍 Case 清洗，并在明确确认后写入人工标注队列。

`run_external_eval.bat` 是旧的 Dataset Experiment Runner，不是数据回收入口。它要求
Langfuse 当前项目已存在 `.env` 中 `LANGFUSE_DATASET_NAME` 指定的 Dataset。

首次使用、配置字段、人工复核和结果导出说明见
[`docs/session_recovery_user_guide.md`](docs/session_recovery_user_guide.md)。

本仓库是面向 Langfuse Dataset `aieval/real_cases` 的轻量评测脚本。入口为 `external_eval_runner.py`，Agent 固定通过 `codex app-server --stdio` 执行。

## 评测边界

Runner 只校验每次运行的工具契约，不校验最终回答文本、SQL、指标数值或表格 Gold。固定写入两个 BOOLEAN Score：

- `expected_tools_match`：预期工具是否按 `tool_match_mode=any|all` 完整出现；
- `tool_result_status`：命中工具的统一生命周期 `status` 是否正常完成。

评分读取 `.aieval/runs/<run_id>/app-server-events/*.jsonl` 的完整事件，不依赖可能被截断的 Langfuse Tool Observation preview。

## 安装

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

`.env` 只填写本地真实密钥，不要提交：

```dotenv
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_DATASET_NAME=aieval/real_cases
```

默认 Agent 工作目录位于系统临时目录 `aieval-agent-workspace`，不包含仓库题库和评测契约。仅在明确调试非评测任务时传入 `--codex-cwd`。

评测回答规范的唯一可编辑模板是 `config/eval-agent/AGENTS.override.md`。Runner 启动前会把它原样同步到本次 run 的隔离工作目录，并在 trace 中记录策略名与模板 SHA-256。不要直接编辑临时目录中的副本。

当机器上同时安装多个 Codex 版本时，使用 `AIEVAL_CODEX_BIN` 固定评测所用的可执行文件，避免 PATH 命中旧版：

```dotenv
AIEVAL_CODEX_BIN=C:\path\to\codex.exe
```

## DatasetItem 格式

```json
{
  "id": "09587e4a-70de-474c-86a0-032829a291ac",
  "input": {
    "input": "每天到货交接的标准入库单类型的集装箱柜数，计算口径是什么？"
  },
  "expectedOutput": {
    "expected_result": "说明该指标的完整业务口径。",
    "expected_values": {}
  },
  "metadata": {
    "case_id": "L1-METRIC-EXPLAIN-001",
    "expected_tools": [
      "mcp__metric_mcp_remote__searchBizMetric",
      "mcp__metric_mcp_remote__searchDerivedMetric"
    ],
    "tool_match_mode": "any"
  }
}
```

必填字段：

- `input.input`：原样发送给 Agent 的真实用户问题；
- `metadata.case_id`：评测用例标识；
- `expectedOutput.expected_result`：整体结果要求；
- `expectedOutput.expected_values`：字段、数值和取数口径要求，没有精确值时填空对象；
- `metadata.expected_tools`：非空工具名数组；
- `metadata.tool_match_mode`：`any` 或 `all`；
- `tool_result_status` Score 固定要求匹配调用的 `mcpToolCall.status` 为 `completed`，不读取 Dataset output 中的状态字段。

工具名匹配允许 MCP 命名空间和 `_`、`-`、`.`、大小写差异，但工具末段必须完整相等。例如 `mcp__metabase__construct_query` 可匹配事件中的 `metabase.construct_query`，不会匹配 `construct_query_detail`。

状态评分只读取 app-server 统一生成的 `mcpToolCall.status`。`completed` 表示调用正常完成；`failed`、超时、中断或缺失均失败。各 MCP 业务 `result` 中的同名字段不参与评分。

## 运行

运行整个线上 Dataset：

```powershell
python external_eval_runner.py --dataset aieval/real_cases --codex-sandbox danger-full-access
```

限制数量或指定 item：

```powershell
python external_eval_runner.py --limit 1
python external_eval_runner.py --item-id 09587e4a-70de-474c-86a0-032829a291ac
```

本地定向调试仍使用同一 real_cases 结构：

```powershell
python external_eval_runner.py --items-json .aieval/target-real-cases.json
```

Runner 会拒绝 `aieval/real_cases` 以外的 dataset。

## 验证

```powershell
python scripts/check_real_cases_evaluation.py
python external_eval_runner.py --print-schemas
python -m compileall external_eval_runner.py aieval_runner scripts/check_real_cases_evaluation.py
```

回归脚本覆盖 real_cases parser、any/all 工具匹配、完整 MCP 事件、生命周期 completed/failed、未知或缺失状态、重复调用和两个固定 Score。

## 上传辅助工具

`scripts/upload_dataitems.py` 仍可上传完整 DatasetItem JSON。目标必须是 `aieval/real_cases`，输入会经过同一严格 parser 校验。默认 dry-run，只有显式传入 `--commit` 才写入 Langfuse。

```powershell
python scripts/upload_dataitems.py --dataset aieval/real_cases --file cases.json --allow-update
python scripts/upload_dataitems.py --dataset aieval/real_cases --file cases.json --allow-update --commit
```
