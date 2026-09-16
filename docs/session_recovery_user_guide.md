# Langfuse Session 数据回收工具使用手册

## 1. 工具目标与流程

工具把指定时间范围内的 Langfuse Session 转成“等待人工复核的数据集候选”。

```text
Langfuse Observation
        ↓ 按 sessionId / traceId 重建
完整 Session JSON
        ↓ 大模型范围过滤、合并和拆分
Case
        ↓ 映射回来源 Trace / Observation
Dataset Recovery Candidate
        ↓ AI 预标注
Langfuse Annotation Queue（Pending）
        ↓ 人工逐条核对、修改
Completed 审核结果
```

AI 只做第一遍预审，不代替人工确认；历史回答声称“查询成功”也不代表结果已经成为
Gold。工具不会自动把审核结果发布成正式 Dataset。无 `sessionId` 的 Observation 无法重建
完整 Session，流程会统计并跳过。

## 2. 文件说明

| 文件 | 用途 |
|---|---|
| `一键启动数据回收.py` | 数据回收图形界面主入口 |
| `config/session_recovery.env` | 本机账号、Key、模型和默认参数；不会提交版本库 |
| `config/session_recovery.env.example` | 配置模板，不包含真实密钥 |
| `config/ai_review_prompt.md` | AI 第一遍 Session 清洗与 Case 拆分规则 |
| `scripts/session_recovery_launcher.py` | 一键工作流和图形界面 |
| `scripts/session_annotation_pipeline.py` | Candidate 创建、预标注、入队与结果导出 |

## 3. 首次使用

需要 Windows 10/11、Python 3.10 或更高版本，并能访问 Langfuse 和大模型 API。

在项目目录打开 PowerShell，安装依赖：

```powershell
python -m pip install -r requirements.txt
```

推荐在项目目录运行：

```powershell
python .\一键启动数据回收.py
```

如果 Windows 已将 `.py` 关联到 Python，也可以双击该文件。当前电脑没有 `.py` 文件关联时，
双击不会可靠启动，应使用上面的命令。`run_external_eval.bat` 属于旧评测流程，不是本工具入口。

第一次启动会自动创建 `config/session_recovery.env`。如果原来的 `.env` 已有 Langfuse
Host、Public Key 和 Secret Key，程序会迁移它们，但不会在日志中显示 Key。

## 4. 配置说明

可在界面填写后点击“保存配置”，也可用文本编辑器修改 `config/session_recovery.env`。

### 4.1 Langfuse

```dotenv
LANGFUSE_HOST=https://langfuse.lecangs.com
LANGFUSE_PROJECT_ID=<目标项目 ID>
LANGFUSE_PROJECT_NAME=<便于识别的项目名>
LANGFUSE_PUBLIC_KEY=<Public Key>
LANGFUSE_SECRET_KEY=<Secret Key>
LANGFUSE_QUEUE_NAME=数据回收-准入审核-POC
LANGFUSE_QUEUE_ID=<Annotation Queue ID>
```

- API 调用只使用 Host、Project ID、Public Key 和 Secret Key。
- 每条返回数据还会验证 `projectId`；缺失或跨项目时立即停止。
- Queue 需要预先配置好人工审核所需的 Score Config。

网页账号密码：

```dotenv
LANGFUSE_LOGIN_EMAIL=
LANGFUSE_LOGIN_PASSWORD=
```

它们只供同事手工登录网页时查阅。程序不会模拟网页登录，也不会把网页密码当作 API Key。

### 4.2 大模型 API

工具使用 OpenAI-compatible Chat Completions 接口：

```dotenv
AI_API_BASE_URL=https://api.openai.com/v1
AI_API_KEY=<模型 API Key>
AI_MODEL=<模型名称>
AI_TIMEOUT_SECONDS=180
AI_MAX_SESSION_CHARS=60000
AI_JSON_MODE=true
```

- API 地址通常填到 `/v1`，程序自动追加 `/chat/completions`；完整接口地址也可直接填写。
- `AI_JSON_MODE=true` 要求 JSON object；兼容接口不支持 `response_format` 时改为 `false`。
- 字符上限只限制送给模型的压缩内容，落盘的原始 Session JSON 不截断。
- 真实配置不得发送给他人；共享时只发送 `.env.example`。

配置后点击“测试连接”。正常日志包括：

```text
Langfuse API：连接成功
大模型 API：连接及 JSON 输出测试成功
```

## 5. 运行步骤

### 5.1 选择 Session

时间格式固定为 `YYYY-MM-DD HH:MM:SS`，默认按东八区解释，再转换为 UTC 查询。

- 普通批次：填写开始和结束时间。
- 全量批次：勾选“观察全部 Session（忽略时间范围）”。
- `Session 上限`：`0` 表示不限；第一次建议填 `1～10`。

“全部 Session”只包含带 `sessionId` 的 Observation。没有 `sessionId` 的数据不会伪造
Session，而是统计后跳过。

### 5.2 先做预览

先不要勾选“实际写入 Langfuse”，点击“开始运行”。预览将：

1. 从 Langfuse 只读导出 Observation。
2. 按 Session 和 Trace 重建本地 JSON。
3. 逐个调用大模型做范围过滤、Case 合并和拆分。
4. 将 Case 映射回相关 Trace 和根 Observation。
5. 生成候选文件，但不创建远端 Candidate，也不修改 Annotation Queue。

检查日志中的 Session、Case 和低置信度映射数量后，再进行正式写入。

### 5.3 正式写入

勾选“实际写入 Langfuse（创建 Candidate、AI 预标注并加入待审队列）”，点击“开始运行”
并确认弹窗。正式流程会：

1. 为每个 Case 创建唯一的审阅 Trace 和 `Dataset Recovery Candidate` Observation。
2. 把完整 Session 的用户/助手轮次放进 Candidate Input。
3. 写入 AI 预标注：人工确认是否数据类、数据集准入状态、优化后问题。
4. 把 Candidate 加入 Annotation Queue，状态保持 `Pending`。

每次运行都有独立 `recovery_run_id`。即使两个批次都包含 `case_0001`，审阅 Trace 和
Score ID 也不会互相覆盖。

## 6. 人工审核

点击“打开人工审核队列”，进入 Langfuse：

1. 查看 `candidate_question`。
2. 展开 `full_session_context`，理解完整对话。
3. 必要时根据 Metadata 的来源 Trace ID 回到 Tracing 检查工具证据。
4. 对照 AI 预标注，填写或修改右侧人工字段。
5. 原答案需要改写时填写 `Corrected Output`；正常答案可留空。
6. 完成后点击 `Mark Completed`。

`Mark Completed` 只表示人工复核结束，不表示 Gold 已独立验证。

## 7. 运行产物

默认目录：

```text
results/session_recovery/recovery_YYYYMMDDTHHMMSS_微秒Z/
```

| 文件 | 内容 |
|---|---|
| `manifest.json` | 时间范围、Session/Case 数量、是否远端写入 |
| `sessions/*.json` | 从 Langfuse 重建的完整 Session |
| `ai_raw/*.json` | 每个 Session 的 AI 原始结构化判断 |
| `ai_cleaning_cases.csv` | 标准化后的 Case 表 |
| `candidate_cases.jsonl` | Case 与来源 Trace/Observation 映射 |
| `review_observations.jsonl` | 正式写入后生成的审阅 Trace/Observation ID |

运行产物默认被 `.gitignore` 忽略。

## 8. 导出人工审核结果

```powershell
python scripts/session_annotation_pipeline.py export `
  --env-file config/session_recovery.env `
  --cases results/session_recovery/<运行编号>/review_observations.jsonl `
  --queue-id <Queue ID> `
  --output results/session_recovery/<运行编号>/reviewed_cases.jsonl
```

导出的仍是“人工已审核候选”，不是自动发布的 Gold Dataset。

## 9. 常见问题

### Langfuse 401/403

检查 Key 是否属于目标 Project、Host 是否正确；不要使用网页密码代替 API Secret Key。

### AI API 不支持 `response_format`

设置 `AI_JSON_MODE=false`，但模型仍必须按 Prompt 返回合法 JSON。

### Case 数量为 0

可能是 Session 全部为非数据内容、数据没有 `sessionId`，或者 AI 正确执行范围门后返回了
空数组。检查 `manifest.json`、`sessions/` 和 `ai_raw/`，不要把非数据内容改成“拒绝入集”。

### 映射置信度低

表示 AI 整理后的问题与原 Session 单轮用户文本相似度不足。应人工重点检查来源 Trace，
不应直接进入正式 Dataset。

### 全量运行慢或花费高

AI 按 Session 逐个调用。先用时间范围和 Session 上限做小批回归，再运行全量。

## 10. 安全规则

- `config/session_recovery.env` 不得提交仓库。
- 日志、测试输出和文档不得打印真实 Key 或密码。
- 远端写入默认关闭，必须显式勾选并再次确认。
- “全部 Session + 上限 0 + 远端写入”会出现额外警告。
- 跨项目数据、缺少 `projectId`、非法 AI JSON或缺少必要配置时一律停止。
