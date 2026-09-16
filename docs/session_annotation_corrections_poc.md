# Session 数据回收与 Corrected Output POC

## 目标

以完整 Session 作为 AI 理解上下文，以候选 Case 作为人工判定单位，在 Langfuse
Annotation Queue 中完成人工准入、问题优化和答案修订。

Langfuse 的 Corrected Output 会存储为：

- `dataType = CORRECTION`
- `name = output`
- 每个 Trace 或 Observation 最多一个 Corrected Output

因此不能把“同一个 Observation 拆出的多个 Case”直接挂回原 Observation。POC 会为每个
候选 Case 创建一个唯一的审阅 Observation，并把完整 Session 上下文放进它的 Input。
原始 Session、Trace、Observation ID 保存在 metadata 中，用于审计和回溯。

## 数据流

```text
原始 Session JSON + AI 清洗 CSV
          |
          v
prepare: 候选 Case + 原始来源映射
          |
          v
materialize: 每个 Case 一个唯一审阅 Observation
          |
          v
enqueue: 数据回收-准入审核-POC
          |
          v
人工填写三个 Score Config + Corrected Output
          |
          v
export: reviewed_cases.jsonl
```

## 凭据

复制 `.env.example` 为 `.env`，仅在本机填写以下变量：

```dotenv
LANGFUSE_HOST=https://langfuse.lecangs.com
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

不要把 `.env`、API Key 或 Authorization Header 提交到仓库或粘贴到对话中。

## 命令

以下命令均从仓库根目录运行。

### 1. 生成候选 Case 映射

```powershell
python scripts/session_annotation_pipeline.py prepare `
  --cleaning-csv D:\lege_2608\outputs\session_cleaning_20260826\session_cleaning_codex.csv `
  --session-dir D:\lege_2608\ori_dataset\session_json `
  --output results\session_annotation_poc\candidate_cases.jsonl
```

程序会报告低置信度映射、缺失 Session 和多个 Case 共用同一原始 Observation 的冲突。
当前样本存在原始 Observation 冲突，所以必须执行下一步，不能直接把原始 Observation
批量加入队列。

### 2. 预览审阅 Observation

```powershell
python scripts/session_annotation_pipeline.py materialize `
  --cases results\session_annotation_poc\candidate_cases.jsonl `
  --session-dir D:\lege_2608\ori_dataset\session_json `
  --output results\session_annotation_poc\review_observations.jsonl
```

默认只预览，不写入 Langfuse。确认数量和环境后追加 `--commit`：

```powershell
python scripts/session_annotation_pipeline.py materialize `
  --cases results\session_annotation_poc\candidate_cases.jsonl `
  --session-dir D:\lege_2608\ori_dataset\session_json `
  --output results\session_annotation_poc\review_observations.jsonl `
  --commit
```

默认排除 `gold_status=拒绝入集` 的 Case；如果确实要复核这些记录，再显式增加
`--include-rejected`。

### 3. 写入 AI 预标注

```powershell
python scripts/session_annotation_pipeline.py prelabel `
  --cases results\session_annotation_poc\review_observations.jsonl
```

确认预览后追加 `--commit`。预标注以 `source=API` 写入，包含：

- `人工确认是否数据类`
- `数据集准入状态`
- `优化后问题`

它不会填写右侧人工字段，也不会把任务标记为 Completed。人工可在详细视图查看 AI
建议，再独立填写或修改人工标注。

### 4. 预览并加入 Annotation Queue

```powershell
python scripts/session_annotation_pipeline.py enqueue `
  --cases results\session_annotation_poc\review_observations.jsonl `
  --queue-name 数据回收-准入审核-POC
```

确认预览后追加 `--commit`。脚本会跳过队列中已有的 Observation，不会重复添加。

### 5. 人工审核

在队列中填写：

- `人工确认是否数据类`
- `数据集准入状态`
- `优化后问题`
- `Corrected Output`：只有答案确实需要重写时填写；正常答案可以留空

填写完成后点击 `Mark Completed`。

### 6. 导出审核结果

```powershell
python scripts/session_annotation_pipeline.py export `
  --cases results\session_annotation_poc\review_observations.jsonl `
  --queue-name 数据回收-准入审核-POC `
  --output results\session_annotation_poc\reviewed_cases.jsonl
```

导出文件同时保留 AI 清洗结果、人工三项标签、Corrected Output 和全部来源 ID。
它仍是“已审核候选数据”，不是独立验证过的 Gold Dataset。

## POC 安全约束

- 所有远端写操作默认 dry-run，必须显式传入 `--commit`。
- 原始 Observation 一对多冲突时停止，不静默合并 Case。
- 不把 `拒绝入集` 默认送入标注队列。
- Corrected Output 不覆盖原始 Output。
- 导出结果不自动标记为已验证 Gold。
