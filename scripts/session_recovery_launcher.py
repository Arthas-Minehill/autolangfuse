"""Langfuse Session 数据回收一键入口。

GUI 流程：选择时间范围（或全部 Session）-> AI 预审 -> 生成候选 -> 可选写入
Langfuse Annotation Queue。远端写入默认关闭。
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
import threading
import traceback
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "session_recovery.env"
CONFIG_TEMPLATE = CONFIG_DIR / "session_recovery.env.example"
DEFAULT_PROMPT_PATH = CONFIG_DIR / "ai_review_prompt.md"
LEGACY_ENV_PATH = ROOT / ".env"
MAX_HTTP_BYTES = 64 * 1024 * 1024
CASE_FIELDS = (
    "recovery_run_id",
    "case_id",
    "source_session_id",
    "source_row",
    "original_input",
    "original_note",
    "original_output",
    "user_turns",
    "assistant_clarifications",
    "resolved_input",
    "primary_intent",
    "secondary_intent",
    "output_format",
    "execution_status",
    "clarification_status",
    "context_required",
    "gold_status",
    "exclude_reason",
    "review_notes",
    "source_trace_id",
    "snapshot_at",
)
SECRET_KEYS = {
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_LOGIN_PASSWORD",
    "AI_API_KEY",
}


class RecoveryError(RuntimeError):
    """可直接展示给操作者的错误。"""


@dataclass(frozen=True)
class RunOptions:
    start_text: str
    end_text: str
    all_sessions: bool
    limit: int
    write_remote: bool


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def quote_env_value(value: str) -> str:
    if not value:
        return ""
    if re.search(r"[\s#='\"]", value):
        return json.dumps(value, ensure_ascii=False)
    return value


def save_local_config(values: dict[str, str], path: Path = CONFIG_PATH) -> None:
    """只写本机忽略文件；日志中永远不输出值。"""
    ordered = [
        "LANGFUSE_HOST",
        "LANGFUSE_PROJECT_ID",
        "LANGFUSE_PROJECT_NAME",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_QUEUE_NAME",
        "LANGFUSE_QUEUE_ID",
        "LANGFUSE_LOGIN_EMAIL",
        "LANGFUSE_LOGIN_PASSWORD",
        "AI_API_BASE_URL",
        "AI_API_KEY",
        "AI_MODEL",
        "AI_TIMEOUT_SECONDS",
        "AI_MAX_SESSION_CHARS",
        "AI_JSON_MODE",
        "SESSION_TIMEZONE_OFFSET",
        "SESSION_DEFAULT_DAYS",
        "SESSION_DEFAULT_LIMIT",
        "SESSION_OUTPUT_ROOT",
        "AI_REVIEW_PROMPT_FILE",
    ]
    lines = [
        "# 本机数据回收配置。不要提交、分享或截图传播。",
        "# 网页账号密码只供手工登录；程序 API 调用不会读取浏览器 Cookie。",
        "",
    ]
    for key in ordered:
        lines.append(f"{key}={quote_env_value(str(values.get(key, '')))}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_local_config() -> dict[str, str]:
    defaults = parse_env_file(CONFIG_TEMPLATE)
    local = parse_env_file(CONFIG_PATH)
    legacy = parse_env_file(LEGACY_ENV_PATH)
    merged = {**defaults, **local}
    # 首次使用时迁移旧 .env 中已有的 Langfuse API 配置，避免重复填写。
    if not local:
        for key in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
            if legacy.get(key):
                merged[key] = legacy[key]
        save_local_config(merged)
    return merged


def redact_config(values: dict[str, str]) -> dict[str, str]:
    return {
        key: ("<已配置>" if value else "<未配置>") if key in SECRET_KEYS else value
        for key, value in values.items()
    }


def parse_offset(value: str) -> timezone:
    match = re.fullmatch(r"([+-])(\d{2}):(\d{2})", value.strip())
    if not match:
        raise RecoveryError("SESSION_TIMEZONE_OFFSET 必须形如 +08:00")
    sign = 1 if match.group(1) == "+" else -1
    delta = timedelta(hours=int(match.group(2)), minutes=int(match.group(3)))
    return timezone(sign * delta)


def parse_local_datetime(value: str, offset: timezone) -> datetime:
    try:
        parsed = datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S")
    except ValueError as exc:
        raise RecoveryError("时间格式必须是 YYYY-MM-DD HH:MM:SS") from exc
    return parsed.replace(tzinfo=offset)


def resolve_time_range(
    start_text: str,
    end_text: str,
    *,
    all_sessions: bool,
    offset_text: str,
) -> tuple[str | None, str | None]:
    if all_sessions:
        return None, None
    offset = parse_offset(offset_text)
    start = parse_local_datetime(start_text, offset)
    end = parse_local_datetime(end_text, offset)
    if start >= end:
        raise RecoveryError("开始时间必须早于结束时间")
    return (
        start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


def request_json(
    method: str,
    url: str,
    *,
    query: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    basic_auth: tuple[str, str] | None = None,
    bearer_token: str = "",
    timeout: int = 120,
) -> dict[str, Any]:
    query_values = {
        key: value for key, value in (query or {}).items() if value not in (None, "")
    }
    if query_values:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(query_values)
    headers = {"Accept": "application/json", "Content-Type": "application/json; charset=utf-8"}
    if basic_auth:
        token = base64.b64encode(f"{basic_auth[0]}:{basic_auth[1]}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"
    elif bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_HTTP_BYTES + 1)
    except urllib.error.HTTPError as exc:
        detail = exc.read(64 * 1024).decode("utf-8", errors="replace")[:1200]
        raise RecoveryError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RecoveryError(f"网络连接失败：{exc.reason}") from exc
    if len(raw) > MAX_HTTP_BYTES:
        raise RecoveryError("服务响应超过 64MB 安全限制")
    value = json.loads(raw.decode("utf-8")) if raw else {}
    if not isinstance(value, dict):
        raise RecoveryError("服务返回的顶层 JSON 不是 object")
    return value


def fetch_observations(
    config: dict[str, str],
    *,
    start_utc: str | None,
    end_utc: str | None,
    log: Callable[[str], None],
) -> list[dict[str, Any]]:
    host = config.get("LANGFUSE_HOST", "").rstrip("/")
    project_id = config.get("LANGFUSE_PROJECT_ID", "").strip()
    public_key = config.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = config.get("LANGFUSE_SECRET_KEY", "").strip()
    if not all((host, project_id, public_key, secret_key)):
        raise RecoveryError("Langfuse Host、Project ID、Public Key、Secret Key 必须配置")
    collected: list[dict[str, Any]] = []
    cursor: str | None = None
    page = 0
    while True:
        page += 1
        query: dict[str, Any] = {
            "projectId": project_id,
            "fields": "core,basic,time,io,metadata,model,usage,prompt,metrics,trace_context",
            "fromStartTime": start_utc,
            "toStartTime": end_utc,
            "limit": 1000,
            "cursor": cursor,
        }
        payload = request_json(
            "GET",
            f"{host}/api/public/v2/observations",
            query=query,
            basic_auth=(public_key, secret_key),
            timeout=120,
        )
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise RecoveryError("Langfuse observations 响应缺少 data 数组")
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_project_id = row.get("projectId")
            if row_project_id is None:
                raise RecoveryError("安全校验失败：返回 Observation 缺少 projectId，已停止")
            if str(row_project_id) != project_id:
                raise RecoveryError("安全校验失败：API 返回了其他项目的 Observation，已停止")
            collected.append(row)
        log(f"已读取 Observation 第 {page} 页，累计 {len(collected)} 条")
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        cursor = str(meta.get("cursor") or "") or None
        if not cursor:
            break
    return collected


def parse_json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def build_sessions(observations: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    no_session_count = 0
    for observation in observations:
        trace_id = str(observation.get("traceId") or "")
        session_id = str(observation.get("sessionId") or "")
        if not trace_id:
            continue
        if not session_id:
            no_session_count += 1
            continue
        grouped[session_id][trace_id].append(observation)
    sessions: list[dict[str, Any]] = []
    for session_id, traces in grouped.items():
        turns: list[dict[str, Any]] = []
        for trace_id, rows in traces.items():
            rows.sort(key=lambda item: str(item.get("startTime") or ""))
            roots = [item for item in rows if item.get("parentObservationId") in (None, "")]
            root = roots[0] if roots else rows[0]
            turns.append(
                {
                    "traceId": trace_id,
                    "startTime": root.get("startTime"),
                    "userInput": parse_json_value(root.get("input")),
                    "finalOutput": parse_json_value(root.get("output")),
                    "observations": rows,
                }
            )
        turns.sort(key=lambda item: str(item.get("startTime") or ""))
        sessions.append({"sessionId": session_id, "turnCount": len(turns), "turns": turns})
    sessions.sort(
        key=lambda session: str((session.get("turns") or [{}])[0].get("startTime") or "")
    )
    return sessions, no_session_count


def safe_filename(value: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", value).strip(" .")
    return (cleaned or "unnamed-session")[:180]


def truncate_text(value: Any, limit: int) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[截断 {len(text) - limit} 字符]"


def compact_session_for_ai(session: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """保留完整对话轮次，并给少量工具证据；避免把巨型工具输出全部送入模型。"""
    compact: dict[str, Any] = {"session_id": session.get("sessionId"), "turns": []}
    per_turn = max(2000, max_chars // max(1, len(session.get("turns", []))))
    for index, turn in enumerate(session.get("turns", []), 1):
        evidence: list[dict[str, Any]] = []
        for observation in turn.get("observations", []):
            if not isinstance(observation, dict):
                continue
            obs_type = str(observation.get("type") or "")
            if obs_type.upper() not in {"TOOL", "SPAN"}:
                continue
            evidence.append(
                {
                    "type": obs_type,
                    "name": observation.get("name"),
                    "status": observation.get("level") or observation.get("statusMessage"),
                    "input": truncate_text(observation.get("input"), 700),
                    "output": truncate_text(observation.get("output"), 1200),
                }
            )
            if len(evidence) >= 8:
                break
        compact["turns"].append(
            {
                "turn_index": index,
                "trace_id": turn.get("traceId"),
                "start_time": turn.get("startTime"),
                "user": truncate_text(turn.get("userInput"), per_turn // 3),
                "assistant": truncate_text(turn.get("finalOutput"), per_turn // 2),
                "tool_evidence": evidence,
            }
        )
    serialized = json.dumps(compact, ensure_ascii=False)
    if len(serialized) > max_chars:
        # 二次硬限制只影响送模内容；落盘的原始 Session 不截断。
        compact["warning"] = f"AI 输入因长度限制被截断到 {max_chars} 字符"
        compact["serialized_preview"] = serialized[:max_chars]
        compact.pop("turns", None)
    return compact


def ai_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def extract_ai_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RecoveryError("AI API 响应缺少 choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [str(item.get("text") or "") for item in content if isinstance(item, dict)]
        return "".join(texts)
    raise RecoveryError("AI API 响应缺少 message.content")


def parse_ai_cases(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise RecoveryError(f"AI 返回的内容不是合法 JSON：{exc}") from exc
    if isinstance(value, dict):
        value = value.get("cases")
    if not isinstance(value, list):
        raise RecoveryError("AI 返回 JSON 必须包含 cases 数组")
    if not all(isinstance(item, dict) for item in value):
        raise RecoveryError("AI 返回 cases 中存在非 object 项")
    return value


def review_session_with_ai(
    config: dict[str, str],
    prompt: str,
    session: dict[str, Any],
) -> list[dict[str, Any]]:
    api_key = config.get("AI_API_KEY", "").strip()
    model = config.get("AI_MODEL", "").strip()
    base_url = config.get("AI_API_BASE_URL", "").strip()
    if not all((api_key, model, base_url)):
        raise RecoveryError("AI_API_BASE_URL、AI_API_KEY、AI_MODEL 必须配置")
    try:
        max_chars = int(config.get("AI_MAX_SESSION_CHARS", "60000"))
        timeout = int(config.get("AI_TIMEOUT_SECONDS", "180"))
    except ValueError as exc:
        raise RecoveryError("AI_TIMEOUT_SECONDS 和 AI_MAX_SESSION_CHARS 必须是整数") from exc
    body: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": "请预审以下 RAW_SESSION：\n"
                + json.dumps(compact_session_for_ai(session, max_chars), ensure_ascii=False),
            },
        ],
    }
    if config.get("AI_JSON_MODE", "true").lower() in {"1", "true", "yes", "on"}:
        body["response_format"] = {"type": "json_object"}
    payload = request_json(
        "POST",
        ai_endpoint(base_url),
        body=body,
        bearer_token=api_key,
        timeout=timeout,
    )
    return parse_ai_cases(extract_ai_content(payload))


def normalize_case(
    raw: dict[str, Any],
    *,
    run_id: str,
    case_number: int,
    session_id: str,
    snapshot_at: str,
) -> dict[str, str]:
    def value(name: str) -> str:
        item = raw.get(name, "")
        if item is None:
            return ""
        if isinstance(item, (dict, list)):
            return json.dumps(item, ensure_ascii=False)
        return str(item).strip()

    resolved_input = value("resolved_input")
    original_input = value("original_input")
    if not resolved_input and not original_input:
        raise RecoveryError(f"Session {session_id} 的 AI Case 缺少 original_input/resolved_input")
    row = {field: "" for field in CASE_FIELDS}
    row.update(
        {
            "recovery_run_id": run_id,
            "case_id": f"case_{case_number:04d}",
            "source_session_id": session_id,
            "source_row": str(case_number + 1),
            "original_input": original_input,
            "original_note": value("original_note"),
            "original_output": value("original_output"),
            "user_turns": value("user_turns"),
            "assistant_clarifications": value("assistant_clarifications"),
            "resolved_input": resolved_input or original_input,
            "primary_intent": value("primary_intent"),
            "secondary_intent": value("secondary_intent"),
            "output_format": value("output_format"),
            "execution_status": value("execution_status") or "无法判断",
            "clarification_status": value("clarification_status") or "需要澄清",
            "context_required": value("context_required") or "上下文不足",
            "gold_status": value("gold_status") or "待独立验证",
            "exclude_reason": value("exclude_reason"),
            "review_notes": value("review_notes"),
            "source_trace_id": value("source_trace_id"),
            "snapshot_at": snapshot_at,
        }
    )
    return row


def write_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(configured: str, default: Path) -> Path:
    if not configured:
        return default
    path = Path(configured)
    return path if path.is_absolute() else ROOT / path


def apply_pipeline_env(config: dict[str, str]) -> None:
    for key in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        os.environ[key] = config.get(key, "")


def run_workflow(
    config: dict[str, str],
    options: RunOptions,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    import session_annotation_pipeline as pipeline

    start_utc, end_utc = resolve_time_range(
        options.start_text,
        options.end_text,
        all_sessions=options.all_sessions,
        offset_text=config.get("SESSION_TIMEZONE_OFFSET", "+08:00"),
    )
    if options.limit < 0:
        raise RecoveryError("Session 上限不能小于 0")
    now = datetime.now(timezone.utc)
    run_id = now.strftime("recovery_%Y%m%dT%H%M%S_%fZ")
    output_root = resolve_path(
        config.get("SESSION_OUTPUT_ROOT", "results/session_recovery"),
        ROOT / "results" / "session_recovery",
    )
    run_dir = output_root / run_id
    session_dir = run_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=False)
    log(f"运行编号：{run_id}")
    log("步骤 1/5：从 Langfuse 读取 Observation")
    observations = fetch_observations(
        config, start_utc=start_utc, end_utc=end_utc, log=log
    )
    sessions, no_session_count = build_sessions(observations)
    if options.limit:
        sessions = sessions[: options.limit]
    for session in sessions:
        path = session_dir / f"{safe_filename(str(session['sessionId']))}.json"
        path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "project_id": config.get("LANGFUSE_PROJECT_ID"),
        "from": start_utc,
        "to": end_utc,
        "all_sessions": options.all_sessions,
        "observation_count": len(observations),
        "session_count": len(sessions),
        "observations_without_session": no_session_count,
        "remote_write": options.write_remote,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"得到 {len(sessions)} 个 Session；跳过 {no_session_count} 条无 sessionId Observation")

    prompt_path = resolve_path(config.get("AI_REVIEW_PROMPT_FILE", ""), DEFAULT_PROMPT_PATH)
    if not prompt_path.exists():
        raise RecoveryError(f"找不到 AI Prompt：{prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8")
    log("步骤 2/5：调用大模型逐个预审 Session")
    case_rows: list[dict[str, str]] = []
    ai_raw_dir = run_dir / "ai_raw"
    ai_raw_dir.mkdir()
    for index, session in enumerate(sessions, 1):
        session_id = str(session.get("sessionId") or "")
        log(f"AI 预审 {index}/{len(sessions)}：{session_id}")
        raw_cases = review_session_with_ai(config, prompt, session)
        (ai_raw_dir / f"{safe_filename(session_id)}.json").write_text(
            json.dumps({"cases": raw_cases}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for raw_case in raw_cases:
            case_rows.append(
                normalize_case(
                    raw_case,
                    run_id=run_id,
                    case_number=len(case_rows) + 1,
                    session_id=session_id,
                    snapshot_at=now.isoformat().replace("+00:00", "Z"),
                )
            )
    cleaning_csv = run_dir / "ai_cleaning_cases.csv"
    write_csv(cleaning_csv, case_rows)
    log(f"AI 保留 {len(case_rows)} 个数据类 Case")

    log("步骤 3/5：映射 Case 与原始 Trace/Observation")
    candidates = []
    for row in case_rows:
        session_path = session_dir / f"{safe_filename(row['source_session_id'])}.json"
        session = json.loads(session_path.read_text(encoding="utf-8"))
        candidates.append(pipeline.build_candidate(row, session))
    candidate_path = run_dir / "candidate_cases.jsonl"
    pipeline.write_jsonl(candidate_path, candidates)
    low_confidence = sum(float(item.get("mapping_confidence") or 0) < 0.72 for item in candidates)
    manifest.update({"case_count": len(candidates), "low_confidence_count": low_confidence})
    if low_confidence:
        log(f"警告：{low_confidence} 个 Case 的来源映射置信度低于 0.72")
        if options.write_remote:
            raise RecoveryError(
                "存在低置信度来源映射。为避免创建无法可靠回溯的远端 Candidate，"
                "本次正式写入已在任何远端变更前停止；请先检查 candidate_cases.jsonl。"
            )

    review_path = run_dir / "review_observations.jsonl"
    if not options.write_remote:
        log("步骤 4/5：预览模式，不创建 Langfuse 审阅 Observation")
        log("步骤 5/5：预览完成；检查输出后再勾选“实际写入 Langfuse”")
    else:
        if not candidates:
            log("没有可写入的 Case，远端未发生变更")
        else:
            apply_pipeline_env(config)
            log("步骤 4/5：创建审阅 Observation 并写入 AI 预标注")
            materialize_args = argparse.Namespace(
                cases=str(candidate_path),
                session_dir=str(session_dir),
                output=str(review_path),
                environment="annotation-candidate",
                limit=0,
                include_rejected=False,
                commit=True,
            )
            pipeline.command_materialize(materialize_args)
            prelabel_args = argparse.Namespace(
                cases=str(review_path), environment="annotation-candidate", limit=0, commit=True
            )
            pipeline.command_prelabel(prelabel_args)
            log("步骤 5/5：加入 Annotation Queue，保持 Pending 等待人工复核")
            enqueue_args = argparse.Namespace(
                cases=str(review_path),
                queue_id=config.get("LANGFUSE_QUEUE_ID", ""),
                queue_name=config.get("LANGFUSE_QUEUE_NAME", "数据回收-准入审核-POC"),
                min_confidence=0.72,
                limit=0,
                include_rejected=False,
                commit=True,
            )
            pipeline.command_enqueue(enqueue_args)
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"完成。运行产物：{run_dir}")
    return {**manifest, "run_dir": str(run_dir), "candidate_path": str(candidate_path)}


def test_connections(config: dict[str, str]) -> list[str]:
    messages: list[str] = []
    host = config.get("LANGFUSE_HOST", "").rstrip("/")
    project_id = config.get("LANGFUSE_PROJECT_ID", "")
    request_json(
        "GET",
        f"{host}/api/public/v2/observations",
        query={"projectId": project_id, "fields": "core", "limit": 1},
        basic_auth=(config.get("LANGFUSE_PUBLIC_KEY", ""), config.get("LANGFUSE_SECRET_KEY", "")),
        timeout=30,
    )
    messages.append("Langfuse API：连接成功")
    if all(config.get(key, "").strip() for key in ("AI_API_BASE_URL", "AI_API_KEY", "AI_MODEL")):
        body: dict[str, Any] = {
            "model": config["AI_MODEL"],
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "只返回 JSON。"},
                {"role": "user", "content": '返回 {"cases": []}'},
            ],
        }
        if config.get("AI_JSON_MODE", "true").lower() in {"1", "true", "yes", "on"}:
            body["response_format"] = {"type": "json_object"}
        payload = request_json(
            "POST",
            ai_endpoint(config["AI_API_BASE_URL"]),
            body=body,
            bearer_token=config["AI_API_KEY"],
            timeout=int(config.get("AI_TIMEOUT_SECONDS", "180")),
        )
        parse_ai_cases(extract_ai_content(payload))
        messages.append("大模型 API：连接及 JSON 输出测试成功")
    else:
        messages.append("大模型 API：尚未完整配置，已跳过联网测试")
    return messages


def queue_url(config: dict[str, str]) -> str:
    queue_id = config.get("LANGFUSE_QUEUE_ID", "").strip()
    return (
        f"{config.get('LANGFUSE_HOST', '').rstrip('/')}/project/"
        f"{config.get('LANGFUSE_PROJECT_ID', '')}/annotation-queues/{queue_id}"
    )


class RecoveryApp:
    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import messagebox, scrolledtext, ttk

        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.config = ensure_local_config()
        self.root = tk.Tk()
        self.root.title("Langfuse Session 数据回收")
        self.root.geometry("1040x820")
        self.root.minsize(920, 720)
        self.fields: dict[str, Any] = {}
        self._busy = False
        self._build(scrolledtext)

    def _build(self, scrolledtext: Any) -> None:
        ttk = self.ttk
        main = ttk.Frame(self.root, padding=14)
        main.pack(fill="both", expand=True)

        scope = ttk.LabelFrame(main, text="1. 选择 Session 范围", padding=10)
        scope.pack(fill="x")
        now = datetime.now()
        default_days = int(self.config.get("SESSION_DEFAULT_DAYS", "7") or 7)
        self.start_var = self.tk.StringVar(value=(now - timedelta(days=default_days)).strftime("%Y-%m-%d 00:00:00"))
        self.end_var = self.tk.StringVar(value=now.strftime("%Y-%m-%d %H:%M:%S"))
        self.all_var = self.tk.BooleanVar(value=False)
        self.limit_var = self.tk.StringVar(value=self.config.get("SESSION_DEFAULT_LIMIT", "10"))
        ttk.Label(scope, text="开始时间").grid(row=0, column=0, sticky="w")
        self.start_entry = ttk.Entry(scope, textvariable=self.start_var, width=24)
        self.start_entry.grid(row=0, column=1, padx=(6, 18))
        ttk.Label(scope, text="结束时间").grid(row=0, column=2, sticky="w")
        self.end_entry = ttk.Entry(scope, textvariable=self.end_var, width=24)
        self.end_entry.grid(row=0, column=3, padx=(6, 18))
        ttk.Checkbutton(
            scope,
            text="观察全部 Session（忽略时间范围）",
            variable=self.all_var,
            command=self._toggle_all,
        ).grid(row=0, column=4, sticky="w")
        ttk.Label(scope, text="Session 上限（0=不限）").grid(row=1, column=0, pady=(10, 0), sticky="w")
        ttk.Entry(scope, textvariable=self.limit_var, width=12).grid(row=1, column=1, pady=(10, 0), sticky="w")
        ttk.Label(scope, text="建议第一次填 1～10；全量运行前先做小批验证。", foreground="#875a00").grid(
            row=1, column=2, columnspan=3, pady=(10, 0), sticky="w"
        )

        config_frame = ttk.LabelFrame(main, text="2. 本机配置（保存在 config/session_recovery.env）", padding=10)
        config_frame.pack(fill="x", pady=(12, 0))
        field_specs = [
            ("LANGFUSE_HOST", "Langfuse 地址", False),
            ("LANGFUSE_PROJECT_ID", "Project ID", False),
            ("LANGFUSE_QUEUE_ID", "Queue ID", False),
            ("LANGFUSE_PUBLIC_KEY", "Public Key", True),
            ("LANGFUSE_SECRET_KEY", "Secret Key", True),
            ("LANGFUSE_LOGIN_EMAIL", "网页账号", False),
            ("LANGFUSE_LOGIN_PASSWORD", "网页密码", True),
            ("AI_API_BASE_URL", "大模型 API 地址", False),
            ("AI_MODEL", "模型名称", False),
            ("AI_API_KEY", "大模型 API Key", True),
        ]
        for index, (key, label, secret) in enumerate(field_specs):
            row, side = divmod(index, 2)
            col = side * 2
            ttk.Label(config_frame, text=label).grid(row=row, column=col, sticky="w", pady=3)
            var = self.tk.StringVar(value=self.config.get(key, ""))
            entry = ttk.Entry(config_frame, textvariable=var, width=42, show="*" if secret else "")
            entry.grid(row=row, column=col + 1, sticky="ew", padx=(6, 18), pady=3)
            self.fields[key] = var
        config_frame.columnconfigure(1, weight=1)
        config_frame.columnconfigure(3, weight=1)
        ttk.Button(config_frame, text="保存配置", command=self._save_config).grid(row=5, column=0, pady=(10, 0), sticky="w")
        ttk.Button(config_frame, text="测试连接", command=self._test_config).grid(row=5, column=1, pady=(10, 0), sticky="w")
        ttk.Label(
            config_frame,
            text="网页账号密码只供手工登录；程序调用 Langfuse 使用 Public/Secret Key。",
            foreground="#555555",
        ).grid(row=5, column=2, columnspan=2, pady=(10, 0), sticky="w")

        action = ttk.LabelFrame(main, text="3. 执行", padding=10)
        action.pack(fill="x", pady=(12, 0))
        self.remote_var = self.tk.BooleanVar(value=False)
        ttk.Checkbutton(
            action,
            text="实际写入 Langfuse（创建 Candidate、AI 预标注并加入待审队列）",
            variable=self.remote_var,
        ).pack(side="left")
        self.run_button = ttk.Button(action, text="开始运行", command=self._start_run)
        self.run_button.pack(side="right")
        ttk.Button(action, text="打开人工审核队列", command=self._open_queue).pack(side="right", padx=(0, 8))

        ttk.Label(main, text="运行日志（不会显示密码或密钥）").pack(anchor="w", pady=(12, 4))
        self.log_box = scrolledtext.ScrolledText(main, height=19, wrap="word", state="disabled")
        self.log_box.pack(fill="both", expand=True)
        self.status_var = self.tk.StringVar(value="就绪")
        ttk.Label(main, textvariable=self.status_var).pack(anchor="w", pady=(6, 0))

    def _toggle_all(self) -> None:
        state = "disabled" if self.all_var.get() else "normal"
        self.start_entry.configure(state=state)
        self.end_entry.configure(state=state)

    def _collect_config(self) -> dict[str, str]:
        updated = dict(self.config)
        for key, var in self.fields.items():
            updated[key] = var.get().strip()
        return updated

    def _save_config(self) -> None:
        self.config = self._collect_config()
        save_local_config(self.config)
        self._log("配置已保存到本机忽略文件；未打印任何密钥。")

    def _log(self, message: str) -> None:
        self.root.after(0, self._append_log, message)

    def _append_log(self, message: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _set_busy(self, busy: bool, status: str) -> None:
        self._busy = busy
        self.status_var.set(status)
        self.run_button.configure(state="disabled" if busy else "normal")

    def _background(self, job: Callable[[], None], status: str) -> None:
        if self._busy:
            return
        self._set_busy(True, status)

        def runner() -> None:
            try:
                job()
            except Exception as exc:  # GUI 边界：转换为可见错误，同时保留本地 traceback。
                self._log(f"失败：{exc}")
                self._log(traceback.format_exc())
                self.root.after(0, lambda: self.messagebox.showerror("执行失败", str(exc)))
            finally:
                self.root.after(0, lambda: self._set_busy(False, "就绪"))

        threading.Thread(target=runner, daemon=True).start()

    def _test_config(self) -> None:
        self._save_config()

        def job() -> None:
            for message in test_connections(self.config):
                self._log(message)
            self.root.after(0, lambda: self.messagebox.showinfo("测试完成", "连接测试已完成，请查看日志。"))

        self._background(job, "正在测试连接...")

    def _start_run(self) -> None:
        self._save_config()
        try:
            limit = int(self.limit_var.get().strip() or "0")
        except ValueError:
            self.messagebox.showerror("参数错误", "Session 上限必须是整数")
            return
        if self.remote_var.get():
            message = "这会在 Langfuse 创建审阅 Observation、写入 AI 预标注并加入待审队列。"
            if self.all_var.get() and limit == 0:
                message += "\n\n你选择了全部 Session 且没有上限，可能产生大量 API 调用和队列项目。"
            if not self.messagebox.askyesno("确认远端写入", message + "\n\n确认继续吗？"):
                return
        options = RunOptions(
            start_text=self.start_var.get(),
            end_text=self.end_var.get(),
            all_sessions=self.all_var.get(),
            limit=limit,
            write_remote=self.remote_var.get(),
        )

        def job() -> None:
            result = run_workflow(self.config, options, self._log)
            self._log(f"结果摘要：Session={result['session_count']}，Case={result['case_count']}")
            self.root.after(0, lambda: self.messagebox.showinfo("运行完成", f"产物目录：\n{result['run_dir']}"))

        self._background(job, "数据回收运行中...")

    def _open_queue(self) -> None:
        self.config = self._collect_config()
        url = queue_url(self.config)
        if not self.config.get("LANGFUSE_QUEUE_ID"):
            self.messagebox.showerror("缺少配置", "请先配置 LANGFUSE_QUEUE_ID")
            return
        webbrowser.open(url)

    def run(self) -> None:
        self.root.mainloop()


def self_test() -> int:
    tests = [
        resolve_time_range(
            "2026-09-01 00:00:00",
            "2026-09-02 00:00:00",
            all_sessions=False,
            offset_text="+08:00",
        )
        == ("2026-08-31T16:00:00Z", "2026-09-01T16:00:00Z"),
        resolve_time_range("", "", all_sessions=True, offset_text="+08:00") == (None, None),
        parse_ai_cases('{"cases": []}') == [],
        parse_ai_cases('```json\n{"cases": [{"resolved_input": "查询库存"}]}\n```')[0][
            "resolved_input"
        ]
        == "查询库存",
    ]
    sample_observations = [
        {
            "projectId": "p",
            "sessionId": "s",
            "traceId": "t",
            "id": "root",
            "parentObservationId": None,
            "startTime": "2026-09-01T00:00:00Z",
            "input": '{"role":"user","content":"查询库存"}',
            "output": '{"role":"assistant","content":"结果"}',
        },
        {
            "projectId": "p",
            "sessionId": "s",
            "traceId": "t",
            "id": "child",
            "parentObservationId": "root",
            "startTime": "2026-09-01T00:00:01Z",
        },
    ]
    sessions, skipped = build_sessions(sample_observations)
    tests.extend([len(sessions) == 1, sessions[0]["turnCount"] == 1, skipped == 0])
    if not all(tests):
        raise AssertionError(f"自检失败：{tests}")
    config = ensure_local_config()
    print(
        json.dumps(
            {
                "self_test": "passed",
                "config_path": str(CONFIG_PATH),
                "config": redact_config(config),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="只运行离线自检")
    parser.add_argument("--test-connections", action="store_true", help="测试 Langfuse/AI API 连接")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.test_connections:
        config = ensure_local_config()
        for message in test_connections(config):
            print(message)
        return 0
    RecoveryApp().run()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RecoveryError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1)
