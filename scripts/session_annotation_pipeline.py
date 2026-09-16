"""Session 清洗结果与 Langfuse 人工标注队列之间的桥接工具。

流程：
1. prepare：把 Session 清洗 CSV 与原始 Session JSON 关联，生成候选 Case JSONL。
2. materialize：为每个 Case 创建包含完整 Session 上下文的唯一审阅 Observation。
3. enqueue：把审阅 Observation 加入 Annotation Queue。
4. export：导出已完成队列的人工分数与 Corrected Output。

所有远端写操作默认关闭；只有显式传入 ``--commit`` 才会写入 Langfuse。
"""
from __future__ import annotations

import argparse
import base64
import csv
import difflib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOST = "https://langfuse.lecangs.com"
DEFAULT_SCORE_NAMES = (
    "人工确认是否数据类",
    "数据集准入状态",
    "优化后问题",
    "output",
)
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class PipelineError(RuntimeError):
    """可向操作者直接展示的流程错误。"""


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def collect_message_text(value: Any, *, role: str | None = None) -> list[str]:
    result: list[str] = []
    if isinstance(value, str):
        if value.strip():
            result.append(value.strip())
    elif isinstance(value, list):
        for item in value:
            result.extend(collect_message_text(item, role=role))
    elif isinstance(value, dict):
        current_role = str(value.get("role") or "").lower()
        if role is None or not current_role or current_role == role:
            content = value.get("content")
            if isinstance(content, str) and content.strip():
                result.append(content.strip())
        for key, child in value.items():
            if key not in {"role", "content"} and isinstance(child, (dict, list)):
                result.extend(collect_message_text(child, role=role))
    return result


def turn_user_text(turn: dict[str, Any]) -> str:
    texts = collect_message_text(turn.get("userInput"), role="user")
    if not texts:
        texts = collect_message_text(turn.get("userInput"))
    return "\n".join(texts)


def turn_assistant_text(turn: dict[str, Any]) -> str:
    texts = collect_message_text(turn.get("finalOutput"), role="assistant")
    if not texts:
        texts = collect_message_text(turn.get("finalOutput"))
    return "\n".join(texts)


def session_review_context(session: dict[str, Any]) -> list[dict[str, Any]]:
    context: list[dict[str, Any]] = []
    for index, turn in enumerate(session.get("turns", []), 1):
        if not isinstance(turn, dict):
            continue
        context.append(
            {
                "turn_index": index,
                "trace_id": turn.get("traceId"),
                "user": turn_user_text(turn),
                "assistant": turn_assistant_text(turn),
            }
        )
    return context


def root_observation(turn: dict[str, Any]) -> dict[str, Any] | None:
    observations = turn.get("observations")
    if not isinstance(observations, list):
        return None
    candidates = [item for item in observations if isinstance(item, dict)]
    for observation in candidates:
        if observation.get("isRootObservation") is True:
            return observation
    for observation in candidates:
        if observation.get("parentObservationId") in (None, ""):
            return observation
    return candidates[0] if candidates else None


def score_turn(case_texts: Iterable[str], turn_text: str) -> float:
    normalized_turn = normalize_text(turn_text)
    if not normalized_turn:
        return 0.0
    best = 0.0
    for case_text in case_texts:
        normalized_case = normalize_text(case_text)
        if not normalized_case:
            continue
        if normalized_turn in normalized_case or normalized_case in normalized_turn:
            overlap = min(len(normalized_turn), len(normalized_case)) / max(
                len(normalized_turn), len(normalized_case)
            )
            best = max(best, 0.8 + 0.2 * overlap)
        else:
            best = max(
                best,
                difflib.SequenceMatcher(None, normalized_case, normalized_turn).ratio(),
            )
    return best


def choose_source_turns(row: dict[str, str], turns: list[dict[str, Any]]) -> tuple[list[int], float, str]:
    case_texts = [
        row.get("original_input", ""),
        row.get("user_turns", ""),
        row.get("resolved_input", ""),
    ]
    scores = [score_turn(case_texts, turn_user_text(turn)) for turn in turns]
    if not scores or max(scores) <= 0:
        return [], 0.0, "没有可比较的用户输入"

    best = max(scores)
    selected = [index for index, score in enumerate(scores) if score >= max(0.72, best - 0.08)]
    if not selected:
        selected = [scores.index(best)]
    confidence = round(best, 4)
    if confidence >= 0.9:
        reason = "文本精确包含或高度一致"
    elif confidence >= 0.72:
        reason = "文本近似匹配"
    else:
        reason = "低置信度候选，必须人工确认锚点"
    return selected, confidence, reason


def build_candidate(row: dict[str, str], session: dict[str, Any]) -> dict[str, Any]:
    turns = [item for item in session.get("turns", []) if isinstance(item, dict)]
    selected, confidence, reason = choose_source_turns(row, turns)
    selected_turns = [turns[index] for index in selected]
    anchor_turn = selected_turns[-1] if selected_turns else None
    anchor_observation = root_observation(anchor_turn) if anchor_turn else None
    trace_ids = [str(turn.get("traceId") or "") for turn in selected_turns]
    trace_ids = list(dict.fromkeys(item for item in trace_ids if item))
    return {
        "recovery_run_id": row.get("recovery_run_id", ""),
        "case_id": row.get("case_id", ""),
        "source_session_id": row.get("source_session_id", ""),
        "source_turn_indexes": [index + 1 for index in selected],
        "source_trace_ids": trace_ids,
        "anchor_trace_id": str(anchor_turn.get("traceId") or "") if anchor_turn else "",
        "anchor_observation_id": str(anchor_observation.get("id") or "") if anchor_observation else "",
        "mapping_confidence": confidence,
        "mapping_reason": reason,
        "original_input": row.get("original_input", ""),
        "original_output": row.get("original_output", ""),
        "resolved_input": row.get("resolved_input", ""),
        "primary_intent": row.get("primary_intent", ""),
        "secondary_intent": row.get("secondary_intent", ""),
        "execution_status": row.get("execution_status", ""),
        "clarification_status": row.get("clarification_status", ""),
        "context_required": row.get("context_required", ""),
        "gold_status": row.get("gold_status", ""),
        "review_notes": row.get("review_notes", ""),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise PipelineError(f"{path} 第 {line_number} 行不是合法 JSON") from exc
        if not isinstance(value, dict):
            raise PipelineError(f"{path} 第 {line_number} 行必须是 object")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class LangfuseApi:
    def __init__(self) -> None:
        self.host = os.getenv("LANGFUSE_HOST", DEFAULT_HOST).rstrip("/")
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        if not self.public_key or not self.secret_key:
            raise PipelineError(
                "缺少 LANGFUSE_PUBLIC_KEY 或 LANGFUSE_SECRET_KEY；请在项目 .env 中配置，"
                "不要把密钥提交到仓库或粘贴到对话中"
            )

    def request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        query = {key: value for key, value in (query or {}).items() if value not in (None, "")}
        url = f"{self.host}/api/public/{path.lstrip('/')}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        token = base64.b64encode(
            f"{self.public_key}:{self.secret_key}".encode("utf-8")
        ).decode("ascii")
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            url,
            data=payload,
            method=method,
            headers={
                "Authorization": f"Basic {token}",
                "Accept": "application/json",
                "Content-Type": "application/json; charset=utf-8",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            detail = exc.read(64 * 1024).decode("utf-8", errors="replace")[:1200]
            raise PipelineError(f"Langfuse API {method} {path} 失败: HTTP {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise PipelineError(f"无法连接 Langfuse: {exc.reason}") from exc
        if len(raw) > MAX_RESPONSE_BYTES:
            raise PipelineError("Langfuse API 响应超过 16MB 限制")
        value = json.loads(raw.decode("utf-8")) if raw else {}
        return value if isinstance(value, dict) else {"data": value}

    def list_pages(self, path: str, *, query: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        base_query = dict(query or {})
        base_query.setdefault("limit", 100)
        collected: list[dict[str, Any]] = []
        page = 1
        cursor: str | None = None
        while True:
            current = dict(base_query)
            if cursor:
                current["cursor"] = cursor
            elif "v3/" not in path:
                current["page"] = page
            response = self.request("GET", path, query=current)
            data = response.get("data")
            if isinstance(data, list):
                collected.extend(item for item in data if isinstance(item, dict))
            meta = response.get("meta") if isinstance(response.get("meta"), dict) else {}
            cursor = str(meta.get("cursor") or "") or None
            if cursor:
                continue
            total_pages = int(meta.get("totalPages") or meta.get("total_pages") or page)
            if page >= total_pages or not data:
                break
            page += 1
        return collected


def resolve_queue(api: LangfuseApi, *, queue_id: str, queue_name: str) -> dict[str, Any]:
    if queue_id:
        return api.request("GET", f"annotation-queues/{urllib.parse.quote(queue_id, safe='')}")
    queues = api.list_pages("annotation-queues")
    matches = [queue for queue in queues if str(queue.get("name") or "") == queue_name]
    if len(matches) != 1:
        raise PipelineError(f"按名称查找队列时需要唯一匹配，实际找到 {len(matches)} 个: {queue_name}")
    return matches[0]


def command_prepare(args: argparse.Namespace) -> int:
    csv_path = Path(args.cleaning_csv)
    session_dir = Path(args.session_dir)
    output = Path(args.output)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    candidates: list[dict[str, Any]] = []
    missing_sessions: list[str] = []
    for row in rows:
        session_id = str(row.get("source_session_id") or "").strip()
        session_path = session_dir / f"{session_id}.json"
        if not session_path.exists():
            missing_sessions.append(session_id)
            continue
        session = json.loads(session_path.read_text(encoding="utf-8"))
        candidates.append(build_candidate(row, session))

    write_jsonl(output, candidates)
    anchor_counts = Counter(item["anchor_observation_id"] for item in candidates if item["anchor_observation_id"])
    collisions = {key: count for key, count in anchor_counts.items() if count > 1}
    summary = {
        "output": str(output),
        "total": len(candidates),
        "mapped": sum(bool(item["anchor_observation_id"]) for item in candidates),
        "low_confidence": sum(item["mapping_confidence"] < args.min_confidence for item in candidates),
        "missing_sessions": len(missing_sessions),
        "anchor_collisions": len(collisions),
        "safe_to_enqueue": not missing_sessions
        and not collisions
        and all(item["mapping_confidence"] >= args.min_confidence for item in candidates),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["safe_to_enqueue"] else 2


def command_enqueue(args: argparse.Namespace) -> int:
    candidates = read_jsonl(Path(args.cases))
    selected = [
        item
        for item in candidates
        if item.get("anchor_observation_id")
        and float(item.get("mapping_confidence") or 0) >= args.min_confidence
        and (args.include_rejected or item.get("gold_status") != "拒绝入集")
    ]
    if args.limit > 0:
        selected = selected[: args.limit]
    object_field = "review_observation_id" if all(
        item.get("review_observation_id") for item in selected
    ) else "anchor_observation_id"
    counts = Counter(str(item[object_field]) for item in selected)
    collisions = [key for key, count in counts.items() if count > 1]
    if collisions:
        raise PipelineError(
            f"有 {len(collisions)} 个 Observation 同时对应多个 Case；请先执行 materialize，已停止"
        )
    if not args.commit:
        print(
            json.dumps(
                {"commit": False, "would_enqueue": len(selected), "object_type": "OBSERVATION"},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    api = LangfuseApi()
    queue = resolve_queue(api, queue_id=args.queue_id, queue_name=args.queue_name)
    queue_id = str(queue.get("id") or "")
    existing = api.list_pages(f"annotation-queues/{urllib.parse.quote(queue_id, safe='')}/items")
    existing_ids = {str(item.get("objectId") or "") for item in existing}
    created = 0
    skipped = 0
    for item in selected:
        object_id = str(item[object_field])
        if object_id in existing_ids:
            skipped += 1
            continue
        api.request(
            "POST",
            f"annotation-queues/{urllib.parse.quote(queue_id, safe='')}/items",
            body={"objectId": object_id, "objectType": "OBSERVATION", "status": "PENDING"},
        )
        existing_ids.add(object_id)
        created += 1
    print(json.dumps({"commit": True, "queue_id": queue_id, "created": created, "skipped": skipped}, ensure_ascii=False, indent=2))
    return 0


def command_materialize(args: argparse.Namespace) -> int:
    """为每个候选 Case 创建唯一的审阅 Observation，解决一对多冲突。"""
    cases = [
        item
        for item in read_jsonl(Path(args.cases))
        if args.include_rejected or item.get("gold_status") != "拒绝入集"
    ]
    if args.limit > 0:
        cases = cases[: args.limit]
    if not args.commit:
        print(
            json.dumps(
                {
                    "commit": False,
                    "would_materialize": len(cases),
                    "environment": args.environment,
                    "output": args.output,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    api = LangfuseApi()  # 先校验凭据，且不输出凭据内容。
    try:
        from langfuse import Langfuse
    except ImportError as exc:
        raise PipelineError("缺少 langfuse SDK，请执行 pip install -r requirements.txt") from exc

    client = Langfuse(
        public_key=api.public_key,
        secret_key=api.secret_key,
        host=api.host,
        environment=args.environment,
    )
    session_dir = Path(args.session_dir)
    result: list[dict[str, Any]] = []
    for case in cases:
        session_id = str(case.get("source_session_id") or "")
        session_path = session_dir / f"{session_id}.json"
        if not session_path.exists():
            raise PipelineError(f"找不到来源 Session: {session_path}")
        session = json.loads(session_path.read_text(encoding="utf-8"))
        recovery_run_id = str(case.get("recovery_run_id") or "legacy")
        trace_id = client.create_trace_id(
            seed=f"dataset-recovery:{recovery_run_id}:{case.get('case_id')}"
        )
        span = client.start_observation(
            trace_context={"trace_id": trace_id},
            name="Dataset Recovery Candidate",
            as_type="span",
            input={
                "case_id": case.get("case_id"),
                "candidate_question": case.get("resolved_input") or case.get("original_input"),
                "full_session_context": session_review_context(session),
            },
            output={"original_answer": case.get("original_output")},
            metadata={
                "workflow": "dataset-recovery-human-review",
                "recovery_run_id": recovery_run_id,
                "source_session_id": session_id,
                "source_trace_ids": case.get("source_trace_ids", []),
                "source_anchor_observation_id": case.get("anchor_observation_id"),
                "mapping_confidence": case.get("mapping_confidence"),
                "primary_intent": case.get("primary_intent"),
                "secondary_intent": case.get("secondary_intent"),
            },
        )
        span.end()
        result.append(
            {
                **case,
                "review_trace_id": span.trace_id,
                "review_observation_id": span.id,
            }
        )
    client.flush()
    write_jsonl(Path(args.output), result)
    print(
        json.dumps(
            {
                "commit": True,
                "materialized": len(result),
                "environment": args.environment,
                "output": args.output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def proposed_admission_status(case: dict[str, Any]) -> tuple[str, str]:
    """从既有 AI 清洗字段生成保守的队列预标注。"""
    notes = str(case.get("review_notes") or "")
    gold_status = str(case.get("gold_status") or "")
    clarification = str(case.get("clarification_status") or "")
    context_required = str(case.get("context_required") or "")
    if "拆分" in notes or "无法可靠拆分" in notes:
        return "需要拆分", "清洗备注显示该候选仍包含多个独立目标"
    if "澄清" in gold_status or clarification in {"待澄清", "需要澄清", "澄清未完成"}:
        return "需要澄清", "既有清洗结果表明关键口径尚未澄清"
    if context_required in {"上下文不足", "是-但缺失", "缺失"}:
        return "上下文不足", "既有清洗结果表明必要上下文缺失"
    return "正常回答", "候选问题已形成独立数据任务；答案质量仍需人工复核"


def command_prelabel(args: argparse.Namespace) -> int:
    cases = read_jsonl(Path(args.cases))
    selected = [item for item in cases if item.get("review_observation_id")]
    if args.limit > 0:
        selected = selected[: args.limit]
    preview = []
    for case in selected:
        status, reason = proposed_admission_status(case)
        preview.append(
            {
                "case_id": case.get("case_id"),
                "is_data_case": True,
                "admission_status": status,
                "optimized_question": case.get("resolved_input") or case.get("original_input"),
                "reason": reason,
            }
        )
    if not args.commit:
        print(json.dumps({"commit": False, "prelabels": preview}, ensure_ascii=False, indent=2))
        return 0

    api = LangfuseApi()
    configs = api.list_pages("score-configs")
    config_by_name = {str(item.get("name") or ""): item for item in configs}
    required = {"人工确认是否数据类", "数据集准入状态", "优化后问题"}
    missing = required - set(config_by_name)
    if missing:
        raise PipelineError(f"缺少 Score Config: {', '.join(sorted(missing))}")
    try:
        from langfuse import Langfuse
    except ImportError as exc:
        raise PipelineError("缺少 langfuse SDK，请执行 pip install -r requirements.txt") from exc
    client = Langfuse(
        public_key=api.public_key,
        secret_key=api.secret_key,
        host=api.host,
        environment=args.environment,
    )
    for case, label in zip(selected, preview):
        recovery_run_id = str(case.get("recovery_run_id") or "legacy")
        score_prefix = f"recovery:{recovery_run_id}:{case.get('case_id')}:prelabel"
        common = {
            "trace_id": str(case.get("review_trace_id") or ""),
            "observation_id": str(case.get("review_observation_id") or ""),
            "metadata": {
                "workflow": "dataset-recovery-ai-prelabel",
                "recovery_run_id": recovery_run_id,
                "case_id": case.get("case_id"),
                "reason": label["reason"],
            },
        }
        client.create_score(
            **common,
            name="人工确认是否数据类",
            value=1,
            data_type="BOOLEAN",
            config_id=str(config_by_name["人工确认是否数据类"].get("id")),
            score_id=f"{score_prefix}:is-data",
            comment=label["reason"],
        )
        client.create_score(
            **common,
            name="数据集准入状态",
            value=str(label["admission_status"]),
            data_type="CATEGORICAL",
            config_id=str(config_by_name["数据集准入状态"].get("id")),
            score_id=f"{score_prefix}:admission",
            comment=label["reason"],
        )
        client.create_score(
            **common,
            name="优化后问题",
            value=str(label["optimized_question"]),
            data_type="TEXT",
            config_id=str(config_by_name["优化后问题"].get("id")),
            score_id=f"{score_prefix}:question",
            comment="AI 清洗候选，等待人工确认",
        )
    client.flush()
    print(json.dumps({"commit": True, "prelabeled": len(selected)}, ensure_ascii=False, indent=2))
    return 0


def subject_key(score: dict[str, Any]) -> tuple[str, str]:
    subject = score.get("subject") if isinstance(score.get("subject"), dict) else {}
    return str(subject.get("kind") or "").upper(), str(subject.get("id") or "")


def command_export(args: argparse.Namespace) -> int:
    api = LangfuseApi()
    queue = resolve_queue(api, queue_id=args.queue_id, queue_name=args.queue_name)
    queue_id = str(queue.get("id") or "")
    items = api.list_pages(
        f"annotation-queues/{urllib.parse.quote(queue_id, safe='')}/items",
        query={"status": "COMPLETED"},
    )
    scores = api.list_pages(
        "v3/scores",
        query={
            "fields": "details,subject,annotation",
            "queueId": queue_id,
            "source": "ANNOTATION",
        },
    )
    by_subject: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for score in scores:
        if str(score.get("name") or "") in DEFAULT_SCORE_NAMES:
            by_subject[subject_key(score)][str(score.get("name"))] = score.get("value")

    case_index: dict[str, dict[str, Any]] = {}
    for case in read_jsonl(Path(args.cases)):
        key = str(case.get("review_observation_id") or case.get("anchor_observation_id") or "")
        if key:
            case_index[key] = case
    exported: list[dict[str, Any]] = []
    for item in items:
        object_type = str(item.get("objectType") or "").upper()
        object_id = str(item.get("objectId") or "")
        case = case_index.get(object_id, {})
        labels = by_subject.get((object_type, object_id), {})
        exported.append(
            {
                **case,
                "queue_id": queue_id,
                "queue_item_id": item.get("id"),
                "review_status": item.get("status"),
                "human_is_data_case": labels.get("人工确认是否数据类"),
                "human_admission_status": labels.get("数据集准入状态"),
                "human_optimized_question": labels.get("优化后问题"),
                "corrected_output": labels.get("output"),
            }
        )
    write_jsonl(Path(args.output), exported)
    print(json.dumps({"queue_id": queue_id, "completed": len(items), "exported": len(exported), "output": args.output}, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="生成带 Langfuse 来源映射的候选 Case")
    prepare.add_argument("--cleaning-csv", required=True)
    prepare.add_argument("--session-dir", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--min-confidence", type=float, default=0.72)
    prepare.set_defaults(handler=command_prepare)

    materialize = subparsers.add_parser("materialize", help="为每个 Case 创建唯一的审阅 Observation")
    materialize.add_argument("--cases", required=True)
    materialize.add_argument("--session-dir", required=True)
    materialize.add_argument("--output", required=True)
    materialize.add_argument("--environment", default="annotation-candidate")
    materialize.add_argument("--limit", type=int, default=0)
    materialize.add_argument("--include-rejected", action="store_true")
    materialize.add_argument("--commit", action="store_true")
    materialize.set_defaults(handler=command_materialize)

    enqueue = subparsers.add_parser("enqueue", help="把候选 Case 的审阅 Observation 加入队列")
    enqueue.add_argument("--cases", required=True)
    enqueue.add_argument("--queue-id", default="")
    enqueue.add_argument("--queue-name", default="数据回收-准入审核-POC")
    enqueue.add_argument("--min-confidence", type=float, default=0.72)
    enqueue.add_argument("--limit", type=int, default=0)
    enqueue.add_argument("--include-rejected", action="store_true")
    enqueue.add_argument("--commit", action="store_true")
    enqueue.set_defaults(handler=command_enqueue)

    prelabel = subparsers.add_parser("prelabel", help="写入 AI 预标注 Score，队列仍保持 Pending")
    prelabel.add_argument("--cases", required=True)
    prelabel.add_argument("--environment", default="annotation-candidate")
    prelabel.add_argument("--limit", type=int, default=0)
    prelabel.add_argument("--commit", action="store_true")
    prelabel.set_defaults(handler=command_prelabel)

    export = subparsers.add_parser("export", help="导出已完成人工标注与 Corrected Output")
    export.add_argument("--cases", required=True)
    export.add_argument("--queue-id", default="")
    export.add_argument("--queue-name", default="数据回收-准入审核-POC")
    export.add_argument("--output", required=True)
    export.set_defaults(handler=command_export)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    load_env_file(Path(args.env_file))
    try:
        return int(args.handler(args))
    except (PipelineError, FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
