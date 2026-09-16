"""
评估脚本：基于 KDDCup 2026 官方评分规则。

核心逻辑（Column-Level Content Matching）：
  1. 读取 prediction.csv 和 gold.csv
  2. 对每列的所有单元格值进行标准化（Null/Numeric/Date/DateTime/String）
  3. 对每列排序后生成 "column signature"
  4. 基于 column signature 匹配 prediction 对 gold 的覆盖
  5. Score = Recall - λ * (Extra Columns / Predicted Columns)
  6. 忽略列名、行顺序，只看列内容

标准化规则：
  - Null: "", "null", "none", "nan", "nat", "<na>" → ""
  - Numeric: Decimal ROUND_HALF_UP 到 2 位小数
  - Date: ISO 8601 YYYY-MM-DD
  - DateTime: 有时区→UTC(Z)；无时区→保持原格式
  - String: 去除首尾空白和 \\r\\n，大小写敏感

用法：
    uv run python scripts/evaluate.py --run_id <run_id>
    uv run python scripts/evaluate.py --run_id <run_id> --lambda_penalty 0.1
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from pathlib import Path

# ---------------------------------------------------------------------------
# 项目路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"
DEFAULT_GOLD_DIR = PROJECT_ROOT / "data" / "public" / "output"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "eval"

# 默认惩罚系数 λ
# 官方规则: Score = Recall - λ × (Extra Columns / Predicted Columns)
# λ=0.5 基于官方评分示例校准:
#   - 1 extra / 3 pred → 0.833 ("slightly below perfect" ✓)
#   - 8 extra / 10 pred → 0.600 ("significantly lower" ✓)
#   - gold=1, pred=5 → 0.600 (较低 ✓)
#   - gold=1, pred=10 → 0.550 (很低 ✓)
DEFAULT_LAMBDA = 0.01


# ---------------------------------------------------------------------------
# 值标准化
# ---------------------------------------------------------------------------

_NULL_VALUES = frozenset({"", "null", "none", "nan", "nat", "<na>", "n/a", "na"})

_DATE_RE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$")
_DATETIME_RE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}[T ]\d{1,2}:\d{2}")


def _normalize_cell(val: str) -> str:
    """按官方规则标准化单元格值。优先级: Null → Numeric → Date/DateTime → String"""
    s = val.strip().replace("\r\n", "").replace("\r", "").replace("\n", "")
    if s.lower() in _NULL_VALUES:
        return ""
    numeric = _try_normalize_numeric(s)
    if numeric is not None:
        return numeric
    dt = _try_normalize_datetime(s)
    if dt is not None:
        return dt
    return s


def _try_normalize_numeric(s: str) -> str | None:
    """尝试解析为数值，Decimal ROUND_HALF_UP 保留2位小数。"""
    cleaned = s.replace(",", "") if "," in s else s
    if re.match(r"^\d{4}-\d{1,2}-\d{1,2}", cleaned):
        return None
    try:
        d = Decimal(cleaned)
        if d.is_nan() or d.is_infinite():
            return None
        rounded = d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return str(rounded)
    except (InvalidOperation, ValueError, OverflowError):
        return None


def _try_normalize_datetime(s: str) -> str | None:
    """尝试解析为日期/日期时间。"""
    if not (_DATE_RE.match(s) or _DATETIME_RE.match(s)):
        return None
    normalized = s.replace("/", "-")
    if _DATE_RE.match(s):
        try:
            parts = re.split(r"[-/]", s)
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            return f"{year:04d}-{month:02d}-{day:02d}"
        except (ValueError, IndexError):
            pass
    dt_formats = [
        "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d %H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M",
    ]
    for fmt in dt_formats:
        try:
            dt = datetime.strptime(normalized, fmt)
            if dt.tzinfo is not None:
                dt_utc = dt.astimezone(timezone.utc)
                return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                return dt.isoformat()
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# CSV 读取
# ---------------------------------------------------------------------------

def _read_csv_as_table(csv_path: Path) -> tuple[list[str], list[list[str]]]:
    """读取 CSV，返回 (columns, rows)。跳过尾部空行。"""
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows_raw = list(reader)
    if not rows_raw:
        return [], []
    columns = [c.strip() for c in rows_raw[0]]
    rows: list[list[str]] = []
    for row in rows_raw[1:]:
        if all(cell.strip() == "" for cell in row):
            continue
        padded = row + [""] * (len(columns) - len(row)) if len(row) < len(columns) else row
        rows.append([cell for cell in padded[:len(columns)]])
    return columns, rows


# ---------------------------------------------------------------------------
# Column Signature
# ---------------------------------------------------------------------------

def _build_column_signature(rows: list[list[str]], col_index: int) -> tuple[str, ...]:
    """提取第 col_index 列的所有值，标准化后排序，返回 tuple 作为签名。"""
    values = []
    for row in rows:
        if col_index < len(row):
            values.append(_normalize_cell(row[col_index]))
        else:
            values.append("")
    return tuple(sorted(values))


def _build_all_signatures(columns: list[str], rows: list[list[str]]) -> list[tuple[str, ...]]:
    """为所有列构建签名列表。"""
    return [_build_column_signature(rows, i) for i in range(len(columns))]


# ---------------------------------------------------------------------------
# Column Matching (with name field support)
# ---------------------------------------------------------------------------

def _match_columns(
    gold_cols: list[str], gold_rows: list[list[str]], gold_sigs: list[tuple[str, ...]],
    pred_cols: list[str], pred_rows: list[list[str]], pred_sigs: list[tuple[str, ...]],
) -> int:
    """
    Column signature matching with name field support.
    Returns matched count (最多 = len(gold_cols)).
    """
    gold_col_count = len(gold_cols)
    pred_col_count = len(pred_cols)

    # Step 1: 直接签名匹配
    pred_sig_counter = Counter(pred_sigs)
    gold_matched = [False] * gold_col_count
    pred_matched = [False] * pred_col_count

    for gi in range(gold_col_count):
        gsig = gold_sigs[gi]
        if pred_sig_counter.get(gsig, 0) > 0:
            pred_sig_counter[gsig] -= 1
            gold_matched[gi] = True
            for pi in range(pred_col_count):
                if not pred_matched[pi] and pred_sigs[pi] == gsig:
                    pred_matched[pi] = True
                    break

    # Step 2: 对未匹配的 gold 列，尝试 name 合并 (gold first+last → pred full name)
    unmatched_gold = [i for i in range(gold_col_count) if not gold_matched[i]]
    if len(unmatched_gold) >= 2:
        i = 0
        while i < len(unmatched_gold) - 1:
            gi1 = unmatched_gold[i]
            gi2 = unmatched_gold[i + 1]
            if gi2 == gi1 + 1 and _is_name_pair(gold_cols[gi1], gold_cols[gi2]):
                merged_sig = _merge_name_signature(gold_rows, gi1, gi2)
                for pi in range(pred_col_count):
                    if not pred_matched[pi] and pred_sigs[pi] == merged_sig:
                        gold_matched[gi1] = True
                        gold_matched[gi2] = True
                        pred_matched[pi] = True
                        i += 2
                        break
                else:
                    i += 1
                continue
            i += 1

    # Step 3: 反向 — pred 有 first+last，gold 有 full name
    unmatched_gold = [i for i in range(gold_col_count) if not gold_matched[i]]
    unmatched_pred = [i for i in range(pred_col_count) if not pred_matched[i]]
    if unmatched_gold and len(unmatched_pred) >= 2:
        for gi in unmatched_gold:
            for j in range(len(unmatched_pred) - 1):
                pi1 = unmatched_pred[j]
                pi2 = unmatched_pred[j + 1]
                if pi2 != pi1 + 1:
                    continue
                if not _is_name_pair(pred_cols[pi1], pred_cols[pi2]):
                    continue
                merged_sig = _merge_name_signature(pred_rows, pi1, pi2)
                if merged_sig == gold_sigs[gi]:
                    gold_matched[gi] = True
                    pred_matched[pi1] = True
                    pred_matched[pi2] = True
                    break

    return sum(gold_matched)


def _is_name_pair(col1: str, col2: str) -> bool:
    """检测两个列名是否构成 first_name/last_name 对。"""
    c1 = col1.lower().replace(" ", "_")
    c2 = col2.lower().replace(" ", "_")
    return (
        ("first" in c1 and ("last" in c2 or "surname" in c2))
        or ("given" in c1 and ("family" in c2 or "surname" in c2))
    )


def _merge_name_signature(rows: list[list[str]], ci1: int, ci2: int) -> tuple[str, ...]:
    """合并两列为 "first last" 格式的签名。"""
    merged = []
    for row in rows:
        v1 = _normalize_cell(row[ci1] if ci1 < len(row) else "")
        v2 = _normalize_cell(row[ci2] if ci2 < len(row) else "")
        if v1 and v2:
            merged.append(f"{v1} {v2}")
        elif v1:
            merged.append(v1)
        elif v2:
            merged.append(v2)
        else:
            merged.append("")
    return tuple(sorted(merged))


# ---------------------------------------------------------------------------
# 核心评分
# ---------------------------------------------------------------------------

def score_single_task(
    task_id: str,
    gold_path: Path,
    pred_path: Path | None,
    lambda_penalty: float,
) -> dict:
    """基于官方规则评分单个 task。"""
    result: dict = {
        "task_id": task_id,
        "has_gold": gold_path.exists(),
        "has_prediction": pred_path is not None and pred_path.exists(),
        "score": 0.0,
        "recall": 0.0,
        "penalty": 0.0,
        "gold_col_count": 0,
        "pred_col_count": 0,
        "matched_col_count": 0,
        "extra_col_count": 0,
        "gold_row_count": 0,
        "pred_row_count": 0,
        "row_count_match": False,
        "gold_columns": "",
        "pred_columns": "",
        "detail": "",
    }

    if not result["has_gold"]:
        result["detail"] = "标准答案文件不存在"
        return result

    gold_cols, gold_rows = _read_csv_as_table(gold_path)
    result["gold_col_count"] = len(gold_cols)
    result["gold_row_count"] = len(gold_rows)
    result["gold_columns"] = "|".join(gold_cols)

    if not result["has_prediction"]:
        result["detail"] = "无预测文件，得分 0"
        return result

    pred_cols, pred_rows = _read_csv_as_table(pred_path)
    result["pred_col_count"] = len(pred_cols)
    result["pred_row_count"] = len(pred_rows)
    result["pred_columns"] = "|".join(pred_cols)
    result["row_count_match"] = (len(gold_rows) == len(pred_rows))

    # 构建 Column Signatures
    gold_sigs = _build_all_signatures(gold_cols, gold_rows)
    pred_sigs = _build_all_signatures(pred_cols, pred_rows)

    # Column Matching
    matched = _match_columns(gold_cols, gold_rows, gold_sigs,
                              pred_cols, pred_rows, pred_sigs)

    gold_col_count = len(gold_cols)
    pred_col_count = len(pred_cols)
    extra_col_count = max(0, pred_col_count - matched)

    result["matched_col_count"] = matched
    result["extra_col_count"] = extra_col_count

    # 计算分数
    recall = matched / gold_col_count if gold_col_count > 0 else 1.0
    penalty = lambda_penalty * (extra_col_count / pred_col_count) if pred_col_count > 0 else 0.0
    score = max(0.0, recall - penalty)

    result["recall"] = round(recall, 6)
    result["penalty"] = round(penalty, 6)
    result["score"] = round(score, 6)

    # Detail
    details = []
    if not result["row_count_match"]:
        details.append(f"行数不匹配: gold={len(gold_rows)} pred={len(pred_rows)}")
    if matched < gold_col_count:
        details.append(f"列匹配: {matched}/{gold_col_count}")
    if extra_col_count > 0:
        details.append(f"多余列: {extra_col_count} (pred={pred_col_count}列 gold={gold_col_count}列)")
    if score == 1.0:
        details.append("完美匹配")
    elif matched == gold_col_count and extra_col_count > 0:
        details.append(f"全部gold列匹配，{extra_col_count}个多余列导致惩罚")
    result["detail"] = "; ".join(details) if details else "完美匹配"

    return result


# ---------------------------------------------------------------------------
# 主评估函数
# ---------------------------------------------------------------------------

def evaluate(
    run_id: str,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    gold_dir: Path = DEFAULT_GOLD_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    lambda_penalty: float = DEFAULT_LAMBDA,
) -> Path:
    run_dir = runs_dir / str(run_id)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"运行结果目录不存在: {run_dir}")
    if not gold_dir.is_dir():
        raise FileNotFoundError(f"标准答案目录不存在: {gold_dir}")

    gold_task_ids = sorted(
        [d.name for d in gold_dir.iterdir() if d.is_dir() and d.name.startswith("task_")],
        key=lambda x: int(x.split("_")[1]),
    )

    results: list[dict] = []
    for task_id in gold_task_ids:
        gold_path = gold_dir / task_id / "gold.csv"
        pred_path = run_dir / task_id / "prediction.csv"
        if not pred_path.exists():
            pred_path = None
        result = score_single_task(task_id, gold_path, pred_path, lambda_penalty)
        results.append(result)

    # 统计
    total = len(results)
    has_prediction = sum(1 for r in results if r["has_prediction"])
    no_prediction = total - has_prediction
    scores = [r["score"] for r in results]
    total_score = sum(scores) / total if total > 0 else 0.0
    perfect_count = sum(1 for r in results if r["score"] == 1.0)
    partial_count = sum(1 for r in results if 0 < r["score"] < 1.0)
    zero_count = sum(1 for r in results if r["score"] == 0.0)

    # 输出
    eval_output_dir = output_dir / str(run_id)
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # 详细 CSV
    detail_csv_path = eval_output_dir / "evaluation_detail.csv"
    detail_headers = [
        "任务ID", "得分", "Recall", "惩罚", "行数匹配",
        "Gold列数", "Pred列数", "匹配列数", "多余列数",
        "Gold行数", "Pred行数", "Gold列名", "Pred列名", "详情",
    ]
    detail_keys = [
        "task_id", "score", "recall", "penalty", "row_count_match",
        "gold_col_count", "pred_col_count", "matched_col_count", "extra_col_count",
        "gold_row_count", "pred_row_count", "gold_columns", "pred_columns", "detail",
    ]
    with detail_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(detail_headers)
        for r in results:
            writer.writerow([r.get(k, "") for k in detail_keys])

    # 摘要 JSON
    summary = {
        "运行ID": str(run_id),
        "评估时间": datetime.now(timezone.utc).isoformat(),
        "惩罚系数λ": lambda_penalty,
        "总任务数": total,
        "有预测结果": has_prediction,
        "无预测结果": no_prediction,
        "总分(平均分)": round(total_score, 6),
        "满分任务数(1.0)": perfect_count,
        "部分得分任务数(0<s<1)": partial_count,
        "零分任务数(0.0)": zero_count,
        "各任务得分": {r["task_id"]: r["score"] for r in results},
    }
    summary_json_path = eval_output_dir / "evaluation_summary.json"
    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 摘要 CSV
    summary_csv_path = eval_output_dir / "evaluation_summary.csv"
    summary_flat = {
        "运行ID": str(run_id),
        "惩罚系数λ": lambda_penalty,
        "总任务数": total,
        "有预测结果": has_prediction,
        "总分(平均分)": round(total_score, 6),
        "满分任务数": perfect_count,
        "部分得分任务数": partial_count,
        "零分任务数": zero_count,
    }
    with summary_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_flat.keys()))
        writer.writeheader()
        writer.writerow(summary_flat)

    # Markdown 报告
    report_md_path = eval_output_dir / "evaluation_report.md"
    _write_report_md(run_id, results, summary, lambda_penalty, report_md_path)

    # 终端打印
    _print_report(run_id, results, summary, lambda_penalty,
                  detail_csv_path, summary_json_path, summary_csv_path, report_md_path)

    return eval_output_dir


def _write_report_md(
    run_id: str,
    results: list[dict],
    summary: dict,
    lambda_penalty: float,
    output_path: Path,
) -> None:
    """生成直观易读的 Markdown 评估报告。"""
    total = summary["总任务数"]
    has_prediction = summary["有预测结果"]
    no_prediction = summary["无预测结果"]
    total_score = summary["总分(平均分)"]
    perfect_count = summary["满分任务数(1.0)"]
    partial_count = summary["部分得分任务数(0<s<1)"]
    zero_count = summary["零分任务数(0.0)"]

    lines: list[str] = []
    w = lines.append

    w(f"# KDDCup 2026 官方规则评估结果 — Run {run_id}\n")
    w(f"## 总分: **{total_score:.4f}** (平均分，λ={lambda_penalty})\n")

    w("| 类别 | 数量 |")
    w("|------|------|")
    w(f"| 总任务数 | {total} |")
    w(f"| 有预测结果 | {has_prediction} ({has_prediction*100//total if total else 0}%) |")
    w(f"| 无预测结果 | {no_prediction} |")
    w(f"| 满分任务 (1.0) | **{perfect_count}** |")
    w(f"| 部分得分 (0<s<1) | **{partial_count}** |")
    w(f"| 零分任务 (0.0) | **{zero_count}** |")
    w("")

    # ── 满分任务 ──
    perfect_tasks = [r for r in results if r["score"] == 1.0]
    if perfect_tasks:
        w("---\n")
        w(f"## ✅ 满分任务 ({len(perfect_tasks)} 个)\n")
        names = ", ".join(r["task_id"] for r in perfect_tasks)
        w(f"{names}\n")

    # ── 部分得分任务 ──
    partial_tasks = sorted(
        [r for r in results if 0 < r["score"] < 1.0],
        key=lambda r: r["score"], reverse=True,
    )
    if partial_tasks:
        w("---\n")
        w(f"## ⚠️ 部分得分任务 ({len(partial_tasks)} 个)\n")
        w("| 任务 | 得分 | Recall | 惩罚 | 匹配列 | 多余列 | Gold列数 | Pred列数 | Gold行数 | Pred行数 | 说明 |")
        w("|------|------|--------|------|--------|--------|----------|----------|----------|----------|------|")
        for r in partial_tasks:
            detail = r["detail"].replace("|", "\\|")
            w(f"| **{r['task_id']}** | {r['score']:.4f} | {r['recall']:.4f} | {r['penalty']:.4f} "
              f"| {r['matched_col_count']}/{r['gold_col_count']} | **{r['extra_col_count']}** "
              f"| {r['gold_col_count']} | {r['pred_col_count']} "
              f"| {r['gold_row_count']} | {r['pred_row_count']} "
              f"| {detail} |")
        w("")

    # ── 零分任务 ──
    zero_tasks = [r for r in results if r["score"] == 0.0]
    if zero_tasks:
        w("---\n")
        w(f"## ❌ 零分任务 ({len(zero_tasks)} 个)\n")
        w("| 任务 | 匹配列 | 多余列 | Gold列数 | Pred列数 | Gold行数 | Pred行数 | 行数匹配 | Gold列名 | Pred列名 | 问题诊断 |")
        w("|------|--------|--------|----------|----------|----------|----------|----------|----------|----------|---------|")
        for r in zero_tasks:
            if not r["has_prediction"]:
                w(f"| **{r['task_id']}** | — | — | {r['gold_col_count']} | 0 | {r['gold_row_count']} | 0 | — | {r['gold_columns']} | — | 无预测文件 |")
            else:
                row_match = "✅" if r["row_count_match"] else "❌"
                detail = r["detail"].replace("|", "\\|")
                gold_cols = r["gold_columns"].replace("|", ", ")
                pred_cols = r["pred_columns"].replace("|", ", ")
                w(f"| **{r['task_id']}** | {r['matched_col_count']}/{r['gold_col_count']} "
                  f"| {r['extra_col_count']} | {r['gold_col_count']} | {r['pred_col_count']} "
                  f"| {r['gold_row_count']} | {r['pred_row_count']} | {row_match} "
                  f"| {gold_cols} | {pred_cols} | {detail} |")
        w("")

    # ── 评分公式 ──
    w("---\n")
    w("## 评分公式\n")
    w("```")
    w("Score = Recall - λ × (Extra Columns / Predicted Columns)")
    w("Recall = Matched Columns / Gold Columns")
    w(f"λ = {lambda_penalty}")
    w("总分 = 所有任务 Score 的平均值")
    w("```\n")

    w("### 标准化规则\n")
    w("| 类型 | 规则 |")
    w("|------|------|")
    w('| Null | "", "null", "none", "nan", "nat", "<na>" → "" |')
    w("| 数值 | Decimal ROUND_HALF_UP 保留2位小数 |")
    w("| 日期 | ISO 8601 YYYY-MM-DD |")
    w("| 日期时间 | 有时区→UTC(Z)；无时区→原格式 |")
    w("| 字符串 | strip空白，大小写敏感 |")
    w("| 姓名 | first+last两列 或 合并为一列均可 |")
    w("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_report(
    run_id: str,
    results: list[dict],
    summary: dict,
    lambda_penalty: float,
    detail_csv_path: Path,
    summary_json_path: Path,
    summary_csv_path: Path,
    report_md_path: Path | None = None,
) -> None:
    total = summary["总任务数"]
    has_prediction = summary["有预测结果"]
    no_prediction = summary["无预测结果"]
    total_score = summary["总分(平均分)"]
    perfect_count = summary["满分任务数(1.0)"]
    partial_count = summary["部分得分任务数(0<s<1)"]
    zero_count = summary["零分任务数(0.0)"]

    print()
    print("=" * 80)
    print(f"  KDDCup 2026 评估报告  run_id = {run_id}  (λ = {lambda_penalty})")
    print("=" * 80)
    print(f"  总任务数:          {total}")
    print(f"  有预测结果:        {has_prediction}")
    print(f"  无预测结果:        {no_prediction}")
    print("-" * 80)
    print(f"  ★ 总分 (平均分):   {total_score:.4f}")
    print(f"  满分任务 (1.0):    {perfect_count}")
    print(f"  部分得分 (0<s<1):  {partial_count}")
    print(f"  零分任务 (0.0):    {zero_count}")
    print("-" * 80)

    perfect_tasks = [r for r in results if r["score"] == 1.0]
    if perfect_tasks:
        print(f"\n  ✅ 满分任务 ({len(perfect_tasks)} 个):")
        for r in perfect_tasks:
            print(f"    {r['task_id']:12s}  score=1.0  "
                  f"cols: gold={r['gold_col_count']} pred={r['pred_col_count']}  "
                  f"rows: gold={r['gold_row_count']} pred={r['pred_row_count']}")

    partial_tasks = sorted(
        [r for r in results if 0 < r["score"] < 1.0],
        key=lambda r: r["score"], reverse=True,
    )
    if partial_tasks:
        print(f"\n  ⚠️ 部分得分任务 ({len(partial_tasks)} 个):")
        for r in partial_tasks:
            print(f"    {r['task_id']:12s}  score={r['score']:.4f}  "
                  f"recall={r['recall']:.4f}  penalty={r['penalty']:.4f}  "
                  f"matched={r['matched_col_count']}/{r['gold_col_count']}  "
                  f"extra={r['extra_col_count']}  "
                  f"rows: gold={r['gold_row_count']} pred={r['pred_row_count']}")
            if r["detail"]:
                print(f"      {r['detail']}")

    zero_tasks = [r for r in results if r["score"] == 0.0]
    if zero_tasks:
        print(f"\n  ❌ 零分任务 ({len(zero_tasks)} 个):")
        for r in zero_tasks:
            if not r["has_prediction"]:
                print(f"    {r['task_id']:12s}  无预测文件")
            else:
                print(f"    {r['task_id']:12s}  score=0.0  "
                      f"matched={r['matched_col_count']}/{r['gold_col_count']}  "
                      f"extra={r['extra_col_count']}  "
                      f"rows: gold={r['gold_row_count']} pred={r['pred_row_count']}")
                if r["detail"]:
                    print(f"      {r['detail']}")

    print()
    print("=" * 80)
    print(f"  详细结果 CSV: {detail_csv_path}")
    print(f"  摘要 JSON:    {summary_json_path}")
    print(f"  摘要 CSV:     {summary_csv_path}")
    if report_md_path:
        print(f"  📊 评估报告:  {report_md_path}")
    print("=" * 80)
    print()


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KDDCup 2026 官方评分规则评估：Column-Level Content Matching",
    )
    parser.add_argument("--run_id", required=True, help="运行 ID")
    parser.add_argument("--runs_dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--gold_dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lambda_penalty", type=float, default=DEFAULT_LAMBDA,
                        help=f"多余列惩罚系数 λ (默认: {DEFAULT_LAMBDA})")
    args = parser.parse_args()

    try:
        evaluate(
            run_id=args.run_id,
            runs_dir=args.runs_dir,
            gold_dir=args.gold_dir,
            output_dir=args.output_dir,
            lambda_penalty=args.lambda_penalty,
        )
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

