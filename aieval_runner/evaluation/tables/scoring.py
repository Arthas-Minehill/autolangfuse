from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, DefaultDict, Dict, FrozenSet, List, Sequence, Tuple


DEFAULT_LAMBDA = 0.01
_NULL_VALUES = frozenset({"", "null", "none", "nan", "nat", "<na>", "n/a", "na"})
_DATE_RE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$")
_DATETIME_RE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}[T ]\d{1,2}:\d{2}")

Table = Dict[str, List[Any]]
Signature = FrozenSet[Tuple[str, int]]


def _normalize_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().replace("\r\n", "").replace("\r", "").replace("\n", "")
    if text.lower() in _NULL_VALUES:
        return ""
    numeric = _try_normalize_numeric(text)
    if numeric is not None:
        return numeric
    normalized_datetime = _try_normalize_datetime(text)
    if normalized_datetime is not None:
        return normalized_datetime
    return text


def _try_normalize_numeric(text: str) -> str | None:
    cleaned = text.replace(",", "") if "," in text else text
    if re.match(r"^\d{4}-\d{1,2}-\d{1,2}", cleaned):
        return None
    try:
        value = Decimal(cleaned)
        if value.is_nan() or value.is_infinite():
            return None
        return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError, OverflowError):
        return None


def _try_normalize_datetime(text: str) -> str | None:
    if not (_DATE_RE.match(text) or _DATETIME_RE.match(text)):
        return None
    normalized = text.replace("/", "-")
    if _DATE_RE.match(text):
        try:
            year, month, day = (int(part) for part in re.split(r"[-/]", text))
            return f"{year:04d}-{month:02d}-{day:02d}"
        except ValueError:
            pass
    formats = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M",
    ]
    for date_format in formats:
        try:
            parsed = datetime.strptime(normalized, date_format)
        except ValueError:
            continue
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return parsed.isoformat()
    return None


def _column_signature(rows: Sequence[Sequence[Any]], column_index: int) -> Signature:
    return frozenset(
        Counter(
            _normalize_cell(row[column_index] if column_index < len(row) else "")
            for row in rows
        ).items()
    )


def _all_signatures(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[Signature]:
    return [_column_signature(rows, index) for index in range(len(columns))]


def _is_name_pair(first: str, second: str) -> bool:
    left = first.lower().replace(" ", "_")
    right = second.lower().replace(" ", "_")
    return (
        ("first" in left and ("last" in right or "surname" in right))
        or ("given" in left and ("family" in right or "surname" in right))
    )


def _merged_name_signature(rows: Sequence[Sequence[Any]], left: int, right: int) -> Signature:
    merged: List[str] = []
    for row in rows:
        first = _normalize_cell(row[left] if left < len(row) else "")
        second = _normalize_cell(row[right] if right < len(row) else "")
        merged.append(" ".join(part for part in (first, second) if part))
    return frozenset(Counter(merged).items())


def _match_columns(
    gold_columns: Sequence[str],
    gold_rows: Sequence[Sequence[Any]],
    gold_signatures: Sequence[Signature],
    predicted_columns: Sequence[str],
    predicted_rows: Sequence[Sequence[Any]],
    predicted_signatures: Sequence[Signature],
) -> int:
    gold_matched = [False] * len(gold_columns)
    predicted_matched = [False] * len(predicted_columns)
    predicted_by_signature: DefaultDict[Signature, List[int]] = defaultdict(list)
    for predicted_index, signature in enumerate(predicted_signatures):
        predicted_by_signature[signature].append(predicted_index)

    for gold_index, signature in enumerate(gold_signatures):
        candidates = predicted_by_signature.get(signature)
        if candidates:
            predicted_index = candidates.pop()
            gold_matched[gold_index] = True
            predicted_matched[predicted_index] = True

    unmatched_gold = [index for index, matched in enumerate(gold_matched) if not matched]
    cursor = 0
    while cursor < len(unmatched_gold) - 1:
        left = unmatched_gold[cursor]
        right = unmatched_gold[cursor + 1]
        if right == left + 1 and _is_name_pair(gold_columns[left], gold_columns[right]):
            merged = _merged_name_signature(gold_rows, left, right)
            for predicted_index, signature in enumerate(predicted_signatures):
                if not predicted_matched[predicted_index] and signature == merged:
                    gold_matched[left] = True
                    gold_matched[right] = True
                    predicted_matched[predicted_index] = True
                    cursor += 2
                    break
            else:
                cursor += 1
        else:
            cursor += 1

    predicted_pair_signatures = {
        (left, left + 1): _merged_name_signature(predicted_rows, left, left + 1)
        for left in range(len(predicted_columns) - 1)
        if _is_name_pair(predicted_columns[left], predicted_columns[left + 1])
    }
    for gold_index, matched in enumerate(gold_matched):
        if matched:
            continue
        unmatched_predicted = [
            index for index, predicted_is_matched in enumerate(predicted_matched)
            if not predicted_is_matched
        ]
        for cursor in range(len(unmatched_predicted) - 1):
            left = unmatched_predicted[cursor]
            right = unmatched_predicted[cursor + 1]
            if (
                right == left + 1
                and predicted_pair_signatures.get((left, right))
                == gold_signatures[gold_index]
            ):
                gold_matched[gold_index] = True
                predicted_matched[left] = True
                predicted_matched[right] = True
                break

    return sum(gold_matched)


def score_tables(
    expected: Table,
    predicted: Table,
    *,
    lambda_penalty: float = DEFAULT_LAMBDA,
) -> Dict[str, Any]:
    gold_columns = list(expected.get("columns") or [])
    gold_rows = list(expected.get("rows") or [])
    predicted_columns = list(predicted.get("columns") or [])
    predicted_rows = list(predicted.get("rows") or [])
    if not gold_columns:
        return {
            "score": 0.0,
            "recall": 0.0,
            "penalty": 0.0,
            "matched_column_count": 0,
            "expected_column_count": 0,
            "predicted_column_count": len(predicted_columns),
            "expected_row_count": len(gold_rows),
            "predicted_row_count": len(predicted_rows),
            "row_count_match": len(gold_rows) == len(predicted_rows),
            "extra_column_count": len(predicted_columns),
        }
    if not gold_rows and not predicted_rows:
        schema_matched = gold_columns == predicted_columns
        return {
            "score": 1.0 if schema_matched else 0.0,
            "recall": 1.0 if schema_matched else 0.0,
            "penalty": 0.0,
            "matched_column_count": len(gold_columns) if schema_matched else 0,
            "expected_column_count": len(gold_columns),
            "predicted_column_count": len(predicted_columns),
            "expected_row_count": 0,
            "predicted_row_count": 0,
            "row_count_match": True,
            "extra_column_count": 0 if schema_matched else len(predicted_columns),
        }

    gold_signatures = _all_signatures(gold_columns, gold_rows)
    predicted_signatures = _all_signatures(predicted_columns, predicted_rows)
    matched = _match_columns(
        gold_columns,
        gold_rows,
        gold_signatures,
        predicted_columns,
        predicted_rows,
        predicted_signatures,
    )
    extra = max(0, len(predicted_columns) - matched)
    recall = matched / len(gold_columns)
    penalty = (
        lambda_penalty * (extra / len(predicted_columns))
        if predicted_columns
        else 0.0
    )
    return {
        "score": round(max(0.0, recall - penalty), 6),
        "recall": round(recall, 6),
        "penalty": round(penalty, 6),
        "matched_column_count": matched,
        "expected_column_count": len(gold_columns),
        "predicted_column_count": len(predicted_columns),
        "expected_row_count": len(gold_rows),
        "predicted_row_count": len(predicted_rows),
        "row_count_match": len(gold_rows) == len(predicted_rows),
        "extra_column_count": extra,
    }
