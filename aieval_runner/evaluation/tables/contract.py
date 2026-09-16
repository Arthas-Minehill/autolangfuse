from __future__ import annotations

from datetime import date, datetime
from math import isfinite
from typing import Any, Dict, List


MAX_TABLE_COLUMNS = 256
MAX_TABLE_ROWS = 10_000
MAX_TABLE_CELLS = 1_000_000
MAX_CELL_STRING_LENGTH = 20_000
MAX_SAFE_INTEGER = 9_007_199_254_740_991


def _normalize_cell(value: Any, *, field_name: str) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return str(value) if abs(value) > MAX_SAFE_INTEGER else value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{field_name} 不能是 NaN 或 Infinity")
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, str):
        if len(value) > MAX_CELL_STRING_LENGTH:
            raise ValueError(
                f"{field_name} 字符串长度不能超过 {MAX_CELL_STRING_LENGTH}"
            )
        return value
    raise ValueError(f"{field_name} 只能是 null、布尔、数值、日期或字符串")


def normalize_table(
    value: Any,
    *,
    field_name: str,
    require_columns: bool = True,
) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} 必须是 object")
    columns = value.get("columns")
    rows = value.get("rows")
    if not isinstance(columns, list):
        raise ValueError(f"{field_name}.columns 必须是字符串数组")
    if require_columns and not columns:
        raise ValueError(f"{field_name}.columns 至少包含一列")
    if len(columns) > MAX_TABLE_COLUMNS:
        raise ValueError(
            f"{field_name}.columns 不能超过 {MAX_TABLE_COLUMNS} 列"
        )

    clean_columns: List[str] = []
    for index, column in enumerate(columns):
        if not isinstance(column, str) or not column.strip():
            raise ValueError(f"{field_name}.columns[{index}] 必须是非空字符串")
        clean_columns.append(column.strip())
    if len(set(clean_columns)) != len(clean_columns):
        raise ValueError(f"{field_name}.columns 不能包含重复列名")

    if not isinstance(rows, list):
        raise ValueError(f"{field_name}.rows 必须是二维数组")
    if len(rows) > MAX_TABLE_ROWS:
        raise ValueError(f"{field_name}.rows 不能超过 {MAX_TABLE_ROWS} 行")
    if len(clean_columns) * len(rows) > MAX_TABLE_CELLS:
        raise ValueError(
            f"{field_name} 单元格总数不能超过 {MAX_TABLE_CELLS}"
        )

    clean_rows: List[List[Any]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, list):
            raise ValueError(f"{field_name}.rows[{row_index}] 必须是数组")
        if len(row) != len(clean_columns):
            raise ValueError(
                f"{field_name}.rows[{row_index}] 列数必须等于 columns 长度"
            )
        clean_rows.append(
            [
                _normalize_cell(
                    cell,
                    field_name=f"{field_name}.rows[{row_index}][{column_index}]",
                )
                for column_index, cell in enumerate(row)
            ]
        )
    return {"columns": clean_columns, "rows": clean_rows}
