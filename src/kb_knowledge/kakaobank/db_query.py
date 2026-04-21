"""In-memory CRUD helpers for KakaoBank transactional DB tables.

This module follows the original ``banking_knowledge/db_query.py`` pattern:
tables are queried by name, records are dicts, and constraints support a small
set of suffix operators such as ``__gt`` and ``__contains``.
"""

from __future__ import annotations

import json
import operator
from collections.abc import Callable
from typing import Any

from kb_knowledge.kakaobank.data_model import KakaoBankDB

Comparison = Callable[[Any, Any], bool]


def list_tables(db: KakaoBankDB) -> list[str]:
    """List all table names available in the DB."""

    return [table_name for table_name, _ in db.iter_tables()]


def get_table_data(
    table_name: str,
    db: KakaoBankDB,
) -> dict[str, dict[str, Any]] | None:
    """Return the data mapping for a table, or None when absent."""

    table = db.get_table(table_name)
    if table is None:
        return None
    return table.data


def query_db(
    table_name: str,
    db: KakaoBankDB,
    *,
    return_ids: bool = False,
    limit: int | None = None,
    **constraints: Any,
) -> list[dict[str, Any]] | list[tuple[str, dict[str, Any]]]:
    """Query records using simple field constraints.

    Supported operators mirror tau3 banking_knowledge:
    ``eq``, ``ne``, ``gt``, ``gte``, ``lt``, ``lte``, ``contains``,
    ``startswith``, ``endswith``, ``in``, and ``nin``.
    """

    table_data = get_table_data(table_name, db)
    if table_data is None:
        return []

    results: list[dict[str, Any]] | list[tuple[str, dict[str, Any]]] = []
    for record_id, record in table_data.items():
        if _record_matches(record, constraints):
            if return_ids:
                results.append((record_id, record))
            else:
                results.append(record)
            if limit is not None and len(results) >= limit:
                break
    return results


def add_to_db(
    table_name: str,
    record_id: str,
    record: dict[str, Any],
    db: KakaoBankDB,
) -> bool:
    """Add a record to a table if the table exists and ID is unused."""

    table = db.get_table(table_name)
    if table is None or record_id in table.data:
        return False
    table.data[record_id] = record
    return True


def update_record_in_db(
    table_name: str,
    db: KakaoBankDB,
    record_id: str,
    updates: dict[str, Any],
) -> tuple[bool, dict[str, Any] | None]:
    """Update an existing record in place."""

    table = db.get_table(table_name)
    if table is None or record_id not in table.data:
        return False, None
    table.data[record_id].update(updates)
    return True, table.data[record_id]


def remove_from_db(
    table_name: str,
    db: KakaoBankDB,
    **constraints: Any,
) -> list[dict[str, Any]]:
    """Remove records matching constraints and return removed records."""

    table = db.get_table(table_name)
    if table is None:
        return []

    removed: list[dict[str, Any]] = []
    to_remove: list[str] = []
    for record_id, record in table.data.items():
        if _record_matches(record, constraints):
            removed.append(record)
            to_remove.append(record_id)

    for record_id in to_remove:
        del table.data[record_id]
    return removed


def query_database_tool(
    table_name: str,
    constraints: str = "{}",
    *,
    db: KakaoBankDB,
) -> str:
    """Tool-style query wrapper that accepts JSON constraints."""

    if table_name not in list_tables(db):
        return f"Error: Database '{table_name}' not found. Available: {list_tables(db)}"

    try:
        constraint_dict = json.loads(constraints) if constraints else {}
    except json.JSONDecodeError as exc:
        return f"Error: Invalid JSON: {exc}"

    results = query_db(table_name, db=db, return_ids=True, **constraint_dict)
    if not results:
        return f"No records found in '{table_name}'."

    lines = [f"Found {len(results)} record(s) in '{table_name}':\n"]
    for index, (record_id, record) in enumerate(results, 1):
        lines.append(f"{index}. Record ID: {record_id}")
        for field, value in record.items():
            lines.append(f"   {field}: {value}")
        lines.append("")
    return "\n".join(lines)


def _parse_constraint(key: str, value: Any) -> tuple[str, str, Any]:
    if "__" not in key:
        return key, "eq", value
    field_name, op_name = key.rsplit("__", 1)
    return field_name, op_name, value


def _record_matches(record: dict[str, Any], constraints: dict[str, Any]) -> bool:
    for key, expected in constraints.items():
        field_name, op_name, expected = _parse_constraint(key, expected)
        actual = record.get(field_name)
        compare = _get_comparison_op(op_name)
        try:
            if not compare(actual, expected):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _get_comparison_op(op_name: str) -> Comparison:
    ops: dict[str, Comparison] = {
        "eq": operator.eq,
        "ne": operator.ne,
        "gt": operator.gt,
        "gte": operator.ge,
        "lt": operator.lt,
        "lte": operator.le,
        "contains": _contains,
        "startswith": lambda actual, expected: (
            str(actual).startswith(str(expected)) if actual is not None else False
        ),
        "endswith": lambda actual, expected: (
            str(actual).endswith(str(expected)) if actual is not None else False
        ),
        "in": lambda actual, expected: actual in expected,
        "nin": lambda actual, expected: actual not in expected,
    }
    return ops.get(op_name, operator.eq)


def _contains(actual: Any, expected: Any) -> bool:
    if actual is None:
        return False
    return expected in actual
