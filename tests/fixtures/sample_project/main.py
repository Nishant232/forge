"""SQL query parser module — fixture project for integration tests."""
from __future__ import annotations

import re
from typing import Any


def parse_query(sql: str) -> dict[str, Any]:
    """Parse a SQL query string into a structured representation.

    Extracts SELECT fields, FROM clause, and optional WHERE condition from a
    simple SQL SELECT statement. Returns a dict with 'fields', 'table', 'where'.
    """
    sql = sql.strip().rstrip(";")
    result: dict[str, Any] = {"fields": [], "table": None, "where": None}

    match = re.match(
        r"SELECT\s+(.+?)\s+FROM\s+(\w+)(?:\s+WHERE\s+(.+))?$",
        sql,
        re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Cannot parse SQL: {sql!r}")

    result["fields"] = [f.strip() for f in match.group(1).split(",")]
    result["table"] = match.group(2)
    result["where"] = match.group(3)
    return result


class SQLParser:
    """Stateful SQL parser that caches previously parsed results."""

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}

    def parse(self, query: str) -> dict[str, Any]:
        """Parse a SQL query, returning a cached result if available."""
        if query not in self._cache:
            self._cache[query] = parse_query(query)
        return self._cache[query]

    def clear_cache(self) -> None:
        self._cache.clear()
