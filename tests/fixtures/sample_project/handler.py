"""Request handler — calls parse_query so find_references can locate it."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from main import parse_query  # noqa: E402


def handle_request(raw_sql: str) -> dict[str, object]:
    """Handle an incoming SQL request by parsing it."""
    return parse_query(raw_sql)


def batch_parse(queries: list[str]) -> list[dict[str, object]]:
    """Parse a list of SQL queries in sequence."""
    return [parse_query(q) for q in queries]
