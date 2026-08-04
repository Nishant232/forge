"""Tests for the SQL parser module — part of the fixture project."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from main import SQLParser, parse_query


def test_parse_simple_select() -> None:
    result = parse_query("SELECT id, name FROM users")
    assert result["fields"] == ["id", "name"]
    assert result["table"] == "users"
    assert result["where"] is None


def test_parse_with_where_clause() -> None:
    result = parse_query("SELECT * FROM orders WHERE status = 'active'")
    assert result["table"] == "orders"
    assert result["where"] is not None


def test_parse_trailing_semicolon() -> None:
    result = parse_query("SELECT id FROM users;")
    assert result["table"] == "users"


def test_parse_invalid_raises_value_error() -> None:
    with pytest.raises(ValueError):
        parse_query("NOT A VALID QUERY")


def test_sql_parser_caches_results() -> None:
    parser = SQLParser()
    q = "SELECT id FROM users"
    r1 = parser.parse(q)
    r2 = parser.parse(q)
    assert r1 is r2


def test_sql_parser_clear_cache() -> None:
    parser = SQLParser()
    q = "SELECT name FROM products"
    parser.parse(q)
    parser.clear_cache()
    r = parser.parse(q)
    assert r["table"] == "products"
