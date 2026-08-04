from __future__ import annotations

from pathlib import Path

import pytest

from agent.code_index.chunker import Chunk
from agent.code_index.symbols import RefRecord, SymbolStore


@pytest.fixture
def store(tmp_path: Path) -> SymbolStore:
    return SymbolStore(tmp_path / "state.db")


def _chunk(
    symbol: str,
    kind: str = "function",
    path: str = "src/foo.py",
    start: int = 1,
    end: int = 5,
) -> Chunk:
    return Chunk(
        path=path,
        start_line=start,
        end_line=end,
        symbol=symbol,
        kind=kind,  # type: ignore[arg-type]
        content=f"def {symbol}(): pass",
        language="python",
        content_hash="abc123",
    )


class TestUpsertAndLookup:
    def test_inserted_symbol_is_findable(self, store: SymbolStore) -> None:
        store.upsert_symbols([_chunk("parse_query")])
        results = store.lookup("parse_query")
        assert len(results) == 1
        assert results[0].name == "parse_query"
        assert results[0].path == "src/foo.py"

    def test_lookup_unknown_returns_empty(self, store: SymbolStore) -> None:
        assert store.lookup("nonexistent_func") == []

    def test_upsert_is_idempotent(self, store: SymbolStore) -> None:
        c = _chunk("my_func")
        store.upsert_symbols([c])
        store.upsert_symbols([c])
        assert len(store.lookup("my_func")) == 1

    def test_multiple_symbols_stored_and_retrieved(self, store: SymbolStore) -> None:
        store.upsert_symbols([_chunk("alpha"), _chunk("beta"), _chunk("gamma")])
        assert len(store.lookup("alpha")) == 1
        assert len(store.lookup("beta")) == 1
        assert len(store.lookup("gamma")) == 1

    def test_anonymous_chunks_skipped(self, store: SymbolStore) -> None:
        anon = Chunk(
            path="a.py",
            start_line=1,
            end_line=10,
            symbol=None,
            kind="module",
            content="X = 1",
            language="python",
            content_hash="xyz",
        )
        store.upsert_symbols([anon])
        assert store.all_symbol_names() == set()

    def test_lookup_returns_correct_line_numbers(self, store: SymbolStore) -> None:
        store.upsert_symbols([_chunk("fn_at_line", start=42, end=55)])
        results = store.lookup("fn_at_line")
        assert results[0].start_line == 42
        assert results[0].end_line == 55

    def test_same_name_different_files(self, store: SymbolStore) -> None:
        store.upsert_symbols([
            _chunk("shared", path="a.py"),
            _chunk("shared", path="b.py"),
        ])
        results = store.lookup("shared")
        assert len(results) == 2
        paths = {r.path for r in results}
        assert paths == {"a.py", "b.py"}


class TestDeleteFile:
    def test_delete_removes_its_symbols(self, store: SymbolStore) -> None:
        store.upsert_symbols([_chunk("doomed", path="victim.py")])
        store.delete_file("victim.py")
        assert store.lookup("doomed") == []

    def test_delete_does_not_affect_other_files(self, store: SymbolStore) -> None:
        store.upsert_symbols([_chunk("a", path="a.py"), _chunk("b", path="b.py")])
        store.delete_file("a.py")
        assert len(store.lookup("b")) == 1

    def test_delete_nonexistent_file_is_noop(self, store: SymbolStore) -> None:
        store.delete_file("does_not_exist.py")  # should not raise


class TestAllSymbolNames:
    def test_returns_all_distinct_names(self, store: SymbolStore) -> None:
        store.upsert_symbols([_chunk("foo"), _chunk("bar"), _chunk("foo", path="b.py")])
        names = store.all_symbol_names()
        assert names == {"foo", "bar"}

    def test_empty_when_no_symbols(self, store: SymbolStore) -> None:
        assert store.all_symbol_names() == set()


class TestRefs:
    def test_upsert_and_find(self, store: SymbolStore) -> None:
        ref = RefRecord(
            target_name="parse_query",
            source_path="handler.py",
            source_line=10,
            line_content="return parse_query(sql)",
        )
        store.upsert_refs([ref])
        results = store.find_refs("parse_query")
        assert len(results) == 1
        assert results[0].source_path == "handler.py"
        assert results[0].source_line == 10
        assert results[0].line_content == "return parse_query(sql)"

    def test_find_refs_unknown_returns_empty(self, store: SymbolStore) -> None:
        assert store.find_refs("nobody") == []

    def test_upsert_refs_idempotent(self, store: SymbolStore) -> None:
        ref = RefRecord("func", "src.py", 5, "func()")
        store.upsert_refs([ref])
        store.upsert_refs([ref])
        assert len(store.find_refs("func")) == 1

    def test_multiple_callers(self, store: SymbolStore) -> None:
        store.upsert_refs([
            RefRecord("target", "a.py", 1, "target()"),
            RefRecord("target", "b.py", 7, "x = target()"),
        ])
        results = store.find_refs("target")
        assert len(results) == 2
        paths = {r.source_path for r in results}
        assert paths == {"a.py", "b.py"}

    def test_delete_file_removes_its_refs(self, store: SymbolStore) -> None:
        store.upsert_refs([RefRecord("sym", "caller.py", 3, "sym()")])
        store.delete_file("caller.py")
        assert store.find_refs("sym") == []
