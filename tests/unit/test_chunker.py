from __future__ import annotations

from pathlib import Path

import pytest

from agent.code_index.chunker import Chunk, chunk_file

_PYTHON_WITH_SYMBOLS = """\
def hello(name: str) -> str:
    return f"Hello, {name}"


class Greeter:
    def greet(self, name: str) -> str:
        return hello(name)

    def farewell(self, name: str) -> str:
        return f"Bye, {name}"
"""

_PYTHON_NO_SYMBOLS = "X = 1\nY = 2\nZ = X + Y\n"


class TestChunkFile:
    def test_returns_chunks_for_python(self, tmp_path: Path) -> None:
        f = tmp_path / "greet.py"
        f.write_text(_PYTHON_WITH_SYMBOLS, encoding="utf-8")
        chunks = chunk_file(f)
        assert len(chunks) >= 2
        symbols = {c.symbol for c in chunks}
        assert "hello" in symbols

    def test_returns_empty_for_unsupported_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "notes.md"
        f.write_text("# Title\nsome text\n", encoding="utf-8")
        assert chunk_file(f) == []

    def test_returns_empty_for_missing_file(self, tmp_path: Path) -> None:
        # parse_file returns None → chunk_file returns []
        f = tmp_path / "ghost.py"
        assert chunk_file(f) == []

    def test_module_fallback_when_no_structural_nodes(self, tmp_path: Path) -> None:
        f = tmp_path / "constants.py"
        f.write_text(_PYTHON_NO_SYMBOLS, encoding="utf-8")
        chunks = chunk_file(f)
        assert len(chunks) == 1
        assert chunks[0].kind == "module"
        assert chunks[0].symbol is None

    def test_line_numbers_are_one_indexed(self, tmp_path: Path) -> None:
        f = tmp_path / "fn.py"
        f.write_text("def foo():\n    pass\n", encoding="utf-8")
        chunks = chunk_file(f)
        assert all(c.start_line >= 1 for c in chunks)
        assert all(c.end_line >= c.start_line for c in chunks)

    def test_content_hash_is_stable_across_calls(self, tmp_path: Path) -> None:
        f = tmp_path / "stable.py"
        f.write_text("def bar():\n    return 42\n", encoding="utf-8")
        h1 = chunk_file(f)[0].content_hash
        h2 = chunk_file(f)[0].content_hash
        assert h1 == h2

    def test_content_hash_changes_when_file_edited(self, tmp_path: Path) -> None:
        f = tmp_path / "change.py"
        f.write_text("def baz():\n    return 1\n", encoding="utf-8")
        h1 = chunk_file(f)[0].content_hash
        f.write_text("def baz():\n    return 2\n", encoding="utf-8")
        h2 = chunk_file(f)[0].content_hash
        assert h1 != h2

    def test_class_chunk_has_correct_kind(self, tmp_path: Path) -> None:
        f = tmp_path / "cls.py"
        f.write_text("class Foo:\n    pass\n", encoding="utf-8")
        chunks = chunk_file(f)
        kinds = {c.kind for c in chunks}
        assert "class" in kinds

    def test_function_inside_class_is_method(self, tmp_path: Path) -> None:
        f = tmp_path / "method.py"
        f.write_text(
            "class Bar:\n    def do_thing(self) -> None:\n        pass\n",
            encoding="utf-8",
        )
        chunks = chunk_file(f)
        method_chunks = [c for c in chunks if c.symbol == "do_thing"]
        assert method_chunks, "Expected a chunk named 'do_thing'"
        assert method_chunks[0].kind == "method"

    def test_chunk_language_matches_extension(self, tmp_path: Path) -> None:
        f = tmp_path / "prog.go"
        f.write_text(
            "package main\n\nfunc Hello() string {\n\treturn \"hi\"\n}\n",
            encoding="utf-8",
        )
        chunks = chunk_file(f)
        assert all(c.language == "go" for c in chunks)

    def test_typescript_file_produces_chunks(self, tmp_path: Path) -> None:
        f = tmp_path / "app.ts"
        f.write_text(
            "function greet(name: string): string {\n  return `Hello ${name}`;\n}\n",
            encoding="utf-8",
        )
        chunks = chunk_file(f)
        assert len(chunks) >= 1
        assert any(c.symbol == "greet" for c in chunks)

    def test_rust_file_produces_chunks(self, tmp_path: Path) -> None:
        f = tmp_path / "lib.rs"
        f.write_text(
            "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n",
            encoding="utf-8",
        )
        chunks = chunk_file(f)
        assert len(chunks) >= 1
        assert any(c.symbol == "add" for c in chunks)
