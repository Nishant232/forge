from __future__ import annotations

import hashlib
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.code_index.chunker import Chunk
    from agent.code_index.parser import RefNode

_LOG = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS symbols (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    path        TEXT NOT NULL,
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL,
    language    TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_path ON symbols(path);

CREATE TABLE IF NOT EXISTS symbol_refs (
    id           TEXT PRIMARY KEY,
    target_name  TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    source_line  INTEGER NOT NULL,
    line_content TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_refs_target ON symbol_refs(target_name);
CREATE INDEX IF NOT EXISTS idx_refs_source ON symbol_refs(source_path);
"""


@dataclass
class SymbolRecord:
    id: str
    name: str
    kind: str
    path: str
    start_line: int
    end_line: int
    language: str
    content_hash: str
    updated_at: str


@dataclass
class RefRecord:
    target_name: str
    source_path: str
    source_line: int
    line_content: str = ""


def _symbol_id(path: str, name: str | None, kind: str, start_line: int) -> str:
    key = f"{path}:{name}:{kind}:{start_line}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _ref_id(target: str, source_path: str, source_line: int) -> str:
    key = f"{target}:{source_path}:{source_line}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class SymbolStore:
    """Thin DAO over the SQLite symbols and symbol_refs tables in state.db."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Symbols
    # ------------------------------------------------------------------

    def upsert_symbols(self, chunks: list[Chunk]) -> None:
        """Insert or replace symbol rows derived from chunk metadata."""
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                _symbol_id(c.path, c.symbol, c.kind, c.start_line),
                c.symbol or "",
                c.kind,
                c.path,
                c.start_line,
                c.end_line,
                c.language,
                c.content_hash,
                now,
            )
            for c in chunks
            if c.symbol  # skip anonymous/module chunks
        ]
        if not rows:
            return
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO symbols
              (id, name, kind, path, start_line, end_line, language, content_hash, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()

    def delete_file(self, path: str) -> None:
        self._conn.execute("DELETE FROM symbols WHERE path = ?", (path,))
        self._conn.execute("DELETE FROM symbol_refs WHERE source_path = ?", (path,))
        self._conn.commit()

    def lookup(self, name: str) -> list[SymbolRecord]:
        rows = self._conn.execute(
            "SELECT * FROM symbols WHERE name = ? ORDER BY path, start_line",
            (name,),
        ).fetchall()
        return [SymbolRecord(**dict(r)) for r in rows]

    def all_symbol_names(self) -> set[str]:
        rows = self._conn.execute("SELECT DISTINCT name FROM symbols").fetchall()
        return {r["name"] for r in rows}

    # ------------------------------------------------------------------
    # References
    # ------------------------------------------------------------------

    def upsert_refs(self, refs: list[RefRecord]) -> None:
        if not refs:
            return
        rows = [
            (
                _ref_id(r.target_name, r.source_path, r.source_line),
                r.target_name,
                r.source_path,
                r.source_line,
                r.line_content,
            )
            for r in refs
        ]
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO symbol_refs
              (id, target_name, source_path, source_line, line_content)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()

    def find_refs(self, target_name: str) -> list[RefRecord]:
        rows = self._conn.execute(
            "SELECT target_name, source_path, source_line, line_content"
            " FROM symbol_refs WHERE target_name = ?"
            " ORDER BY source_path, source_line",
            (target_name,),
        ).fetchall()
        return [
            RefRecord(
                target_name=r["target_name"],
                source_path=r["source_path"],
                source_line=r["source_line"],
                line_content=r["line_content"],
            )
            for r in rows
        ]
