from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lancedb
import pyarrow as pa

from agent.llm.embed import EMBED_DIM

if TYPE_CHECKING:
    from agent.code_index.chunker import Chunk

_LOG = logging.getLogger(__name__)

_TABLE = "chunks"

# Schema version — bump when the schema changes and update migrate().
_SCHEMA_VERSION = 1


def _schema(embed_dim: int = EMBED_DIM) -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("path", pa.string()),
        pa.field("symbol", pa.string()),
        pa.field("kind", pa.string()),
        pa.field("start_line", pa.int32()),
        pa.field("end_line", pa.int32()),
        pa.field("content", pa.string()),
        pa.field("content_hash", pa.string()),
        pa.field("language", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), embed_dim)),
    ])


def migrate(db: Any) -> None:
    """Apply pending schema migrations. No-op for schema version 1.

    When the schema changes in a future version, check the current version
    and apply ALTER TABLE / rebuild steps here before bumping _SCHEMA_VERSION.
    """
    pass  # v1 — nothing to migrate


class VectorStore:
    """LanceDB-backed store for chunk embeddings."""

    def __init__(self, db_path: Path, embed_dim: int = EMBED_DIM) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(db_path))
        self._embed_dim = embed_dim
        migrate(self._db)
        self._table = self._open_or_create()

    def _open_or_create(self) -> Any:
        result = self._db.list_tables()
        # list_tables() returns a ListTablesResponse object in newer lancedb;
        # extract the actual list from .tables if available.
        existing: list[str] = result.tables if hasattr(result, "tables") else list(result)
        if _TABLE in existing:
            return self._db.open_table(_TABLE)
        return self._db.create_table(_TABLE, schema=_schema(self._embed_dim))

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Add chunk+vector rows. Caller must call delete_file first for idempotency."""
        if not chunks or not vectors:
            return
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        rows = [
            {
                "id": f"{c.path}:{c.symbol}:{c.start_line}",
                "path": c.path,
                "symbol": c.symbol or "",
                "kind": c.kind,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "content": c.content,
                "content_hash": c.content_hash,
                "language": c.language,
                "vector": [float(v) for v in vec],
            }
            for c, vec in zip(chunks, vectors)
        ]
        self._table.add(rows)

    def delete_file(self, path: str) -> None:
        """Remove all rows for a given file path."""
        escaped = path.replace("'", "''")
        try:
            self._table.delete(f"path = '{escaped}'")
        except Exception as exc:  # noqa: BLE001
            _LOG.debug("delete_file no-op for %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_hashes(self, path: str) -> dict[str, str]:
        """Return {symbol: content_hash} for all chunks of a file."""
        try:
            rows = (
                self._table.search()
                .where(f"path = '{path.replace(chr(39), chr(39)+chr(39))}'", prefilter=True)
                .limit(10_000)
                .to_list()
            )
            return {r["symbol"]: r["content_hash"] for r in rows}
        except Exception:  # noqa: BLE001
            return {}

    def search(self, query_vector: list[float], k: int = 5) -> list[dict[str, Any]]:
        """Return the top-k nearest chunks by vector similarity."""
        try:
            rows = (
                self._table.search(query_vector)
                .limit(k)
                .to_list()
            )
            return rows
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("Vector search failed: %s", exc)
            return []
