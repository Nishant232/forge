from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent.code_index.parser import ParsedNode, language_for_path, parse_file

_LOG = logging.getLogger(__name__)


@dataclass
class Chunk:
    path: str
    start_line: int   # 1-indexed
    end_line: int     # 1-indexed
    symbol: str | None
    kind: Literal["function", "method", "class", "module"]
    content: str
    language: str
    content_hash: str  # SHA-256 hex — used to skip re-embedding unchanged chunks


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_file(path: Path) -> list[Chunk]:
    """Chunk a source file into function/class/module-level Chunk objects.

    Returns an empty list for unsupported file types.
    If no structural nodes are found, returns a single module-level chunk
    covering the whole file.
    """
    result = parse_file(path)
    if result is None:
        return []

    lang, nodes = result

    try:
        source_bytes = path.read_bytes()
    except OSError as exc:
        _LOG.warning("Cannot read %s for chunking: %s", path, exc)
        return []

    source_text = source_bytes.decode("utf-8", errors="replace")
    lines = source_text.splitlines(keepends=True)

    chunks: list[Chunk] = []
    for node in nodes:
        # tree-sitter uses 0-indexed rows; convert to 1-indexed
        start = node.start_line + 1
        end = node.end_line + 1
        content = "".join(lines[node.start_line: node.end_line + 1])
        chunks.append(Chunk(
            path=str(path),
            start_line=start,
            end_line=end,
            symbol=node.name,
            kind=node.kind,
            content=content,
            language=lang,
            content_hash=_sha256(content),
        ))

    if not chunks:
        # Whole-file fallback chunk
        chunks.append(Chunk(
            path=str(path),
            start_line=1,
            end_line=len(lines),
            symbol=None,
            kind="module",
            content=source_text,
            language=lang,
            content_hash=_sha256(source_text),
        ))

    return chunks
