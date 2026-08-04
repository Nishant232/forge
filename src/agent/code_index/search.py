from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from agent.code_index.symbols import SymbolStore
from agent.code_index.vectors import VectorStore


class CodeChunk(BaseModel):
    """A chunk of source code with retrieval metadata.

    This is the canonical retrieval unit flowing from code_index through the
    orchestrator into the context assembler.
    """

    path: str
    start_line: int
    end_line: int
    symbol: str | None
    kind: Literal["function", "method", "class", "module"]
    content: str
    score: float


def _agent_dir() -> Path:
    root = os.environ.get("AGENT_PROJECT_ROOT", ".")
    return Path(root) / ".agent"


def _project_root() -> Path:
    return Path(os.environ.get("AGENT_PROJECT_ROOT", "."))


async def semantic_search(
    query: str,
    k: int = 5,
    agent_dir: Path | None = None,
) -> list[CodeChunk]:
    """Embed `query` and return the top-k semantically similar code chunks."""
    from agent.llm.embed import embed_texts

    adir = agent_dir or _agent_dir()
    vectors = await embed_texts([query])
    query_vector = vectors[0]

    store = VectorStore(adir / "vectors.lance")
    rows = store.search(query_vector, k=k)

    return [
        CodeChunk(
            path=r["path"],
            start_line=int(r["start_line"]),
            end_line=int(r["end_line"]),
            symbol=r["symbol"] or None,
            kind=r["kind"],
            content=r["content"],
            score=1.0 - float(r.get("_distance", 0.0)),
        )
        for r in rows
    ]


def symbol_lookup(name: str, agent_dir: Path | None = None) -> list[CodeChunk]:
    """Return all symbols with an exact name match."""
    adir = agent_dir or _agent_dir()
    store = SymbolStore(adir / "state.db")
    records = store.lookup(name)
    return [
        CodeChunk(
            path=r.path,
            start_line=r.start_line,
            end_line=r.end_line,
            symbol=r.name,
            kind=r.kind,  # type: ignore[arg-type]
            content=_read_lines(r.path, r.start_line, r.end_line),
            score=1.0,
        )
        for r in records
    ]


def find_references(symbol: str, agent_dir: Path | None = None) -> list[CodeChunk]:
    """Return all recorded call-site references to the named symbol."""
    adir = agent_dir or _agent_dir()
    store = SymbolStore(adir / "state.db")
    refs = store.find_refs(symbol)
    return [
        CodeChunk(
            path=r.source_path,
            start_line=r.source_line,
            end_line=r.source_line,
            symbol=symbol,
            kind="function",
            content=r.line_content,
            score=1.0,
        )
        for r in refs
    ]


def text_search(
    pattern: str,
    project_root: Path | None = None,
    max_results: int = 50,
) -> list[CodeChunk]:
    """Search for a text pattern using ripgrep and wrap results as CodeChunks."""
    root = project_root or _project_root()
    cmd = ["rg", "--line-number", "--no-heading", "--color=never", pattern, str(root)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        raise RuntimeError("ripgrep (rg) not found on PATH")

    chunks: list[CodeChunk] = []
    for line in result.stdout.strip().splitlines()[:max_results]:
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        path_str, lineno_str, content = parts
        try:
            lineno = int(lineno_str)
        except ValueError:
            continue
        chunks.append(CodeChunk(
            path=path_str,
            start_line=lineno,
            end_line=lineno,
            symbol=None,
            kind="module",
            content=content.strip(),
            score=1.0,
        ))
    return chunks


def _read_lines(path: str, start: int, end: int) -> str:
    """Read lines start..end (1-indexed, inclusive) from a file."""
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[start - 1: end])
    except OSError:
        return ""
