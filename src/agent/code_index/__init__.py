from __future__ import annotations

import asyncio
import logging
from pathlib import Path

_LOG = logging.getLogger(__name__)

_MISSING = (
    "Code indexing dependencies are not installed. "
    "Run: pip install 'local-coding-agent[index]'"
)

try:
    from agent.code_index.chunker import chunk_file
    from agent.code_index.parser import language_for_path
    from agent.code_index.symbols import RefRecord, SymbolStore
    from agent.code_index.vectors import VectorStore
    from agent.llm.embed import embed_texts
    _INDEX_AVAILABLE = True
except ImportError:
    _INDEX_AVAILABLE = False


async def index_file(path: Path, agent_dir: Path) -> int:
    """Index a single source file. Returns the number of chunks stored.

    Skips re-embedding if the file content is unchanged (hash check).
    Returns 0 if the file type is unsupported or the file cannot be read.
    """
    if not _INDEX_AVAILABLE:
        raise ImportError(_MISSING)

    chunks = chunk_file(path)
    if not chunks:
        return 0

    vector_store = VectorStore(agent_dir / "vectors.lance")
    symbol_store = SymbolStore(agent_dir / "state.db")

    existing_hashes = vector_store.get_hashes(str(path))
    current_hashes = {(c.symbol or ""): c.content_hash for c in chunks}
    if existing_hashes == current_hashes:
        _LOG.debug("Skipping unchanged file: %s", path)
        return 0

    vector_store.delete_file(str(path))
    symbol_store.delete_file(str(path))

    texts = [c.content for c in chunks]
    vectors = await embed_texts(texts)

    symbol_store.upsert_symbols(chunks)
    vector_store.upsert(chunks, vectors)
    _LOG.info("Indexed %d chunks from %s", len(chunks), path)
    return len(chunks)


async def index_project(project_root: Path, agent_dir: Path) -> int:
    """Index all supported files in the project. Returns total chunks stored."""
    if not _INDEX_AVAILABLE:
        raise ImportError(_MISSING)

    ignore_dirs = {
        ".agent", ".git", "__pycache__", "node_modules", "target", ".venv",
        ".vscode-server", ".vscode", "snap", ".cache", ".local", ".config",
        ".mozilla", ".npm", ".cargo", ".rustup", "dist", "build", ".next",
    }
    max_file_bytes = 200_000

    files = [
        p
        for p in project_root.rglob("*")
        if p.is_file()
        and language_for_path(p) is not None
        and not any(part in ignore_dirs for part in p.parts)
        and p.stat().st_size <= max_file_bytes
    ]

    total = 0
    for path in files:
        try:
            total += await index_file(path, agent_dir)
        except Exception as e:
            _LOG.warning("Failed to index %s: %s", path, e)

    await detect_references(project_root, agent_dir)
    _LOG.info("Project index complete: %d chunks in %d files", total, len(files))
    return total


async def detect_references(project_root: Path, agent_dir: Path) -> None:
    """Scan the project for call-site references to known symbols."""
    if not _INDEX_AVAILABLE:
        raise ImportError(_MISSING)

    from agent.code_index.parser import extract_references
    from agent.code_index.symbols import SymbolStore

    symbol_store = SymbolStore(agent_dir / "state.db")
    known = symbol_store.all_symbol_names()
    if not known:
        return

    ignore_dirs = {".agent", ".git", "__pycache__", "node_modules", "target", ".venv"}
    refs: list[RefRecord] = []

    for path in project_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignore_dirs for part in path.parts):
            continue
        lang = language_for_path(path)
        if lang is None:
            continue
        try:
            source = path.read_bytes()
            for ref_node in extract_references(source, lang, known, str(path)):
                refs.append(RefRecord(
                    target_name=ref_node.target_name,
                    source_path=str(path),
                    source_line=ref_node.source_line,
                    line_content=ref_node.line_content,
                ))
        except Exception as e:
            _LOG.warning("Reference scan failed for %s: %s", path, e)

    symbol_store.upsert_refs(refs)


def index_file_sync(path: Path, agent_dir: Path) -> int:
    """Synchronous wrapper for index_file."""
    return asyncio.run(index_file(path, agent_dir))


def index_project_sync(project_root: Path, agent_dir: Path) -> int:
    """Synchronous wrapper for index_project."""
    return asyncio.run(index_project(project_root, agent_dir))
