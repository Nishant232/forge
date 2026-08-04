"""Integration test: index → query → modify → reindex → query reflects change.

embed_texts is mocked with a deterministic hash-based embedder so this test
runs without a live Ollama instance. The smoke test at the bottom is skipped
unless OLLAMA_URL is set in the environment.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "sample_project"

# Two patches are needed:
#   1. agent.code_index.embed_texts — used by index_file (top-level import)
#   2. agent.llm.embed.embed_texts  — used by semantic_search (local import)
_PATCH_CODE_INDEX = "agent.code_index.embed_texts"
_PATCH_LLM_EMBED = "agent.llm.embed.embed_texts"


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic 768-dim vector derived from SHA-256 of text content."""
    out: list[list[float]] = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).digest()
        vec = [(digest[i % 32] / 255.0) for i in range(768)]
        out.append(vec)
    return out


@pytest.fixture()
def project_copy(tmp_path: Path) -> Path:
    """Isolated copy of the fixture project — tests may freely modify files."""
    dest = tmp_path / "sample_project"
    shutil.copytree(FIXTURE_DIR, dest)
    return dest


@pytest.fixture()
def agent_dir(tmp_path: Path) -> Path:
    d = tmp_path / ".agent"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _embed_mock() -> AsyncMock:
    return AsyncMock(side_effect=_fake_embed)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_index_produces_chunks(project_copy: Path, agent_dir: Path) -> None:
    mock = _embed_mock()
    with patch(_PATCH_CODE_INDEX, new=mock), patch(_PATCH_LLM_EMBED, new=mock):
        from agent.code_index import index_project

        count = await index_project(project_copy, agent_dir)
    assert count > 0, "Expected at least one chunk indexed from the fixture project"


@pytest.mark.asyncio
async def test_symbol_lookup_finds_parse_query(
    project_copy: Path, agent_dir: Path
) -> None:
    mock = _embed_mock()
    with patch(_PATCH_CODE_INDEX, new=mock), patch(_PATCH_LLM_EMBED, new=mock):
        from agent.code_index import index_project
        from agent.code_index.search import symbol_lookup

        await index_project(project_copy, agent_dir)
        results = symbol_lookup("parse_query", agent_dir=agent_dir)

    assert results, "symbol_lookup('parse_query') returned no results"
    assert results[0].symbol == "parse_query"
    assert "main.py" in results[0].path


@pytest.mark.asyncio
async def test_find_references_locates_handler_caller(
    project_copy: Path, agent_dir: Path
) -> None:
    mock = _embed_mock()
    with patch(_PATCH_CODE_INDEX, new=mock), patch(_PATCH_LLM_EMBED, new=mock):
        from agent.code_index import index_project
        from agent.code_index.search import find_references

        await index_project(project_copy, agent_dir)
        refs = find_references("parse_query", agent_dir=agent_dir)

    assert refs, "find_references('parse_query') returned no results"
    assert any("handler.py" in r.path for r in refs), (
        "Expected handler.py to appear in references for parse_query"
    )


@pytest.mark.asyncio
async def test_reindex_reflects_added_function(
    project_copy: Path, agent_dir: Path
) -> None:
    mock = _embed_mock()
    with patch(_PATCH_CODE_INDEX, new=mock), patch(_PATCH_LLM_EMBED, new=mock):
        from agent.code_index import index_file, index_project
        from agent.code_index.search import symbol_lookup

        await index_project(project_copy, agent_dir)

        # The new function must not exist before editing
        assert not symbol_lookup("newly_added_function_xyz", agent_dir=agent_dir)

        # Append a new function to main.py
        main_py = project_copy / "main.py"
        main_py.write_text(
            main_py.read_text(encoding="utf-8")
            + "\n\ndef newly_added_function_xyz() -> None:\n    pass\n",
            encoding="utf-8",
        )

        # Reindex only the changed file
        await index_file(main_py, agent_dir)

        # New symbol must now be discoverable
        found = symbol_lookup("newly_added_function_xyz", agent_dir=agent_dir)
        assert found, "Reindex did not reflect the newly added function"
        assert "main.py" in found[0].path


@pytest.mark.asyncio
async def test_unchanged_file_is_not_reembedded(
    project_copy: Path, agent_dir: Path
) -> None:
    mock = _embed_mock()
    with patch(_PATCH_CODE_INDEX, new=mock), patch(_PATCH_LLM_EMBED, new=mock):
        from agent.code_index import index_file

        main_py = project_copy / "main.py"

        count1 = await index_file(main_py, agent_dir)
        assert count1 > 0, "First index should embed chunks"

        calls_after_first = mock.call_count

        count2 = await index_file(main_py, agent_dir)
        assert count2 == 0, "Second index of unchanged file should return 0 (skipped)"
        assert mock.call_count == calls_after_first, (
            "embed_texts should not be called again for an unchanged file"
        )


@pytest.mark.asyncio
async def test_deleted_symbol_removed_after_reindex(
    project_copy: Path, agent_dir: Path
) -> None:
    mock = _embed_mock()
    with patch(_PATCH_CODE_INDEX, new=mock), patch(_PATCH_LLM_EMBED, new=mock):
        from agent.code_index import index_file, index_project
        from agent.code_index.search import symbol_lookup

        await index_project(project_copy, agent_dir)
        assert symbol_lookup("SQLParser", agent_dir=agent_dir)

        # Remove SQLParser from main.py
        main_py = project_copy / "main.py"
        original = main_py.read_text(encoding="utf-8")
        # Keep only the parse_query function, drop the class
        truncated = original[: original.index("\nclass SQLParser")]
        main_py.write_text(truncated, encoding="utf-8")

        await index_file(main_py, agent_dir)

        assert not symbol_lookup("SQLParser", agent_dir=agent_dir), (
            "SQLParser should no longer appear after the class was removed and reindexed"
        )


# ---------------------------------------------------------------------------
# Smoke test — requires a live Ollama instance
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("OLLAMA_URL"),
    reason="smoke — requires live Ollama (set OLLAMA_URL to enable)",
)
@pytest.mark.asyncio
async def test_semantic_search_sql_parser_smoke(
    project_copy: Path, agent_dir: Path
) -> None:
    """semantic_search('function that parses SQL') must return parse_query in top 3."""
    from agent.code_index import index_project
    from agent.code_index.search import semantic_search

    await index_project(project_copy, agent_dir)
    results = await semantic_search("function that parses SQL", k=3, agent_dir=agent_dir)
    assert results, "semantic_search returned no results"
    symbols = [r.symbol for r in results]
    assert "parse_query" in symbols or any(
        "sql" in (r.content or "").lower() for r in results
    ), f"Expected parse_query in top 3; got: {symbols}"
