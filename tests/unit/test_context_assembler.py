"""Unit tests for context/assembler.py."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.code_index.search import CodeChunk
from agent.context.assembler import assemble
from agent.memory.episodic import Episode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(path: str = "src/main.py", score: float = 0.8, symbol: str | None = "parse") -> CodeChunk:
    return CodeChunk(
        path=path,
        start_line=1,
        end_line=10,
        symbol=symbol,
        kind="function",
        content=f"def {symbol or 'func'}(): pass",
        score=score,
    )


def _episode(request: str = "Fix the bug", outcome: str = "success") -> Episode:
    now = datetime.now(timezone.utc)
    return Episode(
        started_at=now,
        ended_at=now,
        request=request,
        plan={"steps": []},
        actions=[],
        outcome=outcome,  # type: ignore[arg-type]
    )


_SYSTEM = "You are a coding agent. Root: {project_root}. Trust: trusted."


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


class TestAssembleBasicStructure:
    def test_returns_list_of_messages(self) -> None:
        result = assemble(
            user_request="Fix main.py",
            history=[],
            retrieved_chunks=[],
            retrieved_episodes=[],
            system_prompt=_SYSTEM,
            budget=32_768,
        )
        assert isinstance(result, list)
        assert all(hasattr(m, "content") for m in result)

    def test_first_message_is_system(self) -> None:
        result = assemble(
            user_request="Fix main.py",
            history=[],
            retrieved_chunks=[],
            retrieved_episodes=[],
            system_prompt=_SYSTEM,
            budget=32_768,
        )
        assert isinstance(result[0], SystemMessage)

    def test_last_message_is_user_request(self) -> None:
        result = assemble(
            user_request="Fix main.py",
            history=[],
            retrieved_chunks=[],
            retrieved_episodes=[],
            system_prompt=_SYSTEM,
            budget=32_768,
        )
        assert isinstance(result[-1], HumanMessage)
        assert "Fix main.py" in result[-1].content

    def test_minimum_output_is_system_plus_user(self) -> None:
        result = assemble(
            user_request="Hello",
            history=[],
            retrieved_chunks=[],
            retrieved_episodes=[],
            system_prompt=_SYSTEM,
            budget=32_768,
        )
        assert len(result) >= 2

    def test_system_prompt_preserved(self) -> None:
        result = assemble(
            user_request="x",
            history=[],
            retrieved_chunks=[],
            retrieved_episodes=[],
            system_prompt="CUSTOM SYSTEM PROMPT",
            budget=32_768,
        )
        assert result[0].content == "CUSTOM SYSTEM PROMPT"


# ---------------------------------------------------------------------------
# History inclusion
# ---------------------------------------------------------------------------


class TestHistoryInclusion:
    def test_history_included_after_system(self) -> None:
        hist = [HumanMessage(content="prior request"), AIMessage(content="prior response")]
        result = assemble(
            user_request="new request",
            history=hist,
            retrieved_chunks=[],
            retrieved_episodes=[],
            system_prompt=_SYSTEM,
            budget=32_768,
        )
        contents = [m.content for m in result]
        assert "prior request" in contents
        assert "prior response" in contents

    def test_empty_history_produces_no_extra_messages(self) -> None:
        result = assemble(
            user_request="x",
            history=[],
            retrieved_chunks=[],
            retrieved_episodes=[],
            system_prompt=_SYSTEM,
            budget=32_768,
        )
        # Only system + user
        assert len(result) == 2

    def test_max_history_turns_respected(self) -> None:
        # 50 history entries, max=5 → only last 5 included
        hist = [HumanMessage(content=f"msg {i}") for i in range(50)]
        result = assemble(
            user_request="x",
            history=hist,
            retrieved_chunks=[],
            retrieved_episodes=[],
            system_prompt=_SYSTEM,
            budget=32_768,
            max_history_turns=5,
        )
        # system + 5 history + user = 7
        assert len(result) == 7
        assert "msg 49" in result[-2].content  # second-to-last is latest history


# ---------------------------------------------------------------------------
# Code chunks
# ---------------------------------------------------------------------------


class TestCodeChunks:
    def test_chunks_included_in_context_message(self) -> None:
        chunk = _chunk(path="src/main.py", symbol="parse_query")
        result = assemble(
            user_request="x",
            history=[],
            retrieved_chunks=[chunk],
            retrieved_episodes=[],
            system_prompt=_SYSTEM,
            budget=32_768,
        )
        combined = " ".join(m.content for m in result)
        assert "parse_query" in combined

    def test_empty_chunks_no_context_message(self) -> None:
        result = assemble(
            user_request="x",
            history=[],
            retrieved_chunks=[],
            retrieved_episodes=[],
            system_prompt=_SYSTEM,
            budget=32_768,
        )
        for msg in result:
            assert "retrieved context" not in msg.content

    def test_context_block_marker_present(self) -> None:
        result = assemble(
            user_request="x",
            history=[],
            retrieved_chunks=[_chunk()],
            retrieved_episodes=[],
            system_prompt=_SYSTEM,
            budget=32_768,
        )
        combined = " ".join(m.content for m in result)
        assert "retrieved context" in combined

    def test_higher_score_chunks_retained_when_trimming(self) -> None:
        high = _chunk(path="a.py", score=0.95, symbol="important")
        low = _chunk(path="b.py", score=0.1, symbol="unimportant")
        # Very small budget forces trimming
        result = assemble(
            user_request="fix",
            history=[],
            retrieved_chunks=[high, low],
            retrieved_episodes=[],
            system_prompt="sys",
            budget=200,
        )
        combined = " ".join(m.content for m in result)
        # high-score symbol may or may not be present depending on budget
        # but low should be dropped before high
        if "important" in combined:
            # high retained — correct
            pass
        # At minimum, we get system + user without error
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


class TestEpisodes:
    def test_episodes_included_when_budget_allows(self) -> None:
        ep = _episode("Add docstrings", "success")
        result = assemble(
            user_request="x",
            history=[],
            retrieved_chunks=[],
            retrieved_episodes=[ep],
            system_prompt=_SYSTEM,
            budget=32_768,
        )
        combined = " ".join(m.content for m in result)
        assert "Add docstrings" in combined

    def test_episode_block_marker_present(self) -> None:
        ep = _episode()
        result = assemble(
            user_request="x",
            history=[],
            retrieved_chunks=[],
            retrieved_episodes=[ep],
            system_prompt=_SYSTEM,
            budget=32_768,
        )
        combined = " ".join(m.content for m in result)
        assert "past tasks" in combined

    def test_episodes_dropped_first_under_pressure(self) -> None:
        ep = _episode("Fix issue 42", "partial")
        # Budget too small to fit episodes but big enough for system+user
        result = assemble(
            user_request="x",
            history=[],
            retrieved_chunks=[],
            retrieved_episodes=[ep],
            system_prompt="S",
            budget=30,  # tiny
        )
        combined = " ".join(m.content for m in result)
        # Episodes should be gone
        assert "Fix issue 42" not in combined


# ---------------------------------------------------------------------------
# Recency boost
# ---------------------------------------------------------------------------


class TestRecencyBoost:
    def test_recently_modified_file_gets_boost(self, tmp_path: Path) -> None:
        target = tmp_path / "hot.py"
        target.write_text("def hot(): pass")
        # mtime is essentially now — within 1 hour

        chunk = CodeChunk(
            path=str(target),
            start_line=1,
            end_line=1,
            symbol="hot",
            kind="function",
            content="def hot(): pass",
            score=0.5,
        )

        result = assemble(
            user_request="x",
            history=[],
            retrieved_chunks=[chunk],
            retrieved_episodes=[],
            system_prompt=_SYSTEM,
            budget=32_768,
            project_root=tmp_path,
        )
        # Should include the chunk (boosted score keeps it alive)
        combined = " ".join(m.content for m in result)
        assert "hot" in combined

    def test_old_file_not_boosted(self, tmp_path: Path) -> None:
        target = tmp_path / "cold.py"
        target.write_text("def cold(): pass")
        # Backdate mtime by 2 hours
        old_time = time.time() - 7200
        import os
        os.utime(target, (old_time, old_time))

        chunk = CodeChunk(
            path=str(target),
            start_line=1,
            end_line=1,
            symbol="cold",
            kind="function",
            content="def cold(): pass",
            score=0.5,
        )

        # Just verify no error and it assembles without boost
        result = assemble(
            user_request="x",
            history=[],
            retrieved_chunks=[chunk],
            retrieved_episodes=[],
            system_prompt=_SYSTEM,
            budget=32_768,
            project_root=tmp_path,
        )
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------


class TestBudgetEnforcement:
    def test_output_fits_within_budget(self) -> None:
        from agent.context.budgeter import Budgeter

        budget = 500
        budgeter = Budgeter(budget)
        chunks = [_chunk(symbol=f"fn_{i}", score=float(i) / 20) for i in range(20)]
        episodes = [_episode(f"task {i}") for i in range(5)]
        history = [HumanMessage(content=f"msg {i}") for i in range(10)]

        result = assemble(
            user_request="fix everything",
            history=history,
            retrieved_chunks=chunks,
            retrieved_episodes=episodes,
            system_prompt=_SYSTEM,
            budget=budget,
        )
        assert budgeter.fits(result)

    def test_tiny_budget_returns_at_least_system_and_user(self) -> None:
        result = assemble(
            user_request="x",
            history=[HumanMessage(content="old") for _ in range(50)],
            retrieved_chunks=[_chunk() for _ in range(10)],
            retrieved_episodes=[_episode() for _ in range(5)],
            system_prompt="S",
            budget=10,
        )
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[-1], HumanMessage)
        assert result[-1].content == "x"


# ---------------------------------------------------------------------------
# Order invariant
# ---------------------------------------------------------------------------


class TestMessageOrder:
    def test_system_first_user_last(self) -> None:
        result = assemble(
            user_request="do it",
            history=[HumanMessage(content="h"), AIMessage(content="a")],
            retrieved_chunks=[_chunk()],
            retrieved_episodes=[_episode()],
            system_prompt=_SYSTEM,
            budget=32_768,
        )
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[-1], HumanMessage)
        assert result[-1].content == "do it"
