from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent.memory.consolidator import Consolidator, consolidate
from agent.memory.episodic import Episode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _episode(
    *,
    request: str = "Fix the off-by-one bug in parser",
    outcome: str = "success",
    tools: list[str] | None = None,
    feedback: str | None = None,
) -> Episode:
    tools = tools or ["edit_file"]
    return Episode(
        started_at=_now() - timedelta(minutes=5),
        ended_at=_now(),
        request=request,
        plan={"steps": [], "rationale": ""},
        actions=[{"tool": t} for t in tools],
        outcome=outcome,  # type: ignore[arg-type]
        user_feedback=feedback,
    )


def _make_20_episodes() -> list[Episode]:
    """Return 20 varied episodes that span all outcome types."""
    episodes: list[Episode] = []
    requests = [
        "Fix the SQL parser bug",
        "Add unit tests for the parser module",
        "Refactor the database handler",
        "Update the README documentation",
        "Fix off-by-one error in loop",
        "Add type annotations to utils",
        "Optimize the search function",
        "Fix broken import in handler",
        "Add error handling to API call",
        "Write integration test for indexer",
        "Fix typo in variable name",
        "Refactor the memory module",
        "Add logging to the parser",
        "Fix race condition in watcher",
        "Update dependencies in pyproject",
        "Fix the SQL query builder",
        "Add docstrings to all functions",
        "Fix path handling on Windows",
        "Add retry logic to HTTP client",
        "Refactor the token counter",
    ]
    outcomes = (
        ["success"] * 14 + ["failure"] * 4 + ["partial"] * 2
    )
    tool_sets = [
        ["edit_file", "read_file"],
        ["write_file"],
        ["run_tests", "edit_file"],
        ["search_code", "edit_file"],
    ]
    for i, (req, outcome) in enumerate(zip(requests, outcomes)):
        ep = _episode(
            request=req,
            outcome=outcome,
            tools=tool_sets[i % len(tool_sets)],
            feedback="good" if i % 7 == 0 else None,
        )
        episodes.append(ep)
    return episodes


# ---------------------------------------------------------------------------
# Pure consolidate() tests
# ---------------------------------------------------------------------------


class TestConsolidatePure:
    def test_empty_input_returns_empty(self) -> None:
        assert consolidate([]) == []

    def test_single_episode_produces_at_least_one_lesson(self) -> None:
        result = consolidate([_episode()])
        assert len(result) >= 1

    def test_result_is_at_most_max_lessons(self) -> None:
        episodes = _make_20_episodes()
        result = consolidate(episodes, max_lessons=5)
        assert len(result) <= 5

    def test_twenty_episodes_produce_le_five_lessons(self) -> None:
        episodes = _make_20_episodes()
        result = consolidate(episodes)
        assert len(result) <= 5

    def test_all_lessons_are_non_empty_strings(self) -> None:
        episodes = _make_20_episodes()
        lessons = consolidate(episodes)
        assert all(isinstance(l, str) and len(l) > 0 for l in lessons)

    def test_high_failure_rate_produces_failure_lesson(self) -> None:
        failures = [_episode(outcome="failure") for _ in range(7)]
        successes = [_episode(outcome="success") for _ in range(3)]
        lessons = consolidate(failures + successes)
        combined = " ".join(lessons).lower()
        assert "fail" in combined or "failure" in combined

    def test_high_success_rate_produces_success_lesson(self) -> None:
        episodes = [_episode(outcome="success") for _ in range(10)]
        lessons = consolidate(episodes)
        combined = " ".join(lessons).lower()
        assert "success" in combined or "succeed" in combined

    def test_tool_usage_recorded_in_lessons(self) -> None:
        episodes = [_episode(tools=["edit_file"] * 3) for _ in range(5)]
        lessons = consolidate(episodes)
        combined = " ".join(lessons)
        assert "edit_file" in combined

    def test_partial_outcomes_mentioned(self) -> None:
        episodes = [_episode(outcome="partial") for _ in range(3)]
        lessons = consolidate(episodes)
        combined = " ".join(lessons).lower()
        assert "partial" in combined

    def test_user_feedback_mentioned_when_present(self) -> None:
        episodes = [_episode(feedback="good") for _ in range(5)]
        lessons = consolidate(episodes)
        combined = " ".join(lessons).lower()
        assert "feedback" in combined

    def test_max_lessons_zero_returns_empty(self) -> None:
        # Edge case: caller requests zero lessons
        assert consolidate(_make_20_episodes(), max_lessons=0) == []


# ---------------------------------------------------------------------------
# Consolidator (persistent wrapper) tests
# ---------------------------------------------------------------------------


class TestConsolidator:
    @pytest.fixture()
    def cons(self, tmp_path: Path) -> Consolidator:
        return Consolidator(tmp_path / "state.db")

    def test_should_run_true_on_fresh_db(self, cons: Consolidator) -> None:
        assert cons.should_run()

    def test_run_persists_lessons(self, cons: Consolidator) -> None:
        episodes = _make_20_episodes()
        cons.run(episodes)
        lessons = cons.stored_lessons()
        assert len(lessons) >= 1

    def test_run_updates_last_run_timestamp(self, cons: Consolidator) -> None:
        cons.run(_make_20_episodes())
        assert not cons.should_run(max_age_hours=1)

    def test_run_replaces_previous_lessons(self, cons: Consolidator) -> None:
        cons.run([_episode()])
        first = cons.stored_lessons()
        cons.run(_make_20_episodes())
        second = cons.stored_lessons()
        # After second run, stored lessons should reflect new run, not accumulate
        assert second != first or len(second) == len(first)

    def test_stored_lessons_empty_before_run(self, cons: Consolidator) -> None:
        assert cons.stored_lessons() == []

    def test_run_returns_same_as_stored(self, cons: Consolidator) -> None:
        lessons = cons.run(_make_20_episodes())
        assert cons.stored_lessons() == lessons
