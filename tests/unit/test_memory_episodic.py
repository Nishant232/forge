from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent.memory.episodic import Episode, EpisodeStore, _cosine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _episode(
    *,
    request: str = "Fix the off-by-one bug",
    outcome: str = "success",
    delta_seconds: int = 0,
) -> Episode:
    started = _now() - timedelta(seconds=delta_seconds + 60)
    ended = _now() - timedelta(seconds=delta_seconds)
    return Episode(
        started_at=started,
        ended_at=ended,
        request=request,
        plan={"steps": [], "rationale": "test plan"},
        actions=[{"tool": "edit_file", "args": {}}],
        outcome=outcome,  # type: ignore[arg-type]
        user_feedback=None,
    )


@pytest.fixture()
def store(tmp_path: Path) -> EpisodeStore:
    return EpisodeStore(tmp_path / "state.db")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSaveAndLoad:
    def test_save_then_load_round_trips(self, store: EpisodeStore) -> None:
        ep = _episode()
        store.save(ep)
        loaded = store.load(ep.id)
        assert loaded is not None
        assert loaded.id == ep.id
        assert loaded.request == ep.request
        assert loaded.outcome == ep.outcome

    def test_load_unknown_id_returns_none(self, store: EpisodeStore) -> None:
        assert store.load("does-not-exist") is None

    def test_save_is_idempotent(self, store: EpisodeStore) -> None:
        ep = _episode()
        store.save(ep)
        store.save(ep)
        assert store.count() == 1

    def test_plan_and_actions_survive_round_trip(self, store: EpisodeStore) -> None:
        ep = _episode()
        ep.plan["steps"] = [{"description": "step1", "expected_tools": ["read_file"]}]
        ep.actions = [{"tool": "write_file", "args": {"path": "a.py"}}]
        store.save(ep)
        loaded = store.load(ep.id)
        assert loaded is not None
        assert loaded.plan["steps"][0]["description"] == "step1"
        assert loaded.actions[0]["tool"] == "write_file"

    def test_user_feedback_preserved(self, store: EpisodeStore) -> None:
        ep = _episode()
        ep.user_feedback = "great job"
        store.save(ep)
        assert store.load(ep.id).user_feedback == "great job"  # type: ignore[union-attr]

    def test_timestamps_preserved(self, store: EpisodeStore) -> None:
        ep = _episode()
        store.save(ep)
        loaded = store.load(ep.id)
        assert loaded is not None
        assert abs((loaded.started_at - ep.started_at).total_seconds()) < 1
        assert abs((loaded.ended_at - ep.ended_at).total_seconds()) < 1


class TestAll:
    def test_all_returns_all_episodes(self, store: EpisodeStore) -> None:
        for _ in range(3):
            store.save(_episode())
        assert len(store.all()) == 3

    def test_empty_store_returns_empty_list(self, store: EpisodeStore) -> None:
        assert store.all() == []

    def test_count_matches_saves(self, store: EpisodeStore) -> None:
        store.save(_episode())
        store.save(_episode())
        assert store.count() == 2


class TestListOlderThan:
    def test_returns_old_episodes_only(self, store: EpisodeStore) -> None:
        # old episode: ended 3 days ago
        old = _episode(delta_seconds=3 * 24 * 3600)
        # recent episode: ended just now
        recent = _episode(delta_seconds=0)
        store.save(old)
        store.save(recent)
        result = store.list_older_than(days=2)
        ids = {e.id for e in result}
        assert old.id in ids
        assert recent.id not in ids

    def test_returns_empty_when_none_old(self, store: EpisodeStore) -> None:
        store.save(_episode(delta_seconds=0))
        assert store.list_older_than(days=10) == []


class TestSemanticSearch:
    def test_returns_empty_without_embeddings(self, store: EpisodeStore) -> None:
        store.save(_episode())
        assert store.search_semantic([0.1] * 768) == []

    def test_returns_highest_similarity_first(self, store: EpisodeStore) -> None:
        ep_a = _episode(request="SQL parser task")
        ep_b = _episode(request="File rename task")

        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        query = [1.0, 0.0, 0.0]  # identical to vec_a

        store.save(ep_a, embedding=vec_a)
        store.save(ep_b, embedding=vec_b)

        results = store.search_semantic(query, k=2)
        assert results[0].id == ep_a.id

    def test_respects_k_limit(self, store: EpisodeStore) -> None:
        for i in range(5):
            store.save(_episode(), embedding=[float(i)] * 3)
        results = store.search_semantic([1.0, 1.0, 1.0], k=2)
        assert len(results) <= 2

    def test_search_ignores_episodes_without_embedding(
        self, store: EpisodeStore
    ) -> None:
        ep_with = _episode(request="with embedding")
        ep_without = _episode(request="without embedding")
        store.save(ep_with, embedding=[1.0, 0.0])
        store.save(ep_without)  # no embedding

        results = store.search_semantic([1.0, 0.0], k=5)
        ids = {e.id for e in results}
        assert ep_with.id in ids
        assert ep_without.id not in ids


class TestUpdateFeedback:
    def test_update_sets_feedback(self, store: EpisodeStore) -> None:
        ep = _episode()
        store.save(ep)
        store.update_feedback(ep.id, "nice work")
        assert store.load(ep.id).user_feedback == "nice work"  # type: ignore[union-attr]


class TestCosineHelper:
    def test_identical_vectors_give_one(self) -> None:
        assert math.isclose(_cosine([1.0, 0.0], [1.0, 0.0]), 1.0)

    def test_orthogonal_vectors_give_zero(self) -> None:
        assert math.isclose(_cosine([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_zero_vector_gives_zero(self) -> None:
        assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_different_lengths_give_zero(self) -> None:
        assert _cosine([1.0, 2.0], [1.0]) == 0.0
