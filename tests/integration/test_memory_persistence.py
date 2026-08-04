"""Integration test: memory stores survive process restart.

Each test creates a store, writes data, closes it (simulating process exit),
then reopens it and verifies the data is still there.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent.memory.episodic import Episode, EpisodeStore
from agent.memory.project import ProjectStore
from agent.memory.consolidator import Consolidator


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _episode(request: str = "Fix the bug") -> Episode:
    t = _now()
    return Episode(
        started_at=t,
        ended_at=t,
        request=request,
        plan={"steps": [{"description": "step", "expected_tools": ["edit_file"]}]},
        actions=[{"tool": "edit_file", "args": {"path": "a.py"}}],
        outcome="success",
        user_feedback=None,
    )


class TestEpisodicPersistence:
    def test_episode_survives_store_close_and_reopen(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        ep = _episode("Fix the SQL parser off-by-one bug")

        # First session: save
        store1 = EpisodeStore(db)
        store1.save(ep)
        store1.close()

        # Second session: reload
        store2 = EpisodeStore(db)
        loaded = store2.load(ep.id)
        store2.close()

        assert loaded is not None
        assert loaded.id == ep.id
        assert loaded.request == ep.request
        assert loaded.outcome == ep.outcome

    def test_multiple_episodes_all_survive_restart(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        eps = [_episode(f"Task {i}") for i in range(5)]

        store1 = EpisodeStore(db)
        for ep in eps:
            store1.save(ep)
        store1.close()

        store2 = EpisodeStore(db)
        assert store2.count() == 5
        store2.close()

    def test_embedding_survives_restart(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        ep = _episode("SQL search task")
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        store1 = EpisodeStore(db)
        store1.save(ep, embedding=embedding)
        store1.close()

        store2 = EpisodeStore(db)
        results = store2.search_semantic([0.1, 0.2, 0.3, 0.4, 0.5], k=1)
        store2.close()

        assert len(results) == 1
        assert results[0].id == ep.id

    def test_semantic_search_returns_similar_past_tasks(
        self, tmp_path: Path
    ) -> None:
        """Searching episodic memory returns semantically similar past tasks."""
        db = tmp_path / "state.db"

        ep_sql = _episode("Fix the SQL parser function")
        ep_ui = _episode("Update the dashboard colour scheme")

        # Simulate embeddings: SQL task has vec close to query; UI task far away
        vec_sql = [1.0, 0.0, 0.0]
        vec_ui = [0.0, 1.0, 0.0]
        query = [0.9, 0.1, 0.0]  # closest to vec_sql

        store1 = EpisodeStore(db)
        store1.save(ep_sql, embedding=vec_sql)
        store1.save(ep_ui, embedding=vec_ui)
        store1.close()

        store2 = EpisodeStore(db)
        results = store2.search_semantic(query, k=1)
        store2.close()

        assert results[0].id == ep_sql.id


class TestProjectPersistence:
    def test_project_fact_survives_restart(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"

        store1 = ProjectStore(db)
        store1.set("test_runner", "pytest", source="manual")
        store1.close()

        store2 = ProjectStore(db)
        assert store2.get("test_runner") == "pytest"
        store2.close()

    def test_detected_conventions_survive_restart(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        (tmp_path / "pyproject.toml").write_text(
            "[tool.pytest.ini_options]\ntestpaths = ['tests']\n"
            "[tool.ruff]\nline-length = 100\n"
        )

        store1 = ProjectStore(db)
        store1.detect_and_persist(tmp_path)
        store1.close()

        store2 = ProjectStore(db)
        assert store2.get("test_runner") == "pytest"
        assert store2.get("linter") == "ruff"
        store2.close()


class TestConsolidatorPersistence:
    def test_lessons_survive_restart(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"

        eps = [
            _episode(f"Task {i} involving refactor and SQL")
            for i in range(5)
        ]

        cons1 = Consolidator(db)
        lessons1 = cons1.run(eps)
        assert len(lessons1) >= 1
        cons1.close()

        cons2 = Consolidator(db)
        lessons2 = cons2.stored_lessons()
        cons2.close()

        assert lessons1 == lessons2

    def test_last_run_persists_across_restart(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"

        cons1 = Consolidator(db)
        assert cons1.should_run()
        cons1.run([_episode()])
        cons1.close()

        cons2 = Consolidator(db)
        assert not cons2.should_run(max_age_hours=1)
        cons2.close()
