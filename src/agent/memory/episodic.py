"""Episodic memory: persistent store for completed agent tasks.

Each episode records the full lifecycle of one agent task — request, plan,
actions, outcome, and an optional embedding vector for semantic retrieval.

Semantic search is done in-process via cosine similarity over stored embeddings;
no external vector store is needed for the episodic index.
"""
from __future__ import annotations

import json
import math
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

Outcome = Literal["success", "partial", "failure"]


class Episode(BaseModel):
    """One completed agent task.

    `plan` is stored as a dict (serialized JSON) so this module compiles
    without a circular dependency on llm.schemas.  The orchestrator passes
    Plan.model_dump() when saving.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime
    ended_at: datetime
    request: str
    plan: dict[str, Any]
    actions: list[dict[str, Any]]
    outcome: Outcome
    user_feedback: str | None = None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id          TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    ended_at    TEXT NOT NULL,
    request     TEXT NOT NULL,
    plan        TEXT NOT NULL,
    actions     TEXT NOT NULL,
    outcome     TEXT NOT NULL,
    user_feedback TEXT,
    embedding   TEXT
);
CREATE INDEX IF NOT EXISTS idx_episodes_started ON episodes(started_at);
CREATE INDEX IF NOT EXISTS idx_episodes_outcome ON episodes(outcome);
"""

# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class EpisodeStore:
    """SQLite-backed store for episodes with optional embedding-based search."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, episode: Episode, embedding: list[float] | None = None) -> None:
        """Persist an episode, optionally with its embedding vector."""
        emb_json = json.dumps(embedding) if embedding is not None else None
        self._conn.execute(
            """
            INSERT OR REPLACE INTO episodes
              (id, started_at, ended_at, request, plan, actions, outcome, user_feedback, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode.id,
                episode.started_at.isoformat(),
                episode.ended_at.isoformat(),
                episode.request,
                json.dumps(episode.plan),
                json.dumps(episode.actions),
                episode.outcome,
                episode.user_feedback,
                emb_json,
            ),
        )
        self._conn.commit()

    def update_feedback(self, episode_id: str, feedback: str) -> None:
        self._conn.execute(
            "UPDATE episodes SET user_feedback = ? WHERE id = ?",
            (feedback, episode_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(self, episode_id: str) -> Episode | None:
        row = self._conn.execute(
            "SELECT * FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        return _row_to_episode(row) if row else None

    def all(self) -> list[Episode]:
        rows = self._conn.execute(
            "SELECT * FROM episodes ORDER BY started_at"
        ).fetchall()
        return [_row_to_episode(r) for r in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    def list_older_than(self, days: int) -> list[Episode]:
        """Return episodes whose `ended_at` is more than `days` days ago."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE ended_at < ? ORDER BY started_at",
            (cutoff,),
        ).fetchall()
        return [_row_to_episode(r) for r in rows]

    def search_semantic(
        self, query_vec: list[float], k: int = 5
    ) -> list[Episode]:
        """Return up to k episodes sorted by cosine similarity to query_vec.

        Only episodes that have a stored embedding are considered.
        Returns an empty list if no embeddings exist yet.
        """
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE embedding IS NOT NULL"
        ).fetchall()
        if not rows:
            return []

        scored: list[tuple[float, Episode]] = []
        for row in rows:
            emb = json.loads(row["embedding"])
            sim = _cosine(query_vec, emb)
            scored.append((sim, _row_to_episode(row)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:k]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_episode(row: sqlite3.Row) -> Episode:
    return Episode(
        id=row["id"],
        started_at=datetime.fromisoformat(row["started_at"]),
        ended_at=datetime.fromisoformat(row["ended_at"]),
        request=row["request"],
        plan=json.loads(row["plan"]),
        actions=json.loads(row["actions"]),
        outcome=row["outcome"],
        user_feedback=row["user_feedback"],
    )


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity — returns 0.0 on zero-length vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)
