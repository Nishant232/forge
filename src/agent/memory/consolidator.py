"""Memory consolidator: reduce old episodes to compact lessons.

`consolidate()` is a pure function that takes a list of episodes and returns
at most `max_lessons` one-sentence preference statements. It uses heuristics
(outcome statistics, tool frequency, request keywords) rather than an LLM so
it works without a live Ollama instance.

The `Consolidator` class wraps the pure function with SQLite persistence for
lessons and a last-run timestamp so it only fires when the data is stale.
"""
from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent.memory.episodic import Episode

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS lessons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS consolidator_log (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    last_run    TEXT NOT NULL
);
"""

# ---------------------------------------------------------------------------
# Pure consolidation function
# ---------------------------------------------------------------------------


def consolidate(episodes: list[Episode], max_lessons: int = 5) -> list[str]:
    """Distil a list of episodes into at most max_lessons preference statements.

    Returns an empty list for an empty input. The function is deterministic
    and does not call the LLM; the orchestrator may later replace or augment
    this with an LLM-based summarisation step.
    """
    if not episodes:
        return []

    lessons: list[str] = []
    total = len(episodes)
    outcome_counts: Counter[str] = Counter(e.outcome for e in episodes)

    # 1. Outcome statistics
    n_fail = outcome_counts["failure"]
    n_ok = outcome_counts["success"]
    if n_fail > total // 3:
        pct = 100 * n_fail // total
        lessons.append(
            f"{pct}% of recent tasks ended in failure — consider reviewing "
            "tool usage or increasing the verification retry limit."
        )
    elif n_ok >= total * 2 // 3:
        pct = 100 * n_ok // total
        lessons.append(
            f"{pct}% of recent tasks succeed on the first attempt."
        )

    # 2. Most-used tool
    tool_counter: Counter[str] = Counter()
    for ep in episodes:
        for action in ep.actions:
            tool = action.get("tool") or action.get("name", "")
            if tool:
                tool_counter[tool] += 1
    if tool_counter:
        top_tool, count = tool_counter.most_common(1)[0]
        lessons.append(
            f"The most-used tool across recent tasks is '{top_tool}' "
            f"({count} invocations)."
        )

    # 3. Request keyword patterns
    stop_words = {
        "the", "and", "for", "with", "that", "this", "from", "are",
        "have", "has", "been", "into", "not", "but", "can", "will",
    }
    word_counter: Counter[str] = Counter()
    for ep in episodes:
        for word in ep.request.lower().split():
            cleaned = word.strip(".,;:?!()")
            if len(cleaned) > 4 and cleaned not in stop_words:
                word_counter[cleaned] += 1

    common = [w for w, c in word_counter.most_common(5) if c > 1]
    if common:
        lessons.append(
            f"Frequent topics in recent requests: {', '.join(common[:3])}."
        )

    # 4. Partial-outcome hint
    n_partial = outcome_counts["partial"]
    if n_partial > 0:
        lessons.append(
            f"{n_partial} task(s) ended with partial results — "
            "consider raising the iteration cap for complex tasks."
        )

    # 5. User-feedback summary
    feedback_eps = [e for e in episodes if e.user_feedback]
    if feedback_eps:
        lessons.append(
            f"{len(feedback_eps)} task(s) received explicit user feedback — "
            "review these for recurring correction patterns."
        )

    return lessons[:max_lessons]


# ---------------------------------------------------------------------------
# Persistent consolidator
# ---------------------------------------------------------------------------


class Consolidator:
    """Runs `consolidate()` on stale episodes and persists lessons to SQLite."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def should_run(self, max_age_hours: int = 24) -> bool:
        """Return True if the consolidator has not run within max_age_hours."""
        row = self._conn.execute(
            "SELECT last_run FROM consolidator_log WHERE id = 1"
        ).fetchone()
        if row is None:
            return True
        last = datetime.fromisoformat(row["last_run"])
        return datetime.now(timezone.utc) - last > timedelta(hours=max_age_hours)

    def run(
        self,
        episodes: list[Episode],
        max_lessons: int = 5,
    ) -> list[str]:
        """Consolidate episodes, persist lessons, and record the run timestamp."""
        lessons = consolidate(episodes, max_lessons=max_lessons)

        now = datetime.now(timezone.utc).isoformat()

        # Persist lessons (replace previous set)
        self._conn.execute("DELETE FROM lessons")
        self._conn.executemany(
            "INSERT INTO lessons (text, created_at) VALUES (?, ?)",
            [(text, now) for text in lessons],
        )

        # Upsert last-run timestamp
        self._conn.execute(
            "INSERT OR REPLACE INTO consolidator_log (id, last_run) VALUES (1, ?)",
            (now,),
        )
        self._conn.commit()

        _LOG.info("Consolidator produced %d lessons from %d episodes", len(lessons), len(episodes))
        return lessons

    def stored_lessons(self) -> list[str]:
        """Return lessons from the most recent consolidator run."""
        rows = self._conn.execute(
            "SELECT text FROM lessons ORDER BY id"
        ).fetchall()
        return [r["text"] for r in rows]
