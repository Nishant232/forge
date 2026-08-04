"""Structured event logger: writes agent events to SQLite.

Each invocation of run_task() produces one parent trace row plus child rows
for every node entry, LLM call, and tool call. File contents are referenced
by hash, never pasted in, to avoid logging PII or secrets.

Schema
------
traces      — one row per task
trace_events — one row per node/llm/tool event within a task
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


_CREATE_TRACES = """
CREATE TABLE IF NOT EXISTS traces (
    task_id     TEXT PRIMARY KEY,
    request     TEXT NOT NULL,
    started_at  REAL NOT NULL,
    ended_at    REAL,
    outcome     TEXT,
    iterations  INTEGER DEFAULT 0,
    summary     TEXT
)
"""

_CREATE_EVENTS = """
CREATE TABLE IF NOT EXISTS trace_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT NOT NULL,
    event_type  TEXT NOT NULL,   -- 'node_enter'|'node_exit'|'llm_call'|'tool_call'
    name        TEXT NOT NULL,
    ts          REAL NOT NULL,
    duration_ms INTEGER,
    metadata    TEXT             -- JSON blob (small, no raw content)
)
"""


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class EventLogger:
    """Thread-safe SQLite logger for agent events.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TRACES)
        self._conn.execute(_CREATE_EVENTS)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def start_task(self, task_id: str, request: str) -> None:
        """Record the start of a task."""
        self._conn.execute(
            "INSERT OR REPLACE INTO traces (task_id, request, started_at) VALUES (?, ?, ?)",
            (task_id, request, time.time()),
        )
        self._conn.commit()

    def end_task(
        self,
        task_id: str,
        outcome: str,
        iterations: int,
        summary: str,
    ) -> None:
        """Record the end of a task."""
        self._conn.execute(
            """UPDATE traces
               SET ended_at=?, outcome=?, iterations=?, summary=?
               WHERE task_id=?""",
            (time.time(), outcome, iterations, summary, task_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------

    def _log_event(
        self,
        task_id: str,
        event_type: str,
        name: str,
        duration_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO trace_events
               (task_id, event_type, name, ts, duration_ms, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                event_type,
                name,
                time.time(),
                duration_ms,
                json.dumps(metadata) if metadata else None,
            ),
        )
        self._conn.commit()

    def log_node_enter(self, task_id: str, node_name: str) -> None:
        self._log_event(task_id, "node_enter", node_name)

    def log_node_exit(self, task_id: str, node_name: str, duration_ms: int) -> None:
        self._log_event(task_id, "node_exit", node_name, duration_ms=duration_ms)

    def log_llm_call(
        self,
        task_id: str,
        model: str,
        prompt_hash: str,
        duration_ms: int,
        token_estimate: int = 0,
    ) -> None:
        self._log_event(
            task_id,
            "llm_call",
            model,
            duration_ms=duration_ms,
            metadata={"prompt_hash": prompt_hash, "token_estimate": token_estimate},
        )

    def log_tool_call(
        self,
        task_id: str,
        tool_name: str,
        args_hash: str,
        ok: bool,
        duration_ms: int,
    ) -> None:
        self._log_event(
            task_id,
            "tool_call",
            tool_name,
            duration_ms=duration_ms,
            metadata={"args_hash": args_hash, "ok": ok},
        )

    # ------------------------------------------------------------------
    # Query helpers (used by `agent log`)
    # ------------------------------------------------------------------

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM traces WHERE task_id=?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        cols = [d[0] for d in self._conn.execute("SELECT * FROM traces LIMIT 0").description or []]
        # Re-fetch with description
        cur = self._conn.execute("SELECT * FROM traces WHERE task_id=?", (task_id,))
        cols = [d[0] for d in cur.description]
        row = cur.fetchone()
        return dict(zip(cols, row)) if row else None

    def get_events(self, task_id: str) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM trace_events WHERE task_id=? ORDER BY ts", (task_id,)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def list_tasks(self, limit: int = 50) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM traces ORDER BY started_at DESC LIMIT ?", (limit,)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()
