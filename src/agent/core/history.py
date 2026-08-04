"""Persistent conversation history backed by SQLite.

Each "session" is one continuous conversation (like a Claude Code session).
Messages are stored individually and replayed in order to rebuild context.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

_DDL = """
CREATE TABLE IF NOT EXISTS conv_sessions (
    id          TEXT PRIMARY KEY,
    workspace   TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conv_messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES conv_sessions(id),
    role         TEXT NOT NULL,
    content      TEXT NOT NULL DEFAULT '',
    tool_calls   TEXT,
    tool_call_id TEXT,
    tool_name    TEXT,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_msgs_session ON conv_messages(session_id, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize(msg: BaseMessage) -> dict[str, Any]:
    if isinstance(msg, HumanMessage):
        return {"role": "human", "content": str(msg.content)}
    if isinstance(msg, AIMessage):
        return {
            "role": "ai",
            "content": str(msg.content) if msg.content else "",
            "tool_calls": json.dumps(msg.tool_calls or []),
        }
    if isinstance(msg, ToolMessage):
        return {
            "role": "tool",
            "content": str(msg.content),
            "tool_call_id": msg.tool_call_id or "",
            "tool_name": getattr(msg, "name", "") or "",
        }
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": str(msg.content)}
    return {"role": "human", "content": str(msg.content)}


def _deserialize(row: dict[str, Any]) -> BaseMessage:
    role = row["role"]
    content = row.get("content", "")
    if role == "human":
        return HumanMessage(content=content)
    if role == "ai":
        tcs = json.loads(row.get("tool_calls") or "[]")
        return AIMessage(content=content, tool_calls=tcs)
    if role == "tool":
        return ToolMessage(
            content=content,
            tool_call_id=row.get("tool_call_id") or "",
            name=row.get("tool_name") or "",
        )
    if role == "system":
        return SystemMessage(content=content)
    return HumanMessage(content=content)


class ConversationHistory:
    """SQLite-backed conversation history for persistent multi-turn sessions."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def create_session(self, workspace: str, title: str = "") -> str:
        sid = str(uuid.uuid4())
        now = _now()
        self._conn.execute(
            "INSERT INTO conv_sessions VALUES (?,?,?,?,?)",
            (sid, workspace, title, now, now),
        )
        self._conn.commit()
        return sid

    def update_title(self, session_id: str, title: str) -> None:
        self._conn.execute(
            "UPDATE conv_sessions SET title=?, updated_at=? WHERE id=?",
            (title, _now(), session_id),
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM conv_sessions WHERE id=?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM conv_sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Message storage
    # ------------------------------------------------------------------

    def load(self, session_id: str) -> list[BaseMessage]:
        rows = self._conn.execute(
            "SELECT * FROM conv_messages WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [_deserialize(dict(r)) for r in rows]

    def append(self, session_id: str, msg: BaseMessage) -> None:
        d = _serialize(msg)
        self._conn.execute(
            """INSERT INTO conv_messages
               (session_id, role, content, tool_calls, tool_call_id, tool_name, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (
                session_id,
                d["role"],
                d.get("content", ""),
                d.get("tool_calls"),
                d.get("tool_call_id"),
                d.get("tool_name"),
                _now(),
            ),
        )
        self._conn.execute(
            "UPDATE conv_sessions SET updated_at=? WHERE id=?",
            (_now(), session_id),
        )
        self._conn.commit()

    def append_many(self, session_id: str, msgs: list[BaseMessage]) -> None:
        """Persist multiple messages in a single SQLite transaction for efficiency."""
        if not msgs:
            return
        now = _now()
        rows = []
        for msg in msgs:
            d = _serialize(msg)
            rows.append((
                session_id,
                d["role"],
                d.get("content", ""),
                d.get("tool_calls"),
                d.get("tool_call_id"),
                d.get("tool_name"),
                now,
            ))
        self._conn.executemany(
            """INSERT INTO conv_messages
               (session_id, role, content, tool_calls, tool_call_id, tool_name, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            rows,
        )
        self._conn.execute(
            "UPDATE conv_sessions SET updated_at=? WHERE id=?",
            (now, session_id),
        )
        self._conn.commit()

    def message_count(self, session_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM conv_messages WHERE session_id=?", (session_id,)
        ).fetchone()
        return row[0] if row else 0

    def clear_session(self, session_id: str) -> None:
        """Delete all messages for a session, keeping the session metadata."""
        self._conn.execute(
            "DELETE FROM conv_messages WHERE session_id=?", (session_id,)
        )
        self._conn.execute(
            "UPDATE conv_sessions SET updated_at=? WHERE id=?",
            (_now(), session_id),
        )
        self._conn.commit()
