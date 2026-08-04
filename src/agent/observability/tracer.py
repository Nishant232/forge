"""LangGraph event stream consumer: translates astream_events into logger calls.

The Tracer subscribes to the event stream produced by graph.astream_events()
and records structured entries into the EventLogger. One Tracer instance lives
for the lifetime of a single task.

Event types mapped from LangGraph v1 stream:
  on_chain_start   → node_enter (when name is a known node name)
  on_chain_end     → node_exit
  on_chat_model_start / on_llm_start → llm_call start
  on_chat_model_end / on_llm_end     → llm_call end
  on_tool_start    → tool_call start
  on_tool_end      → tool_call end
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from pydantic import BaseModel

from agent.observability.logger import EventLogger

_KNOWN_NODES = frozenset({"planner", "retrieve", "act", "verify", "commit"})


def _hash(data: Any) -> str:
    raw = json.dumps(data, default=str, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class TraceRecord(BaseModel):
    """Lightweight record of one event for display purposes."""

    task_id: str
    event_type: str
    name: str
    ts: float
    duration_ms: int | None = None
    metadata: dict[str, Any] | None = None


class Tracer:
    """Translates LangGraph astream_events into structured log entries.

    Args:
        task_id: The task being traced.
        logger: EventLogger to write into.
    """

    def __init__(self, task_id: str, logger: EventLogger) -> None:
        self._task_id = task_id
        self._logger = logger
        self._node_start: dict[str, float] = {}
        self._llm_start: dict[str, float] = {}
        self._tool_start: dict[str, float] = {}

    def handle_event(self, event: dict[str, Any]) -> TraceRecord | None:
        """Process one LangGraph stream event. Returns a TraceRecord or None."""
        etype = event.get("event", "")
        name = event.get("name", "unknown")
        run_id = event.get("run_id", name)

        if etype == "on_chain_start" and name in _KNOWN_NODES:
            self._node_start[run_id] = time.time()
            self._logger.log_node_enter(self._task_id, name)
            return TraceRecord(task_id=self._task_id, event_type="node_enter", name=name, ts=time.time())

        if etype == "on_chain_end" and name in _KNOWN_NODES:
            start = self._node_start.pop(run_id, time.time())
            dur = int((time.time() - start) * 1000)
            self._logger.log_node_exit(self._task_id, name, dur)
            return TraceRecord(task_id=self._task_id, event_type="node_exit", name=name, ts=time.time(), duration_ms=dur)

        if etype in ("on_chat_model_start", "on_llm_start"):
            self._llm_start[run_id] = time.time()
            return None

        if etype in ("on_chat_model_end", "on_llm_end"):
            start = self._llm_start.pop(run_id, time.time())
            dur = int((time.time() - start) * 1000)
            data = event.get("data", {})
            # Hash input messages — never store raw content
            prompt_hash = _hash(data.get("input", ""))
            model_name = name or "unknown"
            self._logger.log_llm_call(self._task_id, model_name, prompt_hash, dur)
            return TraceRecord(
                task_id=self._task_id,
                event_type="llm_call",
                name=model_name,
                ts=time.time(),
                duration_ms=dur,
                metadata={"prompt_hash": prompt_hash},
            )

        if etype == "on_tool_start":
            self._tool_start[run_id] = time.time()
            return None

        if etype == "on_tool_end":
            start = self._tool_start.pop(run_id, time.time())
            dur = int((time.time() - start) * 1000)
            data = event.get("data", {})
            args_hash = _hash(data.get("input", ""))
            ok = not bool(data.get("error"))
            self._logger.log_tool_call(self._task_id, name, args_hash, ok, dur)
            # Extract a short human-readable hint from the tool input
            tool_input = data.get("input") or {}
            hint = ""
            if isinstance(tool_input, dict):
                hint = (
                    tool_input.get("command")
                    or tool_input.get("path")
                    or tool_input.get("query")
                    or ""
                )
                if len(hint) > 80:
                    hint = hint[:77] + "..."
            return TraceRecord(
                task_id=self._task_id,
                event_type="tool_call",
                name=name,
                ts=time.time(),
                duration_ms=dur,
                metadata={"ok": ok, "args_hash": args_hash, "hint": hint},
            )

        return None


def format_trace(task: dict[str, Any], events: list[dict[str, Any]]) -> str:
    """Render a task trace as human-readable text for `agent log`."""
    lines: list[str] = []
    lines.append(f"Task:    {task.get('task_id', '?')}")
    lines.append(f"Request: {task.get('request', '?')}")
    lines.append(f"Outcome: {task.get('outcome', '?')}")
    lines.append(f"Iters:   {task.get('iterations', 0)}")
    if task.get("summary"):
        lines.append(f"Summary: {task['summary']}")
    lines.append("")
    lines.append("Events:")
    for ev in events:
        etype = ev.get("event_type", "?")
        name = ev.get("name", "?")
        dur = ev.get("duration_ms")
        dur_str = f" ({dur}ms)" if dur is not None else ""
        lines.append(f"  [{etype}] {name}{dur_str}")
    return "\n".join(lines)
