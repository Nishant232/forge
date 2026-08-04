"""Unit tests for observability/logger.py and observability/tracer.py."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from agent.observability.logger import EventLogger
from agent.observability.tracer import Tracer, TraceRecord, format_trace


# ---------------------------------------------------------------------------
# EventLogger
# ---------------------------------------------------------------------------


class TestEventLogger:
    def test_creates_db_file(self, tmp_path: Path) -> None:
        db = tmp_path / "logs.db"
        logger = EventLogger(db)
        assert db.exists()
        logger.close()

    def test_start_task_records_row(self, tmp_path: Path) -> None:
        logger = EventLogger(tmp_path / "logs.db")
        logger.start_task("t1", "Fix the bug")
        task = logger.get_task("t1")
        assert task is not None
        assert task["request"] == "Fix the bug"
        logger.close()

    def test_end_task_updates_outcome(self, tmp_path: Path) -> None:
        logger = EventLogger(tmp_path / "logs.db")
        logger.start_task("t2", "Add docstring")
        logger.end_task("t2", "success", 3, "Added docstring to parse_query")
        task = logger.get_task("t2")
        assert task["outcome"] == "success"
        assert task["iterations"] == 3
        logger.close()

    def test_get_task_returns_none_for_unknown_id(self, tmp_path: Path) -> None:
        logger = EventLogger(tmp_path / "logs.db")
        assert logger.get_task("nonexistent") is None
        logger.close()

    def test_log_node_enter_creates_event(self, tmp_path: Path) -> None:
        logger = EventLogger(tmp_path / "logs.db")
        logger.start_task("t3", "req")
        logger.log_node_enter("t3", "planner")
        events = logger.get_events("t3")
        assert any(e["event_type"] == "node_enter" and e["name"] == "planner" for e in events)
        logger.close()

    def test_log_node_exit_records_duration(self, tmp_path: Path) -> None:
        logger = EventLogger(tmp_path / "logs.db")
        logger.start_task("t4", "req")
        logger.log_node_exit("t4", "act", 42)
        events = logger.get_events("t4")
        exit_ev = next(e for e in events if e["event_type"] == "node_exit")
        assert exit_ev["duration_ms"] == 42
        logger.close()

    def test_log_llm_call_stores_prompt_hash(self, tmp_path: Path) -> None:
        logger = EventLogger(tmp_path / "logs.db")
        logger.start_task("t5", "req")
        logger.log_llm_call("t5", "qwen2.5", "abc123", 200, 512)
        events = logger.get_events("t5")
        llm_ev = next(e for e in events if e["event_type"] == "llm_call")
        import json
        meta = json.loads(llm_ev["metadata"])
        assert meta["prompt_hash"] == "abc123"
        logger.close()

    def test_log_tool_call_records_ok_flag(self, tmp_path: Path) -> None:
        logger = EventLogger(tmp_path / "logs.db")
        logger.start_task("t6", "req")
        logger.log_tool_call("t6", "read_file", "hash1", True, 5)
        events = logger.get_events("t6")
        tool_ev = next(e for e in events if e["event_type"] == "tool_call")
        import json
        meta = json.loads(tool_ev["metadata"])
        assert meta["ok"] is True
        logger.close()

    def test_list_tasks_returns_all(self, tmp_path: Path) -> None:
        logger = EventLogger(tmp_path / "logs.db")
        for i in range(5):
            logger.start_task(f"task_{i}", f"request {i}")
        tasks = logger.list_tasks()
        assert len(tasks) == 5
        logger.close()

    def test_list_tasks_respects_limit(self, tmp_path: Path) -> None:
        logger = EventLogger(tmp_path / "logs.db")
        for i in range(10):
            logger.start_task(f"t_{i}", f"r{i}")
        tasks = logger.list_tasks(limit=3)
        assert len(tasks) == 3
        logger.close()

    def test_get_events_returns_in_order(self, tmp_path: Path) -> None:
        logger = EventLogger(tmp_path / "logs.db")
        logger.start_task("ord", "req")
        logger.log_node_enter("ord", "planner")
        time.sleep(0.01)
        logger.log_node_enter("ord", "act")
        events = logger.get_events("ord")
        names = [e["name"] for e in events]
        assert names.index("planner") < names.index("act")
        logger.close()

    def test_no_raw_content_in_llm_event(self, tmp_path: Path) -> None:
        logger = EventLogger(tmp_path / "logs.db")
        logger.start_task("sec", "secret task")
        secret_text = "password=hunter2"
        # The logger should never store raw content — only hashes
        import hashlib
        hash_val = hashlib.sha256(f'"{secret_text}"'.encode()).hexdigest()[:16]
        logger.log_llm_call("sec", "model", hash_val, 100)
        events = logger.get_events("sec")
        for ev in events:
            assert secret_text not in str(ev.get("metadata", ""))
        logger.close()


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------


class TestTracer:
    def _make_logger(self, tmp_path: Path) -> EventLogger:
        return EventLogger(tmp_path / "logs.db")

    def test_node_enter_event_returns_trace_record(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        logger.start_task("t1", "req")
        tracer = Tracer("t1", logger)
        rec = tracer.handle_event({"event": "on_chain_start", "name": "planner", "run_id": "r1"})
        assert rec is not None
        assert rec.event_type == "node_enter"
        assert rec.name == "planner"
        logger.close()

    def test_node_exit_records_duration(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        logger.start_task("t2", "req")
        tracer = Tracer("t2", logger)
        tracer.handle_event({"event": "on_chain_start", "name": "act", "run_id": "r1"})
        time.sleep(0.01)
        rec = tracer.handle_event({"event": "on_chain_end", "name": "act", "run_id": "r1"})
        assert rec is not None
        assert rec.duration_ms is not None
        assert rec.duration_ms >= 0
        logger.close()

    def test_unknown_node_name_ignored(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        logger.start_task("t3", "req")
        tracer = Tracer("t3", logger)
        rec = tracer.handle_event({"event": "on_chain_start", "name": "LangGraph", "run_id": "r0"})
        assert rec is None
        logger.close()

    def test_llm_end_returns_trace_record(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        logger.start_task("t4", "req")
        tracer = Tracer("t4", logger)
        tracer.handle_event({"event": "on_chat_model_start", "name": "ChatOllama", "run_id": "llm1"})
        rec = tracer.handle_event({
            "event": "on_chat_model_end",
            "name": "ChatOllama",
            "run_id": "llm1",
            "data": {"input": "hello"},
        })
        assert rec is not None
        assert rec.event_type == "llm_call"
        logger.close()

    def test_tool_end_returns_trace_record(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        logger.start_task("t5", "req")
        tracer = Tracer("t5", logger)
        tracer.handle_event({"event": "on_tool_start", "name": "read_file", "run_id": "tc1"})
        rec = tracer.handle_event({
            "event": "on_tool_end",
            "name": "read_file",
            "run_id": "tc1",
            "data": {"input": {"path": "main.py"}},
        })
        assert rec is not None
        assert rec.event_type == "tool_call"
        assert rec.name == "read_file"
        logger.close()

    def test_tool_error_marks_ok_false(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        logger.start_task("t6", "req")
        tracer = Tracer("t6", logger)
        tracer.handle_event({"event": "on_tool_start", "name": "write_file", "run_id": "tc2"})
        rec = tracer.handle_event({
            "event": "on_tool_end",
            "name": "write_file",
            "run_id": "tc2",
            "data": {"input": {}, "error": "permission denied"},
        })
        assert rec is not None
        assert rec.metadata is not None
        assert rec.metadata["ok"] is False
        logger.close()

    def test_unknown_event_type_returns_none(self, tmp_path: Path) -> None:
        logger = self._make_logger(tmp_path)
        logger.start_task("t7", "req")
        tracer = Tracer("t7", logger)
        rec = tracer.handle_event({"event": "on_something_else", "name": "x", "run_id": "y"})
        assert rec is None
        logger.close()


# ---------------------------------------------------------------------------
# format_trace
# ---------------------------------------------------------------------------


class TestFormatTrace:
    def test_contains_task_id(self) -> None:
        task = {"task_id": "abc123", "request": "Fix it", "outcome": "success", "iterations": 2}
        result = format_trace(task, [])
        assert "abc123" in result

    def test_contains_outcome(self) -> None:
        task = {"task_id": "x", "request": "r", "outcome": "failure", "iterations": 5}
        result = format_trace(task, [])
        assert "failure" in result

    def test_contains_event_names(self) -> None:
        task = {"task_id": "x", "request": "r", "outcome": "success", "iterations": 1}
        events = [{"event_type": "node_enter", "name": "planner", "duration_ms": None}]
        result = format_trace(task, events)
        assert "planner" in result

    def test_empty_events_no_error(self) -> None:
        task = {"task_id": "x", "request": "r", "outcome": "success", "iterations": 0}
        result = format_trace(task, [])
        assert isinstance(result, str)
