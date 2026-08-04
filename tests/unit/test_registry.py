from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.tools import BaseTool

from agent.tools.registry import ToolResult, all_tools, dispatch


class TestAllTools:
    def test_returns_non_empty_list(self) -> None:
        tools = all_tools()
        assert len(tools) > 0

    def test_all_elements_are_base_tool(self) -> None:
        for t in all_tools():
            assert isinstance(t, BaseTool), f"{t!r} is not a BaseTool"

    def test_tools_have_names(self) -> None:
        for t in all_tools():
            assert isinstance(t.name, str)
            assert len(t.name) > 0

    def test_tools_have_descriptions(self) -> None:
        for t in all_tools():
            assert isinstance(t.description, str)
            assert len(t.description) > 0

    def test_tools_have_args_schema(self) -> None:
        for t in all_tools():
            assert t.args_schema is not None

    def test_expected_tools_present(self) -> None:
        names = {t.name for t in all_tools()}
        expected = {
            "read_file",
            "write_file",
            "edit_file",
            "list_dir",
            "run_shell",
            "search_code",
            "run_tests",
            "git_status",
            "git_diff",
            "git_add",
            "git_commit",
            "git_log",
            "git_checkpoint",
            "git_rollback",
        }
        assert expected.issubset(names)

    def test_no_duplicate_names(self) -> None:
        names = [t.name for t in all_tools()]
        assert len(names) == len(set(names))

    def test_bindable_to_chat_model_schema(self) -> None:
        """Each tool must produce a valid JSON schema (required for bind_tools)."""
        for t in all_tools():
            schema = t.args_schema.model_json_schema()  # type: ignore[union-attr]
            assert "properties" in schema or schema.get("type") == "object"


class TestDispatch:
    def test_dispatches_known_tool(self, project_root: Path) -> None:
        f = project_root / "d.txt"
        f.write_text("hello")
        result = dispatch({"name": "read_file", "args": {"path": str(f)}})
        assert result.ok is True
        assert "hello" in result.output

    def test_unknown_tool_returns_error(self) -> None:
        result = dispatch({"name": "nonexistent_tool", "args": {}})
        assert result.ok is False
        assert result.error is not None
        assert "nonexistent_tool" in result.error

    def test_tool_exception_captured(self, project_root: Path) -> None:
        # read_file on a missing file raises FileNotFoundError → ToolResult(ok=False)
        result = dispatch({
            "name": "read_file",
            "args": {"path": str(project_root / "ghost.txt")},
        })
        assert result.ok is False
        assert result.error is not None

    def test_duration_ms_recorded(self, project_root: Path) -> None:
        f = project_root / "timed.txt"
        f.write_text("x")
        result = dispatch({"name": "read_file", "args": {"path": str(f)}})
        assert result.duration_ms >= 0

    def test_empty_args_defaults_used(self, project_root: Path) -> None:
        # git_status has no required args — should succeed in a git repo context
        result = dispatch({"name": "git_status", "args": {}})
        # May fail if not a git repo, but should return ToolResult
        assert isinstance(result, ToolResult)

    def test_missing_args_key_handled(self, project_root: Path) -> None:
        # dispatch should handle a tool_call dict with no 'args' key
        result = dispatch({"name": "git_status"})
        assert isinstance(result, ToolResult)

    def test_tool_result_fields(self, project_root: Path) -> None:
        f = project_root / "r.txt"
        f.write_text("data")
        result = dispatch({"name": "read_file", "args": {"path": str(f)}})
        assert hasattr(result, "ok")
        assert hasattr(result, "output")
        assert hasattr(result, "error")
        assert hasattr(result, "duration_ms")

    def test_permission_error_captured(self, project_root: Path) -> None:
        result = dispatch({
            "name": "write_file",
            "args": {"path": "/etc/passwd", "content": "evil"},
        })
        assert result.ok is False
        assert result.error is not None
