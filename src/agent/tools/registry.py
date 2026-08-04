from __future__ import annotations

import time
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from agent.tools.files import edit_file, edit_file_multi, find_files, list_dir, read_file, write_file
from agent.tools.git import (
    git_add,
    git_checkpoint,
    git_commit,
    git_diff,
    git_log,
    git_rollback,
    git_status,
)
from agent.tools.search import search_code
from agent.tools.shell import run_shell
from agent.tools.subtask import run_subtask
from agent.tools.tests import run_tests
from agent.tools.todo import todo_read, todo_write
from agent.tools.web import fetch_url, search_web


class ToolResult(BaseModel):
    ok: bool
    output: str | dict[str, Any]
    error: str | None = None
    duration_ms: int


def all_tools() -> list[BaseTool]:
    """Return all registered tools ready for bind_tools()."""
    return [
        # File operations
        read_file,       # type: ignore[list-item]
        write_file,      # type: ignore[list-item]
        edit_file,       # type: ignore[list-item]
        edit_file_multi, # type: ignore[list-item]
        list_dir,        # type: ignore[list-item]
        find_files,      # type: ignore[list-item]
        # Shell & search
        run_shell,       # type: ignore[list-item]
        search_code,     # type: ignore[list-item]
        run_tests,       # type: ignore[list-item]
        # Git
        git_status,      # type: ignore[list-item]
        git_diff,        # type: ignore[list-item]
        git_add,         # type: ignore[list-item]
        git_commit,      # type: ignore[list-item]
        git_log,         # type: ignore[list-item]
        git_checkpoint,  # type: ignore[list-item]
        git_rollback,    # type: ignore[list-item]
        # Web (yolo mode)
        fetch_url,       # type: ignore[list-item]
        search_web,      # type: ignore[list-item]
        # Agent & task
        run_subtask,     # type: ignore[list-item]
        todo_write,      # type: ignore[list-item]
        todo_read,       # type: ignore[list-item]
    ]


def dispatch(tool_call: Any) -> ToolResult:
    """Execute a tool by name. Returns ToolResult(ok=False) for unknown tools.

    Accepts both plain dicts and LangChain ToolCall TypedDicts.
    """
    tool_name: str = tool_call.get("name", "")
    tool_args: dict[str, Any] = tool_call.get("args", {}) or {}

    tool_map: dict[str, BaseTool] = {t.name: t for t in all_tools()}

    if tool_name not in tool_map:
        return ToolResult(
            ok=False, output="", error=f"Unknown tool: {tool_name!r}", duration_ms=0,
        )

    selected = tool_map[tool_name]
    start = time.monotonic()
    try:
        result = selected.invoke(tool_args)
        return ToolResult(ok=True, output=result, duration_ms=int((time.monotonic() - start) * 1000))
    except Exception as exc:
        return ToolResult(ok=False, output="", error=str(exc), duration_ms=int((time.monotonic() - start) * 1000))
