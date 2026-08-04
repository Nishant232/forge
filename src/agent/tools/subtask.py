"""Sub-task tool: delegate a focused piece of work to a fresh sub-agent."""
from __future__ import annotations

import asyncio
import os
import tempfile
import typing
from pathlib import Path

from langchain_core.tools import tool


@tool  # type: ignore[misc]
def run_subtask(description: str) -> str:
    """Spawn a focused sub-agent to handle a well-scoped sub-task.

    Use this to delegate a specific, self-contained piece of work to a fresh
    agent context. The sub-agent has access to all tools but starts with a
    clean conversation history, so it won't be confused by the parent context.

    Good uses:
    - "Read src/auth/middleware.py and write unit tests in tests/unit/test_auth.py"
    - "Refactor src/db/ to use connection pooling without changing the public API"
    - "Find all usages of old_function and replace with new_function"

    Args:
        description: Clear, self-contained description of the sub-task including
                     relevant file paths, expected outcome, and any constraints.

    Returns:
        The sub-agent's final response summarising what was accomplished.
    """
    # Prevent sub-tasks from spawning their own sub-tasks (one level only)
    if os.environ.get("_AGENT_IN_SUBTASK") == "1":
        return (
            "Sub-task nesting is not supported. "
            "Complete this work directly using the available tools."
        )

    workspace = Path(os.environ.get("AGENT_PROJECT_ROOT", ".")).resolve()
    trust = os.environ.get("AGENT_TRUST_MODE", "trusted")
    if trust not in ("readonly", "trusted", "yolo"):
        trust = "trusted"

    # Propagate env vars into the sub-task
    os.environ["_AGENT_IN_SUBTASK"] = "1"

    # Use a temp database so sub-task history is isolated.
    # mkstemp is used instead of NamedTemporaryFile because Windows does not
    # allow a NamedTemporaryFile to be opened a second time while it is still open.
    fd, tmp_name = tempfile.mkstemp(suffix=".db")
    os.close(fd)  # Close the OS-level file descriptor; SQLite will open it fresh.
    db_path = Path(tmp_name)

    try:
        from agent.core.history import ConversationHistory
        from agent.core.loop import run_turn

        history = ConversationHistory(db_path)
        session_id = history.create_session(
            str(workspace), title=f"subtask: {description[:50]}"
        )

        # asyncio.run() is safe here because we are inside asyncio.to_thread(),
        # which runs in a worker thread with no existing event loop.
        result = asyncio.run(
            run_turn(
                description,
                workspace=workspace,
                session_id=session_id,
                history=history,
                trust=typing.cast(typing.Literal["readonly", "trusted", "yolo"], trust),
            )
        )
        history.close()

        summary = result.text or "(sub-task completed with no text output)"
        if result.files_changed:
            summary += f"\n\nFiles changed: {', '.join(result.files_changed)}"
        if result.tool_calls_made:
            summary += f"\n({result.tool_calls_made} tool calls, {result.iterations} iterations)"
        return summary

    except Exception as exc:
        return f"Sub-task failed: {exc}"
    finally:
        os.environ.pop("_AGENT_IN_SUBTASK", None)
        try:
            db_path.unlink()
        except Exception:
            pass
