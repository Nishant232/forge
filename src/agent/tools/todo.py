"""Per-session todo list — track progress on complex multi-step tasks."""
from __future__ import annotations

import json
import os
from pathlib import Path

from langchain_core.tools import tool


def _todo_path() -> Path:
    root = Path(os.environ.get("AGENT_PROJECT_ROOT", ".")).resolve()
    todo_dir = root / ".agent" / "todos"
    todo_dir.mkdir(parents=True, exist_ok=True)
    session_id = os.environ.get("AGENT_SESSION_ID", "current")
    return todo_dir / f"{session_id[:8]}.json"


@tool  # type: ignore[misc]
def todo_write(items: list[str]) -> str:
    """Set (replace) the session todo list for tracking a complex task.

    Use at the start of multi-step tasks to outline what you need to do,
    and call again with updated items to mark steps complete.
    Prefix completed items with '[done] ' to mark them finished.

    Args:
        items: Ordered list of task strings.
               Mark done items with the '[done] ' prefix.

    Returns:
        Confirmation with pending/done counts.

    Example:
        todo_write([
            "Read src/auth/middleware.py to understand current logic",
            "Write unit tests in tests/unit/test_auth.py",
            "[done] Run pytest to confirm baseline",
            "Refactor validate_token to handle expired tokens",
        ])
    """
    path = _todo_path()
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    done = sum(1 for i in items if i.strip().lower().startswith("[done]"))
    pending = len(items) - done
    return f"Todo list saved: {pending} pending, {done} done ({len(items)} total)"


@tool  # type: ignore[misc]
def todo_read() -> str:
    """Read the current session todo list.

    Returns a formatted checklist showing pending (○) and done (✓) items.
    Returns a message if no todo list exists yet.
    """
    path = _todo_path()
    if not path.exists():
        return "(no todo list — use todo_write to create one)"

    try:
        items: list[str] = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "(todo list is corrupted — use todo_write to reset it)"

    if not items:
        return "(todo list is empty)"

    lines = []
    for i, item in enumerate(items, 1):
        done = item.strip().lower().startswith("[done]")
        marker = "✓" if done else "○"
        text = item.strip()[6:].strip() if done else item.strip()
        lines.append(f"  {marker} {i}. {text}")

    done_count = sum(1 for i in items if i.strip().lower().startswith("[done]"))
    header = f"Todo ({done_count}/{len(items)} done):"
    return header + "\n" + "\n".join(lines)
