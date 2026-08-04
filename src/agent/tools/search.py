from __future__ import annotations

import os
import subprocess
from pathlib import Path

from langchain_core.tools import tool

# Symbol and vector search are implemented in Phase 2 (code_index/).
# This module covers text search via ripgrep only for Phase 1.


def _project_root() -> Path:
    root = os.environ.get("AGENT_PROJECT_ROOT")
    return Path(root).resolve() if root else Path.cwd().resolve()


@tool  # type: ignore[misc]
def search_code(pattern: str, file_glob: str = "", max_results: int = 50) -> str:
    """Search for a text pattern across the project using ripgrep.

    Args:
        pattern: Literal string or regex to search for.
        file_glob: Optional glob to restrict which files are searched (e.g. '*.py').
        max_results: Maximum number of result lines to return (default 50).

    Returns ripgrep output with path:line:content format, or 'No matches found.'
    Requires `rg` (ripgrep) to be installed and on PATH.
    """
    project_root = _project_root()

    cmd: list[str] = ["rg", "--line-number", "--no-heading", "--color=never"]
    if file_glob:
        cmd += ["--glob", file_glob]
    cmd += [pattern, str(project_root)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(project_root),
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ripgrep (rg) not found on PATH. "
            "Install it from https://github.com/BurntSushi/ripgrep"
        )

    lines = result.stdout.strip().splitlines()
    if not lines:
        return "No matches found."

    truncated = False
    if len(lines) > max_results:
        lines = lines[:max_results]
        truncated = True

    output = "\n".join(lines)
    if truncated:
        output += f"\n... (truncated to {max_results} results)"
    return output
