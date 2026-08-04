from __future__ import annotations

import os
import subprocess
from pathlib import Path

from langchain_core.tools import tool


def _project_root() -> Path:
    root = os.environ.get("AGENT_PROJECT_ROOT")
    return Path(root).resolve() if root else Path.cwd().resolve()


@tool  # type: ignore[misc]
def run_tests(test_path: str = "", extra_args: str = "") -> str:
    """Run the project test suite with pytest.

    Args:
        test_path: Specific test file or directory to run. Empty means all tests.
        extra_args: Additional pytest flags, space-separated (e.g. '-v -k test_foo').

    Returns combined pytest output including the pass/fail summary.
    Times out after 300 seconds.
    """
    project_root = _project_root()

    cmd: list[str] = ["python", "-m", "pytest"]
    if test_path:
        cmd.append(test_path)
    if extra_args:
        cmd.extend(extra_args.split())
    cmd += ["--tb=short", "--no-header"]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(project_root),
    )

    output = (result.stdout + result.stderr).strip()
    return output or "(no output)"
