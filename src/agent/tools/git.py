from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from langchain_core.tools import tool

_PROTECTED_BRANCHES: frozenset[str] = frozenset(
    ["main", "master", "develop", "production", "release"]
)


def _project_root() -> Path:
    root = os.environ.get("AGENT_PROJECT_ROOT")
    return Path(root).resolve() if root else Path.cwd().resolve()


def _run_git(args: list[str], cwd: Path | None = None) -> tuple[int, str]:
    result = subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(cwd or _project_root()),
    )
    return result.returncode, (result.stdout + result.stderr).strip()


def _current_branch(cwd: Path) -> str:
    _, output = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    return output.strip()


@tool  # type: ignore[misc]
def git_status() -> str:
    """Show the short working-tree status of the project repository."""
    code, output = _run_git(["status", "--short"])
    if code != 0:
        raise RuntimeError(f"git status failed: {output}")
    return output or "(clean working tree)"


@tool  # type: ignore[misc]
def git_diff(path: str = "") -> str:
    """Show unstaged changes in the repository, optionally filtered to a path."""
    args = ["diff"]
    if path:
        args += ["--", path]
    code, output = _run_git(args)
    if code != 0:
        raise RuntimeError(f"git diff failed: {output}")
    return output or "(no changes)"


@tool  # type: ignore[misc]
def git_add(path: str) -> str:
    """Stage a file or directory for the next commit."""
    code, output = _run_git(["add", path])
    if code != 0:
        raise RuntimeError(f"git add failed: {output}")
    return f"Staged: {path!r}"


@tool  # type: ignore[misc]
def git_commit(message: str) -> str:
    """Create a commit with the given message.

    Blocked on protected branches (main, master, develop, production, release).
    Staged changes must exist before calling this.
    """
    root = _project_root()
    branch = _current_branch(root)

    if branch in _PROTECTED_BRANCHES:
        raise PermissionError(
            f"Direct commits to protected branch {branch!r} are blocked. "
            "The agent must work on a feature branch."
        )

    code, output = _run_git(["commit", "-m", message])
    if code != 0:
        raise RuntimeError(f"git commit failed: {output}")
    return output


@tool  # type: ignore[misc]
def git_log(n: int = 10) -> str:
    """Show the last N commits as a one-line summary (default 10)."""
    _, output = _run_git(["log", f"-{n}", "--oneline"])
    return output or "(no commits)"


@tool  # type: ignore[misc]
def git_checkpoint(label: str = "") -> str:
    """Create a lightweight git tag as a pre-edit rollback point.

    This is a git checkpoint for task rollback — entirely separate from the
    LangGraph checkpoints used for resumability. The tag name is recorded in
    .agent/checkpoints/index.txt for later lookup by git_rollback.
    """
    root = _project_root()
    checkpoints_dir = root / ".agent" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time())
    tag_name = f"agent-cp-{label}-{ts}" if label else f"agent-cp-{ts}"

    code, output = _run_git(["tag", tag_name])
    if code != 0:
        raise RuntimeError(f"Failed to create checkpoint tag: {output}")

    index_file = checkpoints_dir / "index.txt"
    with open(index_file, "a", encoding="utf-8") as f:
        f.write(f"{tag_name}\n")

    return f"Checkpoint created: {tag_name}"


@tool  # type: ignore[misc]
def git_rollback(checkpoint_tag: str = "") -> str:
    """Roll back the working tree to a previous agent checkpoint tag.

    If checkpoint_tag is empty, the most recent agent checkpoint is used.
    WARNING: this discards all uncommitted changes in the working tree.
    """
    if not checkpoint_tag:
        code, output = _run_git(
            ["tag", "--list", "agent-cp-*", "--sort=-version:refname"]
        )
        if code != 0 or not output.strip():
            raise RuntimeError("No agent checkpoints found.")
        checkpoint_tag = output.strip().splitlines()[0]

    code, output = _run_git(["checkout", checkpoint_tag, "--", "."])
    if code != 0:
        raise RuntimeError(f"Rollback to {checkpoint_tag!r} failed: {output}")
    return f"Rolled back to checkpoint: {checkpoint_tag}"
