from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from agent.safety.policy import is_path_allowed


def _project_root() -> Path:
    root = os.environ.get("AGENT_PROJECT_ROOT")
    return Path(root).resolve() if root else Path.cwd().resolve()


def _validated_path(path: str) -> Path:
    """Resolve path and verify it lives inside the project root."""
    root = _project_root()
    # Handle relative paths by resolving against project root first
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    allowed, reason = is_path_allowed(p, root)
    if not allowed:
        raise PermissionError(reason)
    return p.resolve()


@tool  # type: ignore[misc]
def read_file(path: str, start_line: int = 1, end_line: Optional[int] = None) -> str:
    """Read UTF-8 contents of a file, optionally limited to a line range.

    Args:
        path: File path within the project root.
        start_line: First line to return (1-indexed, default 1 = beginning).
        end_line: Last line to return inclusive (default None = read to end).

    Returns:
        File contents with line numbers prefixed when a range is specified,
        or raw contents when reading the whole file.

    Examples:
        read_file("src/app.py")              # full file
        read_file("src/app.py", 50, 100)     # lines 50-100 with line numbers
        read_file("src/app.py", 200)         # from line 200 to end
    """
    p = _validated_path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path!r}")
    if not p.is_file():
        raise IsADirectoryError(f"Path is a directory: {path!r}")

    raw = p.read_text(encoding="utf-8")

    # Full read
    if start_line == 1 and end_line is None:
        return raw

    lines = raw.splitlines(keepends=True)
    total = len(lines)

    s = max(0, start_line - 1)
    e = end_line if end_line is not None else total

    if s >= total:
        return f"(start_line {start_line} exceeds file length {total})"

    slice_ = lines[s:e]
    # Return with line numbers so the model can reference them precisely
    return "".join(f"{s + i + 1}: {ln}" for i, ln in enumerate(slice_))


@tool  # type: ignore[misc]
def write_file(path: str, content: str) -> str:
    """Write UTF-8 content to a file, creating parent directories as needed.

    Overwrites the file if it already exists.
    The path must be within the project root.
    """
    p = _validated_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path!r}"


@tool  # type: ignore[misc]
def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Replace exactly one occurrence of old_string with new_string in a file.

    Raises ValueError if old_string is not found or appears more than once —
    provide a longer, more distinctive search string in the latter case.
    The path must be within the project root.
    """
    p = _validated_path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path!r}")

    content = p.read_text(encoding="utf-8")
    count = content.count(old_string)

    if count == 0:
        raise ValueError(
            f"Search string not found in {path!r}.\n"
            f"Search string was: {old_string!r}"
        )
    if count > 1:
        raise ValueError(
            f"Search string appears {count} times in {path!r}. "
            "Provide a longer, more distinctive search string."
        )

    p.write_text(content.replace(old_string, new_string, 1), encoding="utf-8")
    return f"Replaced 1 occurrence in {path!r}"


@tool  # type: ignore[misc]
def edit_file_multi(path: str, old_strings: list[str], new_strings: list[str]) -> str:
    """Apply multiple non-overlapping edits to a file in a single call.

    More efficient than calling edit_file repeatedly when making several
    changes to the same file. Edits are applied in the order given.

    Args:
        path: File path within the project root.
        old_strings: List of exact strings to find (each must appear exactly once).
        new_strings: Corresponding replacements (must have the same length).

    Returns:
        Summary of replacements made.

    Example:
        edit_file_multi(
            "src/config.py",
            old_strings=["HOST = 'localhost'", "PORT = 8000"],
            new_strings=["HOST = '0.0.0.0'",  "PORT = 9000"],
        )
    """
    if len(old_strings) != len(new_strings):
        raise ValueError(
            f"old_strings ({len(old_strings)}) and new_strings ({len(new_strings)}) "
            "must have the same length."
        )
    if not old_strings:
        return "No edits specified."

    p = _validated_path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path!r}")

    content = p.read_text(encoding="utf-8")

    for i, (old, new) in enumerate(zip(old_strings, new_strings), 1):
        count = content.count(old)
        if count == 0:
            raise ValueError(
                f"Edit {i}: search string not found in {path!r}: {old[:60]!r}"
            )
        if count > 1:
            raise ValueError(
                f"Edit {i}: search string appears {count} times in {path!r}. "
                f"Make it more distinctive: {old[:60]!r}"
            )
        content = content.replace(old, new, 1)

    p.write_text(content, encoding="utf-8")
    return f"Applied {len(old_strings)} edit(s) to {path!r}"


@tool  # type: ignore[misc]
def list_dir(path: str) -> str:
    """List the entries in a directory.

    Returns one line per entry, prefixed with [F] for files and [D] for
    directories. The path must be within the project root.
    """
    p = _validated_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {path!r}")
    if not p.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path!r}")

    entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name))
    if not entries:
        return "(empty directory)"

    lines = [f"[{'F' if e.is_file() else 'D'}] {e.name}" for e in entries]
    return "\n".join(lines)


@tool  # type: ignore[misc]
def find_files(pattern: str, path: str = ".") -> str:
    """Find files matching a glob pattern within the project.

    Args:
        pattern: Glob pattern relative to `path`.
                 Use ** for recursive matching: "**/*.py", "src/**/*.ts"
                 Without **: "*.py" matches only the top level of `path`.
        path: Directory to search from (relative to project root, default ".").

    Returns:
        Newline-separated list of matching paths relative to the project root.
        Returns a message if nothing matches or the count exceeds 500.

    Examples:
        find_files("**/*.py")          # all Python files in the project
        find_files("test_*.py", "tests/unit")  # unit test files only
        find_files("*.toml")           # TOML files in the project root
    """
    root = _project_root()

    if path == ".":
        search_dir = root
    else:
        search_dir = _validated_path(path)
        if not search_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {path!r}")

    try:
        matches = sorted(p for p in search_dir.glob(pattern) if p.is_file())
    except Exception as exc:
        raise ValueError(f"Invalid glob pattern {pattern!r}: {exc}") from exc

    if not matches:
        return f"No files matching {pattern!r} under {path!r}"

    lines = []
    for m in matches[:500]:
        try:
            lines.append(str(m.relative_to(root)))
        except ValueError:
            lines.append(str(m))

    result = "\n".join(lines)
    if len(matches) > 500:
        result += f"\n[...{len(matches) - 500} more files not shown — narrow the pattern]"
    return result
