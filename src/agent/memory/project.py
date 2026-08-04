"""Project memory: stable facts about the project being worked on.

Stores build commands, test runner, linter, type checker, and any conventions
the agent has learned. Facts are auto-detected by inspecting well-known config
files on first run, then persisted to SQLite so they survive restarts.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS project_facts (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    source      TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ProjectFact:
    key: str
    value: str
    source: str   # e.g. "detected:pyproject.toml", "manual", "learned"
    updated_at: datetime


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class ProjectStore:
    """SQLite-backed store for stable project facts."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def set(self, key: str, value: str, source: str = "manual") -> None:
        """Insert or replace a project fact."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO project_facts (key, value, source, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (key, value, source, now),
        )
        self._conn.commit()

    def set_many(self, facts: dict[str, str], source: str = "manual") -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.executemany(
            "INSERT OR REPLACE INTO project_facts (key, value, source, updated_at) VALUES (?, ?, ?, ?)",
            [(k, v, source, now) for k, v in facts.items()],
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM project_facts WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def all(self) -> list[ProjectFact]:
        rows = self._conn.execute(
            "SELECT * FROM project_facts ORDER BY key"
        ).fetchall()
        return [
            ProjectFact(
                key=r["key"],
                value=r["value"],
                source=r["source"],
                updated_at=datetime.fromisoformat(r["updated_at"]),
            )
            for r in rows
        ]

    def delete(self, key: str) -> None:
        self._conn.execute("DELETE FROM project_facts WHERE key = ?", (key,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Auto-detection
    # ------------------------------------------------------------------

    def detect_and_persist(self, project_root: Path) -> list[ProjectFact]:
        """Inspect well-known config files and persist detected conventions.

        Never overwrites a fact whose source is "manual" — manual settings
        always take precedence over auto-detection.
        """
        detected: dict[str, tuple[str, str]] = {}  # key → (value, source)

        detected.update(_detect_pyproject(project_root))
        detected.update(_detect_package_json(project_root))
        detected.update(_detect_cargo(project_root))
        detected.update(_detect_precommit(project_root))

        manual_keys = {
            r["key"]
            for r in self._conn.execute(
                "SELECT key FROM project_facts WHERE source = 'manual'"
            ).fetchall()
        }

        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (k, v, src, now)
            for k, (v, src) in detected.items()
            if k not in manual_keys
        ]
        if rows:
            self._conn.executemany(
                "INSERT OR REPLACE INTO project_facts (key, value, source, updated_at) VALUES (?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()

        return self.all()


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def _detect_pyproject(root: Path) -> dict[str, tuple[str, str]]:
    path = root / "pyproject.toml"
    if not path.exists():
        return {}

    src = "detected:pyproject.toml"
    facts: dict[str, tuple[str, str]] = {}

    try:
        import tomllib
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _LOG.debug("Could not parse pyproject.toml: %s", exc)
        return {}

    tool = data.get("tool", {})

    if "pytest" in tool or "pytest" in data.get("project", {}).get(
        "optional-dependencies", {}
    ):
        facts["test_runner"] = ("pytest", src)

    if "ruff" in tool:
        facts["linter"] = ("ruff", src)

    if "mypy" in tool:
        facts["type_checker"] = ("mypy", src)

    # Build backend
    build = data.get("build-system", {})
    backend = build.get("build-backend", "")
    if backend:
        facts["build_backend"] = (backend, src)

    # Detect test runner from dev dependencies if not already set
    if "test_runner" not in facts:
        for group in data.get("project", {}).get("optional-dependencies", {}).values():
            if any("pytest" in dep for dep in group):
                facts["test_runner"] = ("pytest", src)
                break

    # Also check direct dependencies list
    if "test_runner" not in facts:
        all_deps = data.get("project", {}).get("dependencies", [])
        if any("pytest" in d for d in all_deps):
            facts["test_runner"] = ("pytest", src)

    return facts


def _detect_package_json(root: Path) -> dict[str, tuple[str, str]]:
    path = root / "package.json"
    if not path.exists():
        return {}

    src = "detected:package.json"
    facts: dict[str, tuple[str, str]] = {}

    try:
        import json as _json
        data = _json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _LOG.debug("Could not parse package.json: %s", exc)
        return {}

    scripts = data.get("scripts", {})
    if "test" in scripts:
        facts["test_runner"] = (scripts["test"], src)
    if "lint" in scripts:
        facts["linter"] = (scripts["lint"], src)
    if "build" in scripts:
        facts["build_command"] = (scripts["build"], src)

    dev_deps = data.get("devDependencies", {})
    if "eslint" in dev_deps and "linter" not in facts:
        facts["linter"] = ("eslint", src)
    if "jest" in dev_deps and "test_runner" not in facts:
        facts["test_runner"] = ("jest", src)
    if "vitest" in dev_deps and "test_runner" not in facts:
        facts["test_runner"] = ("vitest", src)

    return facts


def _detect_cargo(root: Path) -> dict[str, tuple[str, str]]:
    path = root / "Cargo.toml"
    if not path.exists():
        return {}

    src = "detected:Cargo.toml"
    return {
        "test_runner": ("cargo test", src),
        "build_command": ("cargo build", src),
        "linter": ("cargo clippy", src),
    }


def _detect_precommit(root: Path) -> dict[str, tuple[str, str]]:
    path = root / ".pre-commit-config.yaml"
    if not path.exists():
        return {}
    return {"pre_commit": ("true", "detected:.pre-commit-config.yaml")}
