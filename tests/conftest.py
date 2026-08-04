from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Temp directory wired up as AGENT_PROJECT_ROOT with trusted mode."""
    monkeypatch.setenv("AGENT_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_TRUST_MODE", "trusted")
    return tmp_path


@pytest.fixture
def readonly_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AGENT_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_TRUST_MODE", "readonly")
    return tmp_path


@pytest.fixture
def yolo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AGENT_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENT_TRUST_MODE", "yolo")
    return tmp_path


@pytest.fixture
def git_repo(project_root: Path) -> Path:
    """Temp directory initialized as a git repo on a non-protected branch."""
    subprocess.run(["git", "init"], cwd=str(project_root), check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(project_root), check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=str(project_root), check=True, capture_output=True,
    )
    # Create an initial commit so HEAD exists, then branch off to avoid protected branch
    (project_root / "init.txt").write_text("init")
    subprocess.run(["git", "add", "."], cwd=str(project_root), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=str(project_root), check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "-b", "agent/work"],
        cwd=str(project_root), check=True, capture_output=True,
    )
    return project_root
