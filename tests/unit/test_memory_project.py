from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.memory.project import ProjectStore


@pytest.fixture()
def store(tmp_path: Path) -> ProjectStore:
    return ProjectStore(tmp_path / "state.db")


class TestSetAndGet:
    def test_set_and_get_round_trips(self, store: ProjectStore) -> None:
        store.set("test_runner", "pytest")
        assert store.get("test_runner") == "pytest"

    def test_get_missing_key_returns_none(self, store: ProjectStore) -> None:
        assert store.get("nonexistent") is None

    def test_set_overwrites_existing(self, store: ProjectStore) -> None:
        store.set("linter", "ruff")
        store.set("linter", "flake8")
        assert store.get("linter") == "flake8"

    def test_set_many_inserts_all(self, store: ProjectStore) -> None:
        store.set_many({"a": "1", "b": "2", "c": "3"})
        assert store.get("a") == "1"
        assert store.get("b") == "2"
        assert store.get("c") == "3"

    def test_delete_removes_key(self, store: ProjectStore) -> None:
        store.set("ephemeral", "yes")
        store.delete("ephemeral")
        assert store.get("ephemeral") is None

    def test_delete_nonexistent_is_noop(self, store: ProjectStore) -> None:
        store.delete("ghost")  # should not raise


class TestAll:
    def test_all_returns_all_facts(self, store: ProjectStore) -> None:
        store.set("test_runner", "pytest", source="manual")
        store.set("linter", "ruff", source="detected:pyproject.toml")
        facts = store.all()
        assert len(facts) == 2
        keys = {f.key for f in facts}
        assert keys == {"test_runner", "linter"}

    def test_source_preserved(self, store: ProjectStore) -> None:
        store.set("k", "v", source="detected:Cargo.toml")
        facts = store.all()
        assert facts[0].source == "detected:Cargo.toml"


class TestDetectPyproject:
    def test_detects_pytest_test_runner(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            "[tool.pytest.ini_options]\ntestpaths = ['tests']\n"
        )
        store = ProjectStore(tmp_path / "state.db")
        store.detect_and_persist(tmp_path)
        assert store.get("test_runner") == "pytest"

    def test_detects_ruff_linter(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            "[tool.ruff]\nline-length = 100\n"
        )
        store = ProjectStore(tmp_path / "state.db")
        store.detect_and_persist(tmp_path)
        assert store.get("linter") == "ruff"

    def test_detects_mypy_type_checker(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            "[tool.mypy]\nstrict = true\n"
        )
        store = ProjectStore(tmp_path / "state.db")
        store.detect_and_persist(tmp_path)
        assert store.get("type_checker") == "mypy"

    def test_detects_build_backend(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[build-system]\nbuild-backend = "hatchling.build"\n'
        )
        store = ProjectStore(tmp_path / "state.db")
        store.detect_and_persist(tmp_path)
        assert store.get("build_backend") == "hatchling.build"


class TestDetectPackageJson:
    def test_detects_npm_test(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text(
            json.dumps({"scripts": {"test": "jest", "build": "tsc"}})
        )
        store = ProjectStore(tmp_path / "state.db")
        store.detect_and_persist(tmp_path)
        assert store.get("test_runner") == "jest"
        assert store.get("build_command") == "tsc"

    def test_detects_eslint_from_devdeps(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text(
            json.dumps({"devDependencies": {"eslint": "^8.0.0"}})
        )
        store = ProjectStore(tmp_path / "state.db")
        store.detect_and_persist(tmp_path)
        assert store.get("linter") == "eslint"


class TestDetectCargo:
    def test_detects_cargo_tools(self, tmp_path: Path) -> None:
        (tmp_path / "Cargo.toml").write_text(
            '[package]\nname = "my-crate"\nversion = "0.1.0"\n'
        )
        store = ProjectStore(tmp_path / "state.db")
        store.detect_and_persist(tmp_path)
        assert store.get("test_runner") == "cargo test"
        assert store.get("linter") == "cargo clippy"
        assert store.get("build_command") == "cargo build"


class TestDetectPrecommit:
    def test_detects_precommit_presence(self, tmp_path: Path) -> None:
        (tmp_path / ".pre-commit-config.yaml").write_text("repos: []\n")
        store = ProjectStore(tmp_path / "state.db")
        store.detect_and_persist(tmp_path)
        assert store.get("pre_commit") == "true"


class TestManualPrecedence:
    def test_manual_fact_not_overwritten_by_detection(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[tool.ruff]\n")
        store = ProjectStore(tmp_path / "state.db")
        store.set("linter", "flake8", source="manual")  # manual override
        store.detect_and_persist(tmp_path)
        # Detection should NOT overwrite the manual setting
        assert store.get("linter") == "flake8"

    def test_non_manual_fact_is_overwritten(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[tool.ruff]\n")
        store = ProjectStore(tmp_path / "state.db")
        store.set("linter", "old-ruff", source="detected:pyproject.toml")
        store.detect_and_persist(tmp_path)
        assert store.get("linter") == "ruff"
