"""Unit tests for verify/runner.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agent.verify.runner import (
    CheckResult,
    VerifierResult,
    _detect_language,
    _find_test_files,
    run_verifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_check(name: str, passed: bool, output: str = "ok") -> CheckResult:
    return CheckResult(name=name, passed=passed, output=output, duration_ms=10)


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


class TestDetectLanguage:
    def test_python_via_pyproject(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'")
        assert _detect_language(tmp_path) == "python"

    def test_python_via_setup_py(self, tmp_path: Path) -> None:
        (tmp_path / "setup.py").write_text("")
        assert _detect_language(tmp_path) == "python"

    def test_typescript_via_tsconfig(self, tmp_path: Path) -> None:
        (tmp_path / "tsconfig.json").write_text("{}")
        assert _detect_language(tmp_path) == "typescript"

    def test_javascript_via_package_json(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text("{}")
        assert _detect_language(tmp_path) == "javascript"

    def test_rust_via_cargo_toml(self, tmp_path: Path) -> None:
        (tmp_path / "Cargo.toml").write_text("[package]\nname='foo'")
        assert _detect_language(tmp_path) == "rust"

    def test_unknown_for_empty_dir(self, tmp_path: Path) -> None:
        assert _detect_language(tmp_path) == "unknown"

    def test_python_takes_precedence_over_js(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("")
        (tmp_path / "package.json").write_text("{}")
        assert _detect_language(tmp_path) == "python"


# ---------------------------------------------------------------------------
# Test file discovery
# ---------------------------------------------------------------------------


class TestFindTestFiles:
    def test_finds_test_file_for_module(self, tmp_path: Path) -> None:
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_main.py").write_text("")
        result = _find_test_files(["src/main.py"], tmp_path)
        assert any("test_main.py" in r for r in result)

    def test_no_test_file_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / "tests").mkdir()
        result = _find_test_files(["src/obscure_module.py"], tmp_path)
        assert result == []

    def test_deduplicates_results(self, tmp_path: Path) -> None:
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_main.py").write_text("")
        result = _find_test_files(["src/main.py", "src/main.py"], tmp_path)
        assert result.count(str(tests_dir / "test_main.py")) == 1


# ---------------------------------------------------------------------------
# VerifierResult model
# ---------------------------------------------------------------------------


class TestVerifierResult:
    def test_passed_true_when_no_failures(self) -> None:
        vr = VerifierResult(
            passed=True, checks=[], output="ok", failed_checks=[], duration_ms=0
        )
        assert vr.passed is True

    def test_failed_checks_list(self) -> None:
        vr = VerifierResult(
            passed=False,
            checks=[_make_check("ruff", False, "E501 line too long")],
            output="FAIL",
            failed_checks=["ruff"],
            duration_ms=10,
        )
        assert "ruff" in vr.failed_checks

    def test_serialises_to_dict(self) -> None:
        vr = VerifierResult(
            passed=True, checks=[], output="", failed_checks=[], duration_ms=5
        )
        d = vr.model_dump()
        assert "passed" in d
        assert "failed_checks" in d


# ---------------------------------------------------------------------------
# run_verifier — no changed files
# ---------------------------------------------------------------------------


class TestRunVerifierNoPythonFiles:
    def test_no_python_files_returns_pass(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("")
        result = run_verifier(["README.md", "docs/spec.md"], tmp_path)
        assert result.passed is True
        assert result.checks == []


# ---------------------------------------------------------------------------
# run_verifier — Python project, mocked subprocess
# ---------------------------------------------------------------------------


class TestRunVerifierPython:
    def _setup_project(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[tool.ruff]\n")
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_app.py").write_text("")

    def test_ruff_pass_produces_passed_result(self, tmp_path: Path) -> None:
        self._setup_project(tmp_path)
        with (
            patch("agent.verify.runner._has_ruff", return_value=True),
            patch("agent.verify.runner._has_mypy", return_value=False),
            patch("agent.verify.runner._has_pytest", return_value=False),
            patch("agent.verify.runner._run", return_value=(0, "")),
        ):
            result = run_verifier(["app.py"], tmp_path)
        assert result.passed is True
        assert any(c.name == "ruff" for c in result.checks)

    def test_ruff_fail_produces_failed_result(self, tmp_path: Path) -> None:
        self._setup_project(tmp_path)
        with (
            patch("agent.verify.runner._has_ruff", return_value=True),
            patch("agent.verify.runner._has_mypy", return_value=False),
            patch("agent.verify.runner._has_pytest", return_value=False),
            patch("agent.verify.runner._run", return_value=(1, "E501 line too long")),
        ):
            result = run_verifier(["app.py"], tmp_path)
        assert result.passed is False
        assert "ruff" in result.failed_checks

    def test_mypy_fail_included_in_failures(self, tmp_path: Path) -> None:
        self._setup_project(tmp_path)
        def fake_run(cmd, cwd, timeout=60):
            if "mypy" in cmd:
                return (1, "error: Incompatible types")
            return (0, "")

        with (
            patch("agent.verify.runner._has_ruff", return_value=True),
            patch("agent.verify.runner._has_mypy", return_value=True),
            patch("agent.verify.runner._has_pytest", return_value=False),
            patch("agent.verify.runner._run", side_effect=fake_run),
        ):
            result = run_verifier(["app.py"], tmp_path)
        assert "mypy" in result.failed_checks

    def test_pytest_runs_only_related_test_files(self, tmp_path: Path) -> None:
        self._setup_project(tmp_path)
        captured: list[list[str]] = []

        def fake_run(cmd, cwd, timeout=60):
            captured.append(cmd)
            return (0, "1 passed")

        with (
            patch("agent.verify.runner._has_ruff", return_value=False),
            patch("agent.verify.runner._has_mypy", return_value=False),
            patch("agent.verify.runner._has_pytest", return_value=True),
            patch("agent.verify.runner._run", side_effect=fake_run),
        ):
            run_verifier(["app.py"], tmp_path)
        # pytest should only have been invoked for the related test file
        pytest_calls = [c for c in captured if "pytest" in " ".join(c)]
        assert any("test_app.py" in " ".join(c) for c in pytest_calls)

    def test_run_lint_false_skips_ruff(self, tmp_path: Path) -> None:
        self._setup_project(tmp_path)
        with (
            patch("agent.verify.runner._has_ruff", return_value=True),
            patch("agent.verify.runner._has_mypy", return_value=False),
            patch("agent.verify.runner._has_pytest", return_value=False),
            patch("agent.verify.runner._run", return_value=(0, "")),
        ):
            result = run_verifier(["app.py"], tmp_path, run_lint=False)
        assert all(c.name != "ruff" for c in result.checks)

    def test_output_contains_check_names(self, tmp_path: Path) -> None:
        self._setup_project(tmp_path)
        with (
            patch("agent.verify.runner._has_ruff", return_value=True),
            patch("agent.verify.runner._has_mypy", return_value=False),
            patch("agent.verify.runner._has_pytest", return_value=False),
            patch("agent.verify.runner._run", return_value=(0, "")),
        ):
            result = run_verifier(["app.py"], tmp_path)
        assert "ruff" in result.output

    def test_duration_ms_is_positive(self, tmp_path: Path) -> None:
        self._setup_project(tmp_path)
        with (
            patch("agent.verify.runner._has_ruff", return_value=True),
            patch("agent.verify.runner._has_mypy", return_value=False),
            patch("agent.verify.runner._has_pytest", return_value=False),
            patch("agent.verify.runner._run", return_value=(0, "")),
        ):
            result = run_verifier(["app.py"], tmp_path)
        assert result.duration_ms >= 0


# ---------------------------------------------------------------------------
# run_verifier — unknown language
# ---------------------------------------------------------------------------


class TestRunVerifierUnknown:
    def test_unknown_language_passes_trivially(self, tmp_path: Path) -> None:
        result = run_verifier(["somefile.go"], tmp_path)
        assert result.passed is True
        assert result.checks == []
