from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.tools.tests import run_tests


def _make_completed(returncode: int, stdout: str, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class TestRunTests:
    def test_runs_pytest_and_returns_output(self, project_root: Path) -> None:
        with patch("agent.tools.tests.subprocess.run") as mock_run:
            mock_run.return_value = _make_completed(0, "1 passed in 0.01s")
            result = run_tests.invoke({})
        assert "passed" in result

    def test_failing_test_output_returned(self, project_root: Path) -> None:
        with patch("agent.tools.tests.subprocess.run") as mock_run:
            mock_run.return_value = _make_completed(
                1, "FAILED test_bad.py::test_bad - AssertionError"
            )
            result = run_tests.invoke({"test_path": "test_bad.py"})
        assert "FAILED" in result or "AssertionError" in result

    def test_default_no_path_runs_all(self, project_root: Path) -> None:
        with patch("agent.tools.tests.subprocess.run") as mock_run:
            mock_run.return_value = _make_completed(0, "3 passed in 0.05s")
            result = run_tests.invoke({})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_extra_args_forwarded(self, project_root: Path) -> None:
        with patch("agent.tools.tests.subprocess.run") as mock_run:
            mock_run.return_value = _make_completed(0, "test_v PASSED")
            result = run_tests.invoke({"test_path": "test_verbose.py", "extra_args": "-v"})
        assert "PASSED" in result

    def test_extra_args_split_into_cmd(self, project_root: Path) -> None:
        with patch("agent.tools.tests.subprocess.run") as mock_run:
            mock_run.return_value = _make_completed(0, "ok")
            run_tests.invoke({"extra_args": "-v --tb=long"})
            call_args = mock_run.call_args[0][0]
        assert "-v" in call_args
        assert "--tb=long" in call_args

    def test_returns_no_output_placeholder(self, project_root: Path) -> None:
        with patch("agent.tools.tests.subprocess.run") as mock_run:
            mock_run.return_value = _make_completed(0, "", "")
            result = run_tests.invoke({})
        assert result == "(no output)"

    def test_timeout_parameter_passed(self, project_root: Path) -> None:
        with patch("agent.tools.tests.subprocess.run") as mock_run:
            mock_run.return_value = _make_completed(0, "ok")
            run_tests.invoke({"test_path": "tests/"})
            _, kwargs = mock_run.call_args
        assert kwargs.get("timeout") == 300

    def test_test_path_included_in_cmd(self, project_root: Path) -> None:
        with patch("agent.tools.tests.subprocess.run") as mock_run:
            mock_run.return_value = _make_completed(0, "ok")
            run_tests.invoke({"test_path": "tests/unit/"})
            call_args = mock_run.call_args[0][0]
        assert "tests/unit/" in call_args

    def test_nonzero_exit_output_included(self, project_root: Path) -> None:
        with patch("agent.tools.tests.subprocess.run") as mock_run:
            mock_run.return_value = _make_completed(1, "1 failed", "some stderr")
            result = run_tests.invoke({})
        assert "failed" in result or "stderr" in result
