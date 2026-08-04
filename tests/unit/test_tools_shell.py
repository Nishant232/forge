from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agent.tools.shell import run_shell


class TestRunShellSafeCommands:
    def test_echo_returns_output(self, project_root: Path) -> None:
        result = run_shell.invoke({"command": "echo hello"})
        assert "hello" in result

    def test_exit_code_nonzero_surfaced(self, project_root: Path) -> None:
        result = run_shell.invoke({"command": "exit 1"})
        assert "Exit code 1" in result

    def test_runs_in_project_root(self, project_root: Path) -> None:
        (project_root / "marker.txt").write_text("found")
        result = run_shell.invoke({"command": "ls"})
        assert "marker.txt" in result

    def test_no_output_message(self, project_root: Path) -> None:
        result = run_shell.invoke({"command": "true"})
        assert result == "(no output)"

    def test_stderr_captured(self, project_root: Path) -> None:
        result = run_shell.invoke({"command": "echo err >&2; exit 0"})
        assert "err" in result


class TestRunShellPolicyEnforcement:
    @pytest.mark.parametrize("mode", ["readonly", "trusted", "yolo"])
    def test_rm_rf_slash_always_blocked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
    ) -> None:
        monkeypatch.setenv("AGENT_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("AGENT_TRUST_MODE", mode)
        with pytest.raises(PermissionError, match="blocked by policy"):
            run_shell.invoke({"command": "rm -rf /"})

    @pytest.mark.parametrize("mode", ["readonly", "trusted", "yolo"])
    def test_rm_rf_dot_always_blocked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
    ) -> None:
        monkeypatch.setenv("AGENT_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("AGENT_TRUST_MODE", mode)
        with pytest.raises(PermissionError):
            run_shell.invoke({"command": "rm -rf ."})

    @pytest.mark.parametrize("mode", ["readonly", "trusted", "yolo"])
    def test_dd_always_blocked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
    ) -> None:
        monkeypatch.setenv("AGENT_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("AGENT_TRUST_MODE", mode)
        with pytest.raises(PermissionError):
            run_shell.invoke({"command": "dd if=/dev/zero of=/dev/null"})

    def test_readonly_blocks_write_redirect(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AGENT_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("AGENT_TRUST_MODE", "readonly")
        with pytest.raises(PermissionError):
            run_shell.invoke({"command": "echo hi > file.txt"})

    def test_trusted_allows_write_redirect(self, project_root: Path) -> None:
        result = run_shell.invoke({"command": "echo hi > file.txt && cat file.txt"})
        assert "hi" in result


class TestGetTrustFallback:
    def test_invalid_trust_mode_falls_back_to_trusted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AGENT_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("AGENT_TRUST_MODE", "INVALID_MODE")
        # Should not raise; falls back to trusted, which allows echo
        result = run_shell.invoke({"command": "echo ok"})
        assert "ok" in result


class TestRunShellNetworkPolicy:
    def test_curl_blocked_in_trusted(self, project_root: Path) -> None:
        with pytest.raises(PermissionError, match="blocked"):
            run_shell.invoke({"command": "curl http://example.com"})

    def test_wget_blocked_in_trusted(self, project_root: Path) -> None:
        with pytest.raises(PermissionError, match="blocked"):
            run_shell.invoke({"command": "wget http://example.com"})

    def test_curl_blocked_in_readonly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AGENT_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("AGENT_TRUST_MODE", "readonly")
        with pytest.raises(PermissionError):
            run_shell.invoke({"command": "curl http://example.com"})

    def test_curl_allowed_in_yolo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AGENT_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("AGENT_TRUST_MODE", "yolo")
        # We don't actually run curl; just verify policy does NOT raise PermissionError.
        # Patch subprocess.run so curl doesn't need to be installed.
        import subprocess
        with patch("agent.tools.shell.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="ok", stderr=""
            )
            result = run_shell.invoke({"command": "curl http://example.com"})
            assert "ok" in result
