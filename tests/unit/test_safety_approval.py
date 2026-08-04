from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.safety.approval import AutoApprove, CLIApprove


class TestAutoApprove:
    def test_always_returns_true(self) -> None:
        gate = AutoApprove()
        assert gate.request("delete_file", "Deleting /tmp/x") is True

    def test_returns_true_regardless_of_action(self) -> None:
        gate = AutoApprove()
        assert gate.request("run_shell", "rm -rf /tmp/safe") is True
        assert gate.request("", "") is True


class TestCLIApprove:
    def test_yes_approved(self) -> None:
        gate = CLIApprove()
        with patch("builtins.input", return_value="y"):
            assert gate.request("write_file", "Writing to output.txt") is True

    def test_yes_full_word_approved(self) -> None:
        gate = CLIApprove()
        with patch("builtins.input", return_value="yes"):
            assert gate.request("write_file", "Writing to output.txt") is True

    def test_no_denied(self) -> None:
        gate = CLIApprove()
        with patch("builtins.input", return_value="n"):
            assert gate.request("write_file", "Writing to output.txt") is False

    def test_empty_denied(self) -> None:
        gate = CLIApprove()
        with patch("builtins.input", return_value=""):
            assert gate.request("write_file", "Writing to output.txt") is False

    def test_random_input_denied(self) -> None:
        gate = CLIApprove()
        with patch("builtins.input", return_value="maybe"):
            assert gate.request("run_shell", "echo hi") is False

    def test_eof_denied(self) -> None:
        gate = CLIApprove()
        with patch("builtins.input", side_effect=EOFError):
            assert gate.request("run_shell", "echo hi") is False

    def test_prints_action_and_description(self, capsys: pytest.CaptureFixture[str]) -> None:
        gate = CLIApprove()
        with patch("builtins.input", return_value="n"):
            gate.request("my_action", "some description")
        captured = capsys.readouterr()
        assert "my_action" in captured.out
        assert "some description" in captured.out
