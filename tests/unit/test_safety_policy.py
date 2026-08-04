from __future__ import annotations

from pathlib import Path

import pytest

from agent.safety.policy import (
    TrustMode,
    is_network_allowed,
    is_path_allowed,
    is_shell_allowed,
)

# ---------------------------------------------------------------------------
# is_shell_allowed — always-deny patterns
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["readonly", "trusted", "yolo"])
def test_rm_rf_slash_denied_all_modes(mode: TrustMode) -> None:
    allowed, reason = is_shell_allowed("rm -rf /", mode)
    assert not allowed
    assert reason


@pytest.mark.parametrize("mode", ["readonly", "trusted", "yolo"])
def test_rm_rf_dot_denied_all_modes(mode: TrustMode) -> None:
    allowed, _ = is_shell_allowed("rm -rf .", mode)
    assert not allowed


@pytest.mark.parametrize("mode", ["readonly", "trusted", "yolo"])
def test_rm_fr_variant_denied(mode: TrustMode) -> None:
    allowed, _ = is_shell_allowed("rm -fr /tmp", mode)
    assert not allowed


@pytest.mark.parametrize("mode", ["readonly", "trusted", "yolo"])
def test_rm_rvf_denied(mode: TrustMode) -> None:
    # -rvf contains r and f → denied
    allowed, _ = is_shell_allowed("rm -rvf /home", mode)
    assert not allowed


@pytest.mark.parametrize("cmd,mode", [
    ("dd if=/dev/zero of=/dev/sda", "trusted"),
    ("dd if=/dev/zero of=/dev/sda", "yolo"),
    ("mkfs.ext4 /dev/sdb", "trusted"),
    ("shred /dev/sdc", "trusted"),
    ("fdisk /dev/sda", "trusted"),
    ("parted /dev/sda", "trusted"),
    ("wipefs -a /dev/sda", "trusted"),
])
def test_dangerous_commands_denied(cmd: str, mode: TrustMode) -> None:
    allowed, reason = is_shell_allowed(cmd, mode)
    assert not allowed
    assert reason


def test_write_to_dev_denied() -> None:
    allowed, _ = is_shell_allowed("echo evil > /dev/sda", "trusted")
    assert not allowed


# ---------------------------------------------------------------------------
# is_shell_allowed — readonly-specific blocks
# ---------------------------------------------------------------------------

def test_write_redirect_blocked_readonly() -> None:
    allowed, reason = is_shell_allowed("echo hello > output.txt", "readonly")
    assert not allowed
    assert "readonly" in reason.lower() or reason


def test_git_commit_blocked_readonly() -> None:
    allowed, _ = is_shell_allowed("git commit -m 'msg'", "readonly")
    assert not allowed


def test_git_push_blocked_readonly() -> None:
    allowed, _ = is_shell_allowed("git push origin main", "readonly")
    assert not allowed


# ---------------------------------------------------------------------------
# is_shell_allowed — safe commands
# ---------------------------------------------------------------------------

def test_echo_allowed_trusted() -> None:
    allowed, reason = is_shell_allowed("echo hello", "trusted")
    assert allowed
    assert reason == ""


def test_ls_allowed_all_modes() -> None:
    for mode in ("readonly", "trusted", "yolo"):
        allowed, _ = is_shell_allowed("ls -la", mode)  # type: ignore[arg-type]
        assert allowed


def test_python_run_allowed_trusted() -> None:
    allowed, _ = is_shell_allowed("python -m pytest tests/", "trusted")
    assert allowed


def test_cat_allowed_readonly() -> None:
    allowed, _ = is_shell_allowed("cat file.txt", "readonly")
    assert allowed


# ---------------------------------------------------------------------------
# is_path_allowed
# ---------------------------------------------------------------------------

def test_path_inside_root_allowed(tmp_path: Path) -> None:
    allowed, reason = is_path_allowed(str(tmp_path / "src" / "file.py"), tmp_path)
    assert allowed
    assert reason == ""


def test_path_at_root_allowed(tmp_path: Path) -> None:
    allowed, _ = is_path_allowed(str(tmp_path), tmp_path)
    assert allowed


def test_path_outside_root_denied(tmp_path: Path) -> None:
    allowed, reason = is_path_allowed("/etc/passwd", tmp_path)
    assert not allowed
    assert "/etc/passwd" in reason or "outside" in reason


def test_path_parent_traversal_denied(tmp_path: Path) -> None:
    # ../sibling should be outside root
    allowed, _ = is_path_allowed(str(tmp_path.parent / "other"), tmp_path)
    assert not allowed


def test_path_object_accepted(tmp_path: Path) -> None:
    allowed, _ = is_path_allowed(tmp_path / "sub" / "file.txt", tmp_path)
    assert allowed


# ---------------------------------------------------------------------------
# is_network_allowed
# ---------------------------------------------------------------------------

def test_network_blocked_readonly() -> None:
    allowed, reason = is_network_allowed("readonly")
    assert not allowed
    assert reason


def test_network_blocked_trusted() -> None:
    allowed, _ = is_network_allowed("trusted")
    assert not allowed


def test_network_allowed_yolo() -> None:
    allowed, reason = is_network_allowed("yolo")
    assert allowed
    assert reason == ""
