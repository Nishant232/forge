from __future__ import annotations

import fnmatch
import re
import tomllib
from pathlib import Path
from typing import Literal

TrustMode = Literal["readonly", "trusted", "yolo"]

# Permanently denied in every trust mode (cannot be overridden by allowlist).
# Note: bare `rm` is handled separately by _is_dangerous_rm() which also requires
# recursive+force flags (-rf/-fr). The patterns below cover other destructive binaries.
_ALWAYS_DENY: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bdd\b"), "dd"),
    (re.compile(r"\bmkfs\b"), "mkfs"),
    (re.compile(r"\bshred\b"), "shred"),
    (re.compile(r"\bfdisk\b"), "fdisk"),
    (re.compile(r"\bparted\b"), "parted"),
    (re.compile(r"\bwipefs\b"), "wipefs"),
    (re.compile(r">\s*/dev/[^\s]"), "write_to_dev"),
]

_RF_FLAGS = re.compile(
    r"-[a-zA-Z]*[rR][a-zA-Z]*[fF]"
    r"|-[a-zA-Z]*[fF][a-zA-Z]*[rR]"
    r"|--recursive\b.*--force\b"
    r"|--force\b.*--recursive\b"
)

# Write-operation patterns — blocked in readonly mode.
_WRITE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\becho\b.*>"),
    re.compile(r"\btee\b"),
    re.compile(r"\bsed\b.*-i"),
    re.compile(r"\bcp\b"),
    re.compile(r"\bmv\b"),
    re.compile(r"\bmkdir\b"),
    re.compile(r"\btouch\b"),
    re.compile(r"\bchmod\b"),
    re.compile(r"\bchown\b"),
    re.compile(r"\bgit\s+(commit|push|merge|rebase|reset|clean)\b"),
    re.compile(r">[^>]"),
]

# Network commands blocked unless trust == "yolo" or in allowlist.
NETWORK_COMMANDS: frozenset[str] = frozenset([
    "curl", "wget", "nc", "netcat", "ssh", "scp", "rsync",
    "ftp", "sftp", "telnet", "nmap", "ping", "traceroute",
    "dig", "nslookup", "host", "http", "https",
])


# ---------------------------------------------------------------------------
# Allowlist — loaded from .agent/config.toml
# ---------------------------------------------------------------------------

def load_shell_allowlist(project_root: Path) -> list[str]:
    """Return glob patterns from .agent/config.toml [shell_allowlist] table.

    Example config.toml entry:
        shell_allowlist = ["npm run *", "pytest *", "python -m pytest *"]
    """
    config_path = project_root / ".agent" / "config.toml"
    if not config_path.exists():
        return []
    try:
        data = tomllib.loads(config_path.read_text())
        patterns = data.get("shell_allowlist", [])
        return [p for p in patterns if isinstance(p, str)]
    except Exception:
        return []


def is_command_in_allowlist(command: str, patterns: list[str]) -> bool:
    """Return True if command matches any glob pattern in patterns."""
    cmd = command.strip()
    return any(fnmatch.fnmatch(cmd, p) for p in patterns)


# ---------------------------------------------------------------------------
# Core policy checks
# ---------------------------------------------------------------------------

def _is_dangerous_rm(command: str) -> bool:
    return bool(re.search(r"\brm\b", command) and _RF_FLAGS.search(command))


def is_shell_allowed(
    command: str,
    trust: TrustMode,
    allowlist: list[str] | None = None,
) -> tuple[bool, str]:
    """Return (allowed, reason).

    Evaluation order:
      1. Permanent denies — always blocked, allowlist cannot override.
      2. Allowlist match   — if matched, skip trust-mode restrictions.
      3. Trust-mode rules  — readonly blocks writes; non-yolo blocks network via _check_network.
    """
    # 1. Permanent denies
    if _is_dangerous_rm(command):
        return False, "rm with recursive+force flags is permanently denied"

    for pattern, label in _ALWAYS_DENY:
        if pattern.search(command):
            return False, f"Command matches permanently-denied pattern: {label!r}"

    # 2. Allowlist override (skips trust-mode write/network restrictions)
    if allowlist and is_command_in_allowlist(command, allowlist):
        return True, ""

    # 3. Trust-mode write restriction
    if trust == "readonly":
        for pat in _WRITE_PATTERNS:
            if pat.search(command):
                return False, f"Write blocked in readonly mode (pattern: {pat.pattern!r})"

    return True, ""


def is_path_allowed(path: str | Path, project_root: Path) -> tuple[bool, str]:
    """Return (allowed, reason). reason is empty when path is within project_root."""
    try:
        resolved = Path(path).resolve()
        resolved.relative_to(project_root.resolve())
        return True, ""
    except ValueError:
        return False, f"Path {str(path)!r} is outside project root {str(project_root)!r}"


def is_network_allowed(trust: TrustMode) -> tuple[bool, str]:
    """Network from tools is blocked unless trust == yolo."""
    if trust == "yolo":
        return True, ""
    return False, "Outbound network is disabled. Set trust=yolo to enable."
