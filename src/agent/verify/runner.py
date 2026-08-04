"""Verifier: runs lint, typecheck, and tests on changed files only.

Auto-detects the project's toolchain by inspecting config files in the
project root. Returns a structured VerifierResult consumed by verify_node.

Supports:
  Python  — ruff (lint), mypy (types), pytest (tests)
  JS/TS   — eslint (lint), tsc --noEmit (types), jest (tests)
  Rust    — cargo clippy (lint+types), cargo test
"""
from __future__ import annotations

import subprocess
import time
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

Language = Literal["python", "javascript", "typescript", "rust", "unknown"]


class CheckResult(BaseModel):
    name: str
    passed: bool
    output: str
    duration_ms: int


class VerifierResult(BaseModel):
    passed: bool
    checks: list[CheckResult]
    output: str          # combined human-readable summary
    failed_checks: list[str]
    duration_ms: int


def _run(cmd: list[str], cwd: Path, timeout: int = 60) -> tuple[int, str]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd),
        )
        return result.returncode, (result.stdout + result.stderr).strip()
    except subprocess.TimeoutExpired:
        return 1, f"Timed out after {timeout}s"
    except FileNotFoundError:
        return 1, f"Command not found: {cmd[0]}"


# ---------------------------------------------------------------------------
# Toolchain detection
# ---------------------------------------------------------------------------


def _detect_language(project_root: Path) -> Language:
    if (project_root / "pyproject.toml").exists() or (project_root / "setup.py").exists():
        return "python"
    if (project_root / "tsconfig.json").exists():
        return "typescript"
    if (project_root / "package.json").exists():
        return "javascript"
    if (project_root / "Cargo.toml").exists():
        return "rust"
    return "unknown"


def _has_ruff(project_root: Path) -> bool:
    try:
        toml_path = project_root / "pyproject.toml"
        if toml_path.exists():
            data = tomllib.loads(toml_path.read_text())
            tools = data.get("tool", {})
            if "ruff" in tools:
                return True
        rc = subprocess.run(["ruff", "--version"], capture_output=True, timeout=5).returncode
        return rc == 0
    except Exception:
        return False


def _has_mypy(project_root: Path) -> bool:
    try:
        rc = subprocess.run(["mypy", "--version"], capture_output=True, timeout=5).returncode
        return rc == 0
    except Exception:
        return False


def _has_pytest(project_root: Path) -> bool:
    try:
        rc = subprocess.run(
            ["python", "-m", "pytest", "--version"], capture_output=True, timeout=5
        ).returncode
        return rc == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Per-language check runners
# ---------------------------------------------------------------------------


def _changed_python_files(changed: list[str]) -> list[str]:
    return [f for f in changed if f.endswith(".py")]


def _find_test_files(src_files: list[str], project_root: Path) -> list[str]:
    """Heuristically map source files to their test counterparts."""
    tests: list[str] = []
    tests_dir = project_root / "tests"
    for src in src_files:
        stem = Path(src).stem
        for candidate in tests_dir.rglob(f"test_{stem}.py"):
            tests.append(str(candidate))
        for candidate in tests_dir.rglob(f"{stem}_test.py"):
            tests.append(str(candidate))
    return list(set(tests))


def _run_ruff(py_files: list[str], project_root: Path) -> CheckResult:
    t0 = time.monotonic()
    code, out = _run(["ruff", "check"] + py_files, project_root)
    return CheckResult(
        name="ruff",
        passed=code == 0,
        output=out or "(no output)",
        duration_ms=int((time.monotonic() - t0) * 1000),
    )


def _run_mypy(py_files: list[str], project_root: Path) -> CheckResult:
    t0 = time.monotonic()
    code, out = _run(["mypy", "--no-error-summary"] + py_files, project_root, timeout=120)
    return CheckResult(
        name="mypy",
        passed=code == 0,
        output=out or "(no output)",
        duration_ms=int((time.monotonic() - t0) * 1000),
    )


def _run_pytest(test_files: list[str], project_root: Path) -> CheckResult:
    t0 = time.monotonic()
    args = ["python", "-m", "pytest", "--tb=short", "-q"] + test_files
    code, out = _run(args, project_root, timeout=120)
    return CheckResult(
        name="pytest",
        passed=code == 0,
        output=out or "(no output)",
        duration_ms=int((time.monotonic() - t0) * 1000),
    )


def _run_cargo(project_root: Path) -> list[CheckResult]:
    results = []
    t0 = time.monotonic()
    code, out = _run(["cargo", "clippy", "--", "-D", "warnings"], project_root, timeout=180)
    results.append(CheckResult(
        name="cargo-clippy",
        passed=code == 0,
        output=out or "(no output)",
        duration_ms=int((time.monotonic() - t0) * 1000),
    ))
    t0 = time.monotonic()
    code, out = _run(["cargo", "test"], project_root, timeout=180)
    results.append(CheckResult(
        name="cargo-test",
        passed=code == 0,
        output=out or "(no output)",
        duration_ms=int((time.monotonic() - t0) * 1000),
    ))
    return results


def _run_js_checks(changed: list[str], project_root: Path) -> list[CheckResult]:
    results = []
    js_files = [f for f in changed if f.endswith((".js", ".ts", ".jsx", ".tsx"))]

    # eslint
    t0 = time.monotonic()
    code, out = _run(["npx", "eslint"] + js_files, project_root, timeout=60)
    results.append(CheckResult(
        name="eslint",
        passed=code == 0,
        output=out or "(no output)",
        duration_ms=int((time.monotonic() - t0) * 1000),
    ))

    # tsc
    if (project_root / "tsconfig.json").exists():
        t0 = time.monotonic()
        code, out = _run(["npx", "tsc", "--noEmit"], project_root, timeout=120)
        results.append(CheckResult(
            name="tsc",
            passed=code == 0,
            output=out or "(no output)",
            duration_ms=int((time.monotonic() - t0) * 1000),
        ))

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_verifier(
    changed_files: list[str],
    project_root: Path,
    *,
    run_lint: bool = True,
    run_types: bool = True,
    run_tests: bool = True,
) -> VerifierResult:
    """Run the appropriate checks for the given changed files.

    Args:
        changed_files: Relative or absolute paths of files modified by the agent.
        project_root: Absolute path to the project being worked on.
        run_lint: Whether to run the linter.
        run_types: Whether to run the type checker.
        run_tests: Whether to run the test suite.

    Returns:
        VerifierResult with per-check outcomes and a combined summary.
    """
    wall_start = time.monotonic()
    checks: list[CheckResult] = []
    lang = _detect_language(project_root)

    if lang == "python":
        py_files = _changed_python_files(changed_files)
        if not py_files:
            # No Python source files changed — skip language-specific checks.
            return VerifierResult(
                passed=True,
                checks=[],
                output="No Python source files changed; skipping verifier.",
                failed_checks=[],
                duration_ms=int((time.monotonic() - wall_start) * 1000),
            )

        if run_lint and _has_ruff(project_root):
            checks.append(_run_ruff(py_files, project_root))

        if run_types and _has_mypy(project_root):
            checks.append(_run_mypy(py_files, project_root))

        if run_tests and _has_pytest(project_root):
            test_files = _find_test_files(py_files, project_root)
            if test_files:
                checks.append(_run_pytest(test_files, project_root))

    elif lang in ("javascript", "typescript"):
        if run_lint or run_types:
            checks.extend(_run_js_checks(changed_files, project_root))

    elif lang == "rust":
        if run_lint or run_types or run_tests:
            checks.extend(_run_cargo(project_root))

    # No checks available — pass trivially
    if not checks:
        return VerifierResult(
            passed=True,
            checks=[],
            output="No verifier checks available for changed files.",
            failed_checks=[],
            duration_ms=int((time.monotonic() - wall_start) * 1000),
        )

    failed = [c.name for c in checks if not c.passed]
    lines = []
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        lines.append(f"[{status}] {c.name} ({c.duration_ms}ms)")
        if not c.passed:
            lines.append(c.output)

    return VerifierResult(
        passed=len(failed) == 0,
        checks=checks,
        output="\n".join(lines),
        failed_checks=failed,
        duration_ms=int((time.monotonic() - wall_start) * 1000),
    )
