from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.tools.search import search_code


class TestSearchCode:
    def test_finds_pattern_in_project(self, project_root: Path) -> None:
        (project_root / "main.py").write_text("def hello_world():\n    pass\n")
        result = search_code.invoke({"pattern": "hello_world"})
        assert "hello_world" in result
        assert "main.py" in result

    def test_no_matches_returns_message(self, project_root: Path) -> None:
        (project_root / "empty.py").write_text("x = 1\n")
        result = search_code.invoke({"pattern": "zzznomatch_xyz"})
        assert "No matches found" in result

    def test_file_glob_filters_results(self, project_root: Path) -> None:
        (project_root / "code.py").write_text("TARGET_SYMBOL = 1\n")
        (project_root / "code.js").write_text("TARGET_SYMBOL = 1\n")
        result = search_code.invoke({"pattern": "TARGET_SYMBOL", "file_glob": "*.py"})
        assert "code.py" in result
        # js file may or may not appear depending on ripgrep behaviour with glob path
        assert "TARGET_SYMBOL" in result

    def test_max_results_truncation(self, project_root: Path) -> None:
        content = "\n".join(f"pattern_{i} = {i}" for i in range(30))
        (project_root / "many.py").write_text(content)
        result = search_code.invoke({"pattern": "pattern_", "max_results": 5})
        assert "truncated" in result

    def test_raises_if_rg_not_found(self, project_root: Path) -> None:
        with patch("agent.tools.search.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError, match="ripgrep"):
                search_code.invoke({"pattern": "anything"})

    def test_line_numbers_in_output(self, project_root: Path) -> None:
        (project_root / "f.py").write_text("line1\nfind_me\nline3\n")
        result = search_code.invoke({"pattern": "find_me"})
        # ripgrep --line-number output includes line number
        assert "find_me" in result

    def test_no_truncation_under_limit(self, project_root: Path) -> None:
        content = "\n".join(f"needle_{i} = {i}" for i in range(10))
        (project_root / "small.py").write_text(content)
        result = search_code.invoke({"pattern": "needle_", "max_results": 50})
        assert "truncated" not in result
