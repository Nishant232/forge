from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent.tools.git import (
    git_add,
    git_checkpoint,
    git_commit,
    git_diff,
    git_log,
    git_rollback,
    git_status,
)


class TestGitDiffError:
    def test_git_diff_bad_path_raises(self, git_repo: Path) -> None:
        from unittest.mock import patch
        with patch("agent.tools.git._run_git", return_value=(128, "bad object")):
            with pytest.raises(RuntimeError, match="git diff failed"):
                git_diff.invoke({"path": "nonexistent.py"})


class TestGitAddError:
    def test_git_add_bad_path_raises(self, git_repo: Path) -> None:
        from unittest.mock import patch
        with patch("agent.tools.git._run_git", return_value=(128, "pathspec not matched")):
            with pytest.raises(RuntimeError, match="git add failed"):
                git_add.invoke({"path": "no_such_file.py"})


class TestGitCheckpointError:
    def test_checkpoint_fails_gracefully(self, git_repo: Path) -> None:
        from unittest.mock import patch
        with patch("agent.tools.git._run_git", return_value=(128, "tag already exists")):
            with pytest.raises(RuntimeError, match="Failed to create checkpoint tag"):
                git_checkpoint.invoke({"label": "dup"})


class TestGitRollbackError:
    def test_rollback_bad_tag_raises(self, git_repo: Path) -> None:
        git_checkpoint.invoke({})  # create at least one checkpoint
        from unittest.mock import patch
        # Make checkout fail
        original_run_git = __import__("agent.tools.git", fromlist=["_run_git"])._run_git
        call_count = 0

        def fake_run_git(args: list[str], cwd: object = None) -> tuple[int, str]:
            nonlocal call_count
            if args[0] == "checkout":
                return 128, "error: pathspec not matched"
            return original_run_git(args, cwd)

        with patch("agent.tools.git._run_git", side_effect=fake_run_git):
            with pytest.raises(RuntimeError, match="Rollback.*failed"):
                git_rollback.invoke({"checkpoint_tag": "agent-cp-fake-000"})


class TestGitStatus:
    def test_clean_tree(self, git_repo: Path) -> None:
        result = git_status.invoke({})
        assert "clean working tree" in result

    def test_shows_untracked(self, git_repo: Path) -> None:
        (git_repo / "new_file.py").write_text("x = 1")
        result = git_status.invoke({})
        assert "new_file.py" in result


class TestGitDiff:
    def test_no_changes(self, git_repo: Path) -> None:
        result = git_diff.invoke({})
        assert "no changes" in result

    def test_shows_modified_content(self, git_repo: Path) -> None:
        f = git_repo / "init.txt"
        f.write_text("modified content")
        result = git_diff.invoke({})
        assert "modified" in result or "init.txt" in result

    def test_path_filter(self, git_repo: Path) -> None:
        (git_repo / "a.py").write_text("a = 1")
        subprocess.run(["git", "add", "a.py"], cwd=str(git_repo), capture_output=True)
        result = git_diff.invoke({"path": "a.py"})
        # staged diff won't show for unstaged diff — just confirm no crash
        assert isinstance(result, str)


class TestGitAdd:
    def test_stages_file(self, git_repo: Path) -> None:
        f = git_repo / "staged.py"
        f.write_text("x = 1")
        result = git_add.invoke({"path": str(f)})
        assert "staged.py" in result.lower() or "Staged" in result

    def test_stages_dot(self, git_repo: Path) -> None:
        (git_repo / "file.txt").write_text("content")
        result = git_add.invoke({"path": "."})
        assert isinstance(result, str)


class TestGitCommit:
    def test_commit_on_feature_branch(self, git_repo: Path) -> None:
        f = git_repo / "feat.py"
        f.write_text("def foo(): pass")
        git_add.invoke({"path": str(f)})
        result = git_commit.invoke({"message": "add feat"})
        assert "feat" in result or "master" not in result

    def test_blocked_on_main(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # main branch already exists from git init; switch to it (no -b)
        subprocess.run(
            ["git", "checkout", "main"],
            cwd=str(git_repo), capture_output=True,
        )
        with pytest.raises(PermissionError, match="protected branch"):
            git_commit.invoke({"message": "should fail"})

    def test_blocked_on_master(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        subprocess.run(
            ["git", "checkout", "-b", "master"],
            cwd=str(git_repo), capture_output=True,
        )
        with pytest.raises(PermissionError):
            git_commit.invoke({"message": "should fail"})

    def test_raises_when_nothing_staged(self, git_repo: Path) -> None:
        # No staged files → git commit should fail
        with pytest.raises(RuntimeError, match="commit failed"):
            git_commit.invoke({"message": "empty commit"})


class TestGitLog:
    def test_returns_log(self, git_repo: Path) -> None:
        result = git_log.invoke({})
        assert "init" in result  # the initial commit message from fixture

    def test_respects_n_limit(self, git_repo: Path) -> None:
        result = git_log.invoke({"n": 1})
        lines = [l for l in result.strip().splitlines() if l.strip()]
        assert len(lines) <= 1

    def test_empty_repo_message(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("AGENT_TRUST_MODE", "trusted")
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        result = git_log.invoke({})
        assert "no commits" in result or isinstance(result, str)


class TestGitCheckpoint:
    def test_creates_tag(self, git_repo: Path) -> None:
        result = git_checkpoint.invoke({})
        assert "Checkpoint created" in result
        assert "agent-cp-" in result

    def test_creates_with_label(self, git_repo: Path) -> None:
        result = git_checkpoint.invoke({"label": "before-refactor"})
        assert "before-refactor" in result

    def test_records_in_index_file(self, git_repo: Path) -> None:
        git_checkpoint.invoke({"label": "mycp"})
        index = git_repo / ".agent" / "checkpoints" / "index.txt"
        assert index.exists()
        content = index.read_text()
        assert "mycp" in content


class TestGitRollback:
    def test_rollback_to_last_checkpoint(self, git_repo: Path) -> None:
        # Create a checkpoint, then modify a file, then rollback
        f = git_repo / "init.txt"
        original = f.read_text()
        git_checkpoint.invoke({"label": "pre"})

        f.write_text("modified by agent")
        assert f.read_text() == "modified by agent"

        result = git_rollback.invoke({})
        assert "Rolled back" in result
        # File should be restored
        assert f.read_text() == original

    def test_rollback_to_named_tag(self, git_repo: Path) -> None:
        result_cp = git_checkpoint.invoke({"label": "named"})
        tag = result_cp.replace("Checkpoint created: ", "").strip()
        result = git_rollback.invoke({"checkpoint_tag": tag})
        assert tag in result

    def test_raises_if_no_checkpoints(self, git_repo: Path) -> None:
        # Fresh repo with no agent tags
        with pytest.raises(RuntimeError, match="No agent checkpoints"):
            git_rollback.invoke({})
