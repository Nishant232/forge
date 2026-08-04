from __future__ import annotations

import pytest

import agent.llm.prompts as P


# Every exported constant must be a non-empty string.
_ALL_CONSTANTS = [
    ("SYSTEM", P.SYSTEM),
    ("PLAN", P.PLAN),
    ("ACT", P.ACT),
    ("VERIFY_FAILURE", P.VERIFY_FAILURE),
    ("COMMIT", P.COMMIT),
    ("CONTEXT_BLOCK", P.CONTEXT_BLOCK),
    ("EPISODE_BLOCK", P.EPISODE_BLOCK),
]


class TestConstantsExist:
    @pytest.mark.parametrize("name,value", _ALL_CONSTANTS)
    def test_constant_is_non_empty_string(self, name: str, value: object) -> None:
        assert isinstance(value, str), f"{name} should be str"
        assert len(value.strip()) > 0, f"{name} should not be blank"


class TestSystemPrompt:
    def test_has_project_root_placeholder(self) -> None:
        assert "{project_root}" in P.SYSTEM

    def test_has_trust_mode_placeholder(self) -> None:
        assert "{trust_mode}" in P.SYSTEM

    def test_has_conventions_placeholder(self) -> None:
        assert "{conventions}" in P.SYSTEM

    def test_format_produces_complete_string(self) -> None:
        result = P.SYSTEM.format(
            project_root="/home/user/myproject",
            trust_mode="trusted",
            conventions="test_runner: pytest\nlinter: ruff",
        )
        assert "/home/user/myproject" in result
        assert "trusted" in result
        assert "pytest" in result

    def test_no_unformatted_braces_after_format(self) -> None:
        result = P.SYSTEM.format(
            project_root="/proj",
            trust_mode="readonly",
            conventions="none",
        )
        # Should not raise a KeyError and should not contain literal '{' from
        # unresolved placeholders (braces that were meant to be filled)
        assert "project_root" not in result


class TestPlanPrompt:
    def test_has_required_placeholders(self) -> None:
        for placeholder in ("{request}", "{context}", "{tool_names}"):
            assert placeholder in P.PLAN, f"PLAN missing {placeholder}"

    def test_formats_correctly(self) -> None:
        result = P.PLAN.format(
            request="Fix the off-by-one bug in loop",
            context="def loop(): ...",
            tool_names="read_file, edit_file, run_tests",
        )
        assert "Fix the off-by-one bug" in result
        assert "read_file" in result


class TestActPrompt:
    def test_has_required_placeholders(self) -> None:
        for p in ("{step_description}", "{iteration}", "{max_iterations}"):
            assert p in P.ACT, f"ACT missing {p}"

    def test_formats_correctly(self) -> None:
        result = P.ACT.format(
            step_description="Read the main.py file",
            expected_tools="read_file, write_file",
            iteration=2,
            max_iterations=25,
        )
        assert "Read the main.py" in result
        assert "2" in result


class TestVerifyFailurePrompt:
    def test_has_required_placeholders(self) -> None:
        for p in ("{verifier_output}", "{files_changed}", "{retries_left}"):
            assert p in P.VERIFY_FAILURE, f"VERIFY_FAILURE missing {p}"

    def test_formats_correctly(self) -> None:
        result = P.VERIFY_FAILURE.format(
            verifier_output="E   AssertionError: 1 != 2",
            files_changed="['main.py']",
            retries_left=1,
        )
        assert "AssertionError" in result
        assert "1" in result


class TestCommitPrompt:
    def test_has_required_placeholders(self) -> None:
        for p in ("{request}", "{action_count}", "{outcome}"):
            assert p in P.COMMIT, f"COMMIT missing {p}"

    def test_formats_correctly(self) -> None:
        result = P.COMMIT.format(
            request="Add tests", action_count=5, outcome="success", files_changed="none"
        )
        assert "Add tests" in result
        assert "success" in result


class TestContextBlocks:
    def test_context_block_has_chunks_placeholder(self) -> None:
        assert "{chunks}" in P.CONTEXT_BLOCK

    def test_episode_block_has_episodes_placeholder(self) -> None:
        assert "{episodes}" in P.EPISODE_BLOCK

    def test_context_block_formats(self) -> None:
        result = P.CONTEXT_BLOCK.format(chunks="def parse_query(): ...")
        assert "parse_query" in result

    def test_episode_block_formats(self) -> None:
        result = P.EPISODE_BLOCK.format(episodes="Task: fixed bug\nOutcome: success")
        assert "fixed bug" in result
