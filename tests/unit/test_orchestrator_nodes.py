"""Unit tests for orchestrator nodes and routing.

All LLM calls are mocked — these tests run without a live Ollama instance.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.code_index.search import CodeChunk
from agent.llm.schemas import Plan, PlanStep
from agent.orchestrator.routing import after_act, after_plan, after_verify
from agent.orchestrator.state import AgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides: Any) -> AgentState:
    base: AgentState = {  # type: ignore[misc]
        "messages": [HumanMessage(content="Fix the bug")],
        "user_request": "Fix the bug",
        "plan": None,
        "current_step_index": 0,
        "retrieved": [],
        "iteration": 0,
        "verifier_failures": 0,
        "task_id": str(uuid.uuid4()),
        "outcome": "pending",
        "files_changed": [],
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def _make_plan() -> Plan:
    return Plan(
        steps=[
            PlanStep(description="Read main.py", expected_tools=["read_file"]),
            PlanStep(description="Edit main.py", expected_tools=["edit_file"]),
        ],
        rationale="Need to read before editing",
    )


def _make_chunk(symbol: str = "parse") -> CodeChunk:
    return CodeChunk(
        path="src/main.py",
        start_line=1,
        end_line=10,
        symbol=symbol,
        kind="function",
        content=f"def {symbol}(): pass",
        score=0.8,
    )


# ---------------------------------------------------------------------------
# Routing — after_plan
# ---------------------------------------------------------------------------


class TestAfterPlan:
    def test_always_routes_to_retrieve(self) -> None:
        state = _make_state()
        assert after_plan(state) == "retrieve"


# ---------------------------------------------------------------------------
# Routing — after_act
# ---------------------------------------------------------------------------


class TestAfterAct:
    def test_routes_to_verify_when_files_changed(self) -> None:
        state = _make_state(files_changed=["src/main.py"])
        assert after_act(state) == "verify"

    def test_routes_to_commit_when_no_files_changed(self) -> None:
        # iteration >= 6 forces commit even with no files changed
        state = _make_state(files_changed=[], iteration=6)
        assert after_act(state) == "commit"

    def test_routes_to_commit_at_iteration_cap(self) -> None:
        state = _make_state(iteration=25, files_changed=["a.py"])
        assert after_act(state) == "commit"

    def test_routes_to_commit_beyond_cap(self) -> None:
        state = _make_state(iteration=30)
        assert after_act(state) == "commit"


# ---------------------------------------------------------------------------
# Routing — after_verify
# ---------------------------------------------------------------------------


class TestAfterVerify:
    def test_routes_to_commit_when_no_failures(self) -> None:
        state = _make_state(verifier_failures=0)
        assert after_verify(state) == "commit"

    def test_routes_to_act_on_first_failure(self) -> None:
        state = _make_state(verifier_failures=1)
        assert after_verify(state) == "act"

    def test_routes_to_commit_when_retries_exhausted(self) -> None:
        state = _make_state(verifier_failures=2)
        assert after_verify(state) == "commit"

    def test_routes_to_commit_at_iteration_cap_regardless_of_failures(self) -> None:
        state = _make_state(verifier_failures=1, iteration=25)
        assert after_verify(state) == "commit"


# ---------------------------------------------------------------------------
# plan_node
# ---------------------------------------------------------------------------


class TestPlanNode:
    @pytest.mark.asyncio
    async def test_plan_node_sets_plan(self) -> None:
        from agent.orchestrator.nodes import plan_node

        expected_plan = _make_plan()

        mock_runnable = AsyncMock()
        mock_runnable.ainvoke = AsyncMock(return_value=expected_plan)

        with (
            patch("agent.orchestrator.nodes.structured", return_value=mock_runnable),
            patch("agent.orchestrator.nodes.chat_model", return_value=MagicMock()),
            patch(
                "agent.orchestrator.nodes.ProjectStore",
                return_value=MagicMock(all=MagicMock(return_value=[])),
            ),
        ):
            result = await plan_node(_make_state())

        assert result["plan"] == expected_plan
        assert "messages" in result

    @pytest.mark.asyncio
    async def test_plan_node_resets_step_index(self) -> None:
        from agent.orchestrator.nodes import plan_node

        mock_runnable = AsyncMock()
        mock_runnable.ainvoke = AsyncMock(return_value=_make_plan())

        with (
            patch("agent.orchestrator.nodes.structured", return_value=mock_runnable),
            patch("agent.orchestrator.nodes.chat_model", return_value=MagicMock()),
            patch(
                "agent.orchestrator.nodes.ProjectStore",
                return_value=MagicMock(all=MagicMock(return_value=[])),
            ),
        ):
            result = await plan_node(_make_state(current_step_index=3))

        assert result["current_step_index"] == 0


# ---------------------------------------------------------------------------
# act_node
# ---------------------------------------------------------------------------


class TestActNode:
    @pytest.mark.asyncio
    async def test_act_node_increments_iteration(self) -> None:
        from agent.orchestrator.nodes import act_node

        ai_response = AIMessage(content="Done", tool_calls=[])

        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=ai_response)

        with (
            patch("agent.orchestrator.nodes.with_tools", return_value=mock_model),
            patch("agent.orchestrator.nodes.chat_model", return_value=MagicMock()),
        ):
            state = _make_state(iteration=3)
            result = await act_node(state)

        assert result["iteration"] == 4

    @pytest.mark.asyncio
    async def test_act_node_dispatches_tool_calls(self) -> None:
        from agent.orchestrator.nodes import act_node

        tool_call = {
            "name": "read_file",
            "args": {"path": "main.py"},
            "id": "call_001",
            "type": "tool_call",
        }
        ai_response = AIMessage(content="", tool_calls=[tool_call])

        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=ai_response)

        from agent.tools.registry import ToolResult

        mock_dispatch_result = ToolResult(ok=True, output="file content", duration_ms=5)

        with (
            patch("agent.orchestrator.nodes.with_tools", return_value=mock_model),
            patch("agent.orchestrator.nodes.chat_model", return_value=MagicMock()),
            patch("agent.orchestrator.nodes.dispatch", return_value=mock_dispatch_result),
        ):
            result = await act_node(_make_state())

        msgs = result["messages"]
        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].name == "read_file"

    @pytest.mark.asyncio
    async def test_act_node_tracks_write_file_changes(self) -> None:
        from agent.orchestrator.nodes import act_node

        tool_call = {
            "name": "write_file",
            "args": {"path": "src/out.py", "content": "x"},
            "id": "call_002",
            "type": "tool_call",
        }
        ai_response = AIMessage(content="", tool_calls=[tool_call])

        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=ai_response)

        from agent.tools.registry import ToolResult

        mock_result = ToolResult(ok=True, output="written", duration_ms=2)

        with (
            patch("agent.orchestrator.nodes.with_tools", return_value=mock_model),
            patch("agent.orchestrator.nodes.chat_model", return_value=MagicMock()),
            patch("agent.orchestrator.nodes.dispatch", return_value=mock_result),
        ):
            result = await act_node(_make_state())

        assert "src/out.py" in result["files_changed"]

    @pytest.mark.asyncio
    async def test_act_node_no_tool_calls_returns_empty_tool_msgs(self) -> None:
        from agent.orchestrator.nodes import act_node

        ai_response = AIMessage(content="No tools needed.", tool_calls=[])

        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=ai_response)

        with (
            patch("agent.orchestrator.nodes.with_tools", return_value=mock_model),
            patch("agent.orchestrator.nodes.chat_model", return_value=MagicMock()),
        ):
            result = await act_node(_make_state())

        tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert tool_msgs == []


# ---------------------------------------------------------------------------
# verify_node
# ---------------------------------------------------------------------------


class TestVerifyNode:
    @pytest.mark.asyncio
    async def test_verify_node_pass_does_not_increment_failures(self) -> None:
        from agent.orchestrator.nodes import verify_node
        from agent.verify.runner import VerifierResult

        mock_result = VerifierResult(
            passed=True, checks=[], output="all good", failed_checks=[], duration_ms=5
        )

        with patch("agent.orchestrator.nodes.run_verifier", return_value=mock_result):
            state = _make_state(files_changed=["src/main.py"], verifier_failures=0)
            result = await verify_node(state)

        assert result.get("verifier_failures", 0) == 0

    @pytest.mark.asyncio
    async def test_verify_node_fail_increments_failures(self) -> None:
        from agent.orchestrator.nodes import verify_node
        from agent.verify.runner import VerifierResult

        mock_result = VerifierResult(
            passed=False,
            checks=[],
            output="E   AssertionError",
            failed_checks=["pytest"],
            duration_ms=10,
        )

        with patch("agent.orchestrator.nodes.run_verifier", return_value=mock_result):
            state = _make_state(files_changed=["src/main.py"], verifier_failures=0)
            result = await verify_node(state)

        assert result["verifier_failures"] == 1

    @pytest.mark.asyncio
    async def test_verify_node_fail_appends_feedback_message(self) -> None:
        from agent.orchestrator.nodes import verify_node
        from agent.verify.runner import VerifierResult

        mock_result = VerifierResult(
            passed=False,
            checks=[],
            output="E   assert 1 == 2",
            failed_checks=["pytest"],
            duration_ms=10,
        )

        with patch("agent.orchestrator.nodes.run_verifier", return_value=mock_result):
            state = _make_state(files_changed=["src/main.py"])
            result = await verify_node(state)

        msgs = result.get("messages", [])
        assert any(isinstance(m, HumanMessage) for m in msgs)

    @pytest.mark.asyncio
    async def test_verify_node_no_files_changed_skips(self) -> None:
        from agent.orchestrator.nodes import verify_node

        with patch("agent.orchestrator.nodes.run_verifier") as mock_rv:
            state = _make_state(files_changed=[])
            await verify_node(state)
            mock_rv.assert_not_called()


# ---------------------------------------------------------------------------
# commit_node
# ---------------------------------------------------------------------------


class TestCommitNode:
    @pytest.mark.asyncio
    async def test_commit_node_sets_outcome_success(self) -> None:
        from agent.orchestrator.nodes import commit_node

        mock_ai = AIMessage(content="Task complete: added docstring.")

        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_ai)

        with (
            patch("agent.orchestrator.nodes.chat_model", return_value=mock_model),
            patch("agent.orchestrator.nodes.EpisodeStore", return_value=MagicMock()),
        ):
            result = await commit_node(_make_state(outcome="pending"))

        assert result["outcome"] == "success"

    @pytest.mark.asyncio
    async def test_commit_node_preserves_failure_outcome(self) -> None:
        from agent.orchestrator.nodes import commit_node

        mock_ai = AIMessage(content="Failed.")
        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_ai)

        with (
            patch("agent.orchestrator.nodes.chat_model", return_value=mock_model),
            patch("agent.orchestrator.nodes.EpisodeStore", return_value=MagicMock()),
        ):
            result = await commit_node(_make_state(outcome="failure"))

        assert result["outcome"] == "failure"

    @pytest.mark.asyncio
    async def test_commit_node_adds_summary_message(self) -> None:
        from agent.orchestrator.nodes import commit_node

        mock_ai = AIMessage(content="Done: fixed the off-by-one.")
        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_ai)

        with (
            patch("agent.orchestrator.nodes.chat_model", return_value=mock_model),
            patch("agent.orchestrator.nodes.EpisodeStore", return_value=MagicMock()),
        ):
            result = await commit_node(_make_state())

        assert any(isinstance(m, AIMessage) for m in result["messages"])


# ---------------------------------------------------------------------------
# retrieve_node
# ---------------------------------------------------------------------------


class TestRetrieveNode:
    @pytest.mark.asyncio
    async def test_retrieve_node_stores_chunks(self) -> None:
        from agent.orchestrator.nodes import retrieve_node

        chunks = [_make_chunk("parse_query")]

        with (
            patch("agent.orchestrator.nodes.semantic_search", AsyncMock(return_value=chunks)),
            patch("agent.orchestrator.nodes.embed_texts", AsyncMock(return_value=[[0.1] * 768]), create=True),
            patch("agent.orchestrator.nodes.EpisodeStore", return_value=MagicMock(search_semantic=MagicMock(return_value=[]))),
            patch("agent.orchestrator.nodes.ProjectStore", return_value=MagicMock(all=MagicMock(return_value=[]))),
        ):
            result = await retrieve_node(_make_state())

        assert len(result["retrieved"]) == 1
        assert result["retrieved"][0].symbol == "parse_query"

    @pytest.mark.asyncio
    async def test_retrieve_node_handles_search_failure_gracefully(self) -> None:
        from agent.orchestrator.nodes import retrieve_node

        with (
            patch("agent.orchestrator.nodes.semantic_search", AsyncMock(side_effect=Exception("index not ready"))),
            patch("agent.orchestrator.nodes.ProjectStore", return_value=MagicMock(all=MagicMock(return_value=[]))),
        ):
            result = await retrieve_node(_make_state())

        assert result["retrieved"] == []

    @pytest.mark.asyncio
    async def test_retrieve_node_returns_messages(self) -> None:
        from agent.orchestrator.nodes import retrieve_node

        with (
            patch("agent.orchestrator.nodes.semantic_search", AsyncMock(return_value=[])),
            patch("agent.orchestrator.nodes.ProjectStore", return_value=MagicMock(all=MagicMock(return_value=[]))),
        ):
            result = await retrieve_node(_make_state())

        assert "messages" in result


# ---------------------------------------------------------------------------
# Graph compilation smoke test
# ---------------------------------------------------------------------------


class TestGraphCompilation:
    def test_build_graph_compiles_without_error(self) -> None:
        from agent.orchestrator.graph import build_graph

        graph = build_graph(checkpointer=None)
        assert graph is not None

    def test_graph_has_correct_nodes(self) -> None:
        from agent.orchestrator.graph import build_graph

        graph = build_graph(checkpointer=None)
        # The compiled graph should have a graph attribute with nodes
        assert hasattr(graph, "nodes") or hasattr(graph, "graph")
