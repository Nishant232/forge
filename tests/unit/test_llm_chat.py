"""Unit tests for llm/chat.py.

All tests here run without a live LLM. Smoke tests requiring a real model
are at the bottom and are skipped unless OLLAMA_URL or GROQ_API_KEY is set.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableBinding

from agent.llm.chat import chat_model, structured, with_tools
from agent.llm.schemas import Plan, PlanStep
from agent.tools.registry import all_tools


# ---------------------------------------------------------------------------
# chat_model() — Ollama provider (default)
# ---------------------------------------------------------------------------


class TestChatModelOllama:
    def setup_method(self) -> None:
        os.environ.pop("LLM_PROVIDER", None)
        os.environ.pop("GROQ_API_KEY", None)

    def test_returns_base_chat_model(self) -> None:
        m = chat_model()
        assert isinstance(m, BaseChatModel)

    def test_default_temperature_is_zero(self) -> None:
        m = chat_model()
        assert m.temperature == 0

    def test_custom_temperature_applied(self) -> None:
        m = chat_model(temperature=0.2)
        assert abs(m.temperature - 0.2) < 1e-6

    def test_respects_ollama_url_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_URL", "http://custom-host:11434")
        import importlib
        import agent.llm.chat as chat_mod
        importlib.reload(chat_mod)
        m = chat_mod.chat_model()
        assert "custom-host" in m.base_url
        importlib.reload(chat_mod)

    def test_respects_chat_model_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("CHAT_MODEL", "llama3:8b")
        import importlib
        import agent.llm.chat as chat_mod
        importlib.reload(chat_mod)
        m = chat_mod.chat_model()
        assert m.model == "llama3:8b"
        importlib.reload(chat_mod)

    def test_model_name_matches_module_constant(self) -> None:
        import agent.llm.chat as chat_mod
        m = chat_model()
        assert m.model == chat_mod._CHAT_MODEL


# ---------------------------------------------------------------------------
# chat_model() — Groq provider
# ---------------------------------------------------------------------------


class TestChatModelGroq:
    def test_groq_provider_returns_chat_groq(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "dummy_key")
        import importlib
        import agent.llm.chat as chat_mod
        importlib.reload(chat_mod)
        m = chat_mod.chat_model()
        assert isinstance(m, BaseChatModel)
        from langchain_groq import ChatGroq
        assert isinstance(m, ChatGroq)
        importlib.reload(chat_mod)

    def test_groq_uses_groq_model_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "dummy_key")
        monkeypatch.setenv("GROQ_MODEL", "mixtral-8x7b-32768")
        import importlib
        import agent.llm.chat as chat_mod
        importlib.reload(chat_mod)
        m = chat_mod.chat_model()
        assert m.model_name == "mixtral-8x7b-32768"
        importlib.reload(chat_mod)


# ---------------------------------------------------------------------------
# with_tools()
# ---------------------------------------------------------------------------


class TestWithTools:
    def test_returns_runnable(self) -> None:
        m = chat_model()
        bound = with_tools(m, all_tools())
        assert isinstance(bound, Runnable)

    def test_returns_runnable_binding(self) -> None:
        bound = with_tools(chat_model(), all_tools())
        assert isinstance(bound, RunnableBinding)

    def test_empty_tools_list_accepted(self) -> None:
        bound = with_tools(chat_model(), [])
        assert bound is not None

    def test_all_tools_bind_without_error(self) -> None:
        bound = with_tools(chat_model(), all_tools())
        assert bound is not None

    def test_bound_model_is_not_plain_base_model(self) -> None:
        from agent.tools.files import read_file, write_file
        bound = with_tools(chat_model(), [read_file, write_file])  # type: ignore[list-item]
        assert not isinstance(bound, BaseChatModel)

    def test_mocked_tool_call_response(self) -> None:
        """Verify with_tools produces tool_calls when the LLM is mocked."""
        tool_call = {
            "name": "read_file",
            "args": {"path": "/tmp/test.py"},
            "id": "call_abc123",
            "type": "tool_call",
        }
        ai_msg = AIMessage(content="", tool_calls=[tool_call])
        mock_bound = MagicMock(spec=Runnable)
        mock_bound.invoke.return_value = ai_msg

        with patch("langchain_ollama.ChatOllama.bind_tools", return_value=mock_bound):
            bound = with_tools(chat_model(), all_tools())
            response = bound.invoke([HumanMessage(content="Read test.py")])

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "read_file"


# ---------------------------------------------------------------------------
# structured()
# ---------------------------------------------------------------------------


class TestStructured:
    def test_returns_runnable(self) -> None:
        runnable = structured(chat_model(), Plan)
        assert isinstance(runnable, Runnable)

    def test_returns_different_type_from_raw_model(self) -> None:
        runnable = structured(chat_model(), Plan)
        assert not isinstance(runnable, BaseChatModel)

    def test_works_with_final_answer_schema(self) -> None:
        from agent.llm.schemas import FinalAnswer
        runnable = structured(chat_model(), FinalAnswer)
        assert isinstance(runnable, Runnable)

    def test_mocked_structured_invoke_returns_plan(self) -> None:
        expected_plan = Plan(
            steps=[PlanStep(description="Read file", expected_tools=["read_file"])],
            rationale="Need to inspect the file first",
        )
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.invoke.return_value = expected_plan

        with patch("langchain_ollama.ChatOllama.with_structured_output", return_value=mock_runnable):
            runnable = structured(chat_model(), Plan)
            result = runnable.invoke([HumanMessage(content="Fix the bug")])

        assert isinstance(result, Plan)
        assert result.rationale == "Need to inspect the file first"


# ---------------------------------------------------------------------------
# Smoke tests — Groq (uses real API, skipped unless GROQ_API_KEY is set)
# ---------------------------------------------------------------------------

_SKIP_NO_GROQ = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="smoke — requires GROQ_API_KEY",
)

_SKIP_NO_OLLAMA = pytest.mark.skipif(
    not os.environ.get("OLLAMA_URL"),
    reason="smoke — requires live Ollama (set OLLAMA_URL to enable)",
)


@_SKIP_NO_GROQ
def test_groq_chat_smoke() -> None:
    """Groq provider returns a non-empty response."""
    import importlib
    import agent.llm.chat as chat_mod
    os.environ["LLM_PROVIDER"] = "groq"
    importlib.reload(chat_mod)
    m = chat_mod.chat_model()
    response = m.invoke([HumanMessage(content="Reply with just: hello")])
    assert isinstance(response, AIMessage)
    assert response.content
    os.environ.pop("LLM_PROVIDER", None)
    importlib.reload(chat_mod)


@_SKIP_NO_GROQ
def test_groq_structured_plan_smoke() -> None:
    """structured() with Groq returns a valid Plan."""
    import importlib
    import agent.llm.chat as chat_mod
    os.environ["LLM_PROVIDER"] = "groq"
    importlib.reload(chat_mod)
    runnable = chat_mod.structured(chat_mod.chat_model(), Plan)
    result = runnable.invoke(
        [HumanMessage(content="Read file main.py and add a docstring to parse_query")]
    )
    assert isinstance(result, Plan)
    assert len(result.steps) >= 1
    os.environ.pop("LLM_PROVIDER", None)
    importlib.reload(chat_mod)


@_SKIP_NO_OLLAMA
def test_ollama_structured_plan_smoke() -> None:
    """structured(chat_model(), Plan).invoke(...) must return a valid Plan."""
    runnable = structured(chat_model(), Plan)
    result = runnable.invoke(
        [HumanMessage(content="Read file main.py and add a docstring to parse_query")]
    )
    assert isinstance(result, Plan)
    assert len(result.steps) >= 1
    assert result.rationale
