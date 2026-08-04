"""LangGraph state definition for the coding agent."""
from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agent.code_index.search import CodeChunk
from agent.llm.schemas import Plan

AgentState = TypedDict(
    "AgentState",
    {
        "messages": Annotated[list[BaseMessage], add_messages],
        "user_request": str,
        "plan": Plan | None,
        "current_step_index": int,
        "retrieved": list[CodeChunk],
        "iteration": int,
        "verifier_failures": int,
        "task_id": str,
        "outcome": Literal["pending", "success", "partial", "failure"],
        "files_changed": list[str],
    },
)
