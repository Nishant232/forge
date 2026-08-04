"""Conditional edge functions for the LangGraph state machine.

Each function takes AgentState and returns a string node name indicating
where control should flow next.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from agent.orchestrator.state import AgentState

_MAX_ITERATIONS = 25
_MAX_VERIFIER_RETRIES = 2


def after_act(state: AgentState) -> str:
    """Route after act_node.

    - Iteration cap hit → commit (partial)
    - No tool calls in last act + task not done → act (force tool use)
    - Files were changed → verify
    - Otherwise → commit
    """
    iteration = state.get("iteration", 0)
    if iteration >= _MAX_ITERATIONS:
        return "commit"

    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None,
    )
    no_tool_calls = last_ai is None or not last_ai.tool_calls

    # If the model produced no tool calls and we haven't hit the cap,
    # loop back so act_node can use the stronger ACT_NO_TOOLS_WARNING prompt.
    # Cap this retry at 3 iterations to avoid spinning on a stuck model.
    if no_tool_calls and not state.get("files_changed") and iteration < 6:
        return "act"

    if state.get("files_changed"):
        return "verify"
    return "commit"


def after_verify(state: AgentState) -> str:
    """Route after verify_node.

    - Iteration cap hit → commit
    - No failures recorded → commit (pass)
    - Retries exhausted → commit (failure)
    - Otherwise → act (retry)
    """
    if state.get("iteration", 0) >= _MAX_ITERATIONS:
        return "commit"

    failures = state.get("verifier_failures", 0)
    if failures == 0:
        return "commit"
    if failures >= _MAX_VERIFIER_RETRIES:
        return "commit"
    return "act"


def after_plan(state: AgentState) -> str:
    """Route after plan_node — always retrieve first."""
    return "retrieve"
