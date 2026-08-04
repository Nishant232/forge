"""One async function per LangGraph node.

Each function receives the full AgentState and returns a partial-state dict
that LangGraph merges back. Nodes never mutate state in place.

Node order in the happy path:
  plan_node → retrieve_node → act_node → verify_node → commit_node
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

try:
    from agent.code_index.search import CodeChunk, semantic_search  # [index] extra
    _CODE_INDEX_AVAILABLE = True
except ImportError:
    _CODE_INDEX_AVAILABLE = False
    CodeChunk = None  # type: ignore[assignment,misc]
from agent.context.assembler import assemble
from agent.llm.chat import chat_model, structured, with_tools
from agent.llm.prompts import ACT, ACT_NO_TOOLS_WARNING, PLAN, SYSTEM, VERIFY_FAILURE
from agent.llm.schemas import FinalAnswer, Plan
from agent.memory.episodic import Episode, EpisodeStore
from agent.memory.project import ProjectStore
from agent.orchestrator.state import AgentState
from agent.safety.policy import TrustMode
from agent.tools.registry import all_tools, dispatch
from agent.verify.runner import run_verifier

_MAX_ITERATIONS = 25
_MAX_VERIFIER_RETRIES = 2
_CONTEXT_BUDGET = 28_000  # tokens — leaves headroom inside a 32k context


def _agent_dir(workspace: Path) -> Path:
    return workspace / ".agent"


def _project_root() -> Path:
    root = os.environ.get("AGENT_PROJECT_ROOT")
    return Path(root).resolve() if root else Path.cwd().resolve()


# ---------------------------------------------------------------------------
# plan_node
# ---------------------------------------------------------------------------


async def plan_node(state: AgentState) -> dict[str, Any]:
    """Produce a Plan from the user request and initial context."""
    workspace = _project_root()

    # Minimal system prompt for planning — no tool schema needed here.
    project_store = ProjectStore(_agent_dir(workspace) / "state.db")
    facts = project_store.all()
    conventions = "\n".join(f"{f.key}: {f.value}" for f in facts) or "none detected"
    sys_text = SYSTEM.format(
        project_root=str(workspace),
        trust_mode=os.environ.get("AGENT_TRUST_MODE", "trusted"),
        conventions=conventions,
    )

    tool_names = ", ".join(t.name for t in all_tools())
    plan_prompt = PLAN.format(
        request=state["user_request"],
        context="(retrieval will occur in the next step)",
        tool_names=tool_names,
    )

    messages = [SystemMessage(content=sys_text), HumanMessage(content=plan_prompt)]
    model = chat_model(temperature=0)
    plan: Plan = await structured(model, Plan).ainvoke(messages)  # type: ignore[assignment]

    return {
        "plan": plan,
        "current_step_index": 0,
        "messages": [HumanMessage(content=f"Plan ready: {plan.rationale}")],
    }


# ---------------------------------------------------------------------------
# retrieve_node
# ---------------------------------------------------------------------------


async def retrieve_node(state: AgentState) -> dict[str, Any]:
    """Retrieve relevant code chunks and past episodes, then assemble context."""
    workspace = _project_root()
    agent_dir = _agent_dir(workspace)

    # Semantic search (best-effort — skip if index not available)
    chunks: list = []
    if _CODE_INDEX_AVAILABLE:
        try:
            chunks = await semantic_search(state["user_request"], k=6)  # type: ignore[name-defined]
        except Exception:
            pass

    # Episode retrieval (best-effort)
    episodes = []
    try:
        from agent.llm.embed import embed_texts
        from agent.memory.episodic import EpisodeStore

        ep_store = EpisodeStore(agent_dir / "state.db")
        vecs = await embed_texts([state["user_request"]])
        if vecs:
            episodes = ep_store.search_semantic(vecs[0], k=3)
    except Exception:
        pass

    # Build system prompt
    project_store = ProjectStore(agent_dir / "state.db")
    facts = project_store.all()
    conventions = "\n".join(f"{f.key}: {f.value}" for f in facts) or "none detected"
    sys_text = SYSTEM.format(
        project_root=str(workspace),
        trust_mode=os.environ.get("AGENT_TRUST_MODE", "trusted"),
        conventions=conventions,
    )

    assembled = assemble(
        user_request=state["user_request"],
        history=list(state["messages"]),
        retrieved_chunks=chunks,
        retrieved_episodes=episodes,
        system_prompt=sys_text,
        budget=_CONTEXT_BUDGET,
        project_root=workspace,
        agent_dir=agent_dir,
    )

    return {
        "retrieved": chunks,
        "messages": assembled,
    }


# ---------------------------------------------------------------------------
# act_node
# ---------------------------------------------------------------------------


async def act_node(state: AgentState) -> dict[str, Any]:
    """Execute the next plan step using tool calls."""
    plan: Plan | None = state["plan"]
    step_index: int = state.get("current_step_index", 0)
    iteration: int = state.get("iteration", 0)

    step_description = "complete the task"
    expected_tools = "write_file, read_file, run_shell, edit_file"
    if plan and plan.steps and step_index < len(plan.steps):
        step = plan.steps[step_index]
        step_description = step.description
        if step.expected_tools:
            expected_tools = ", ".join(step.expected_tools)

    # If the previous act produced no tool calls, use the stronger warning prompt
    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None,
    )
    no_tools_last_time = last_ai is not None and not last_ai.tool_calls

    prompt_template = ACT_NO_TOOLS_WARNING if no_tools_last_time else ACT
    act_prompt = prompt_template.format(
        step_description=step_description,
        expected_tools=expected_tools,
        iteration=iteration + 1,
        max_iterations=_MAX_ITERATIONS,
    )

    messages = list(state["messages"]) + [HumanMessage(content=act_prompt)]
    tools = all_tools()
    model = with_tools(chat_model(temperature=0), tools)

    response: AIMessage = await model.ainvoke(messages)  # type: ignore[assignment]

    new_messages: list[Any] = [response]
    files_changed: list[str] = list(state.get("files_changed", []))

    # Dispatch tool calls
    if response.tool_calls:
        for tc in response.tool_calls:
            result = dispatch(tc)
            tool_msg = ToolMessage(
                content=str(result.output) if result.ok else f"ERROR: {result.error}",
                tool_call_id=tc.get("id", ""),
                name=tc.get("name", ""),
            )
            new_messages.append(tool_msg)

            # Track file-modifying tool calls
            if tc.get("name") in {"write_file", "edit_file", "edit_file_multi"} and result.ok:
                path_arg = tc.get("args", {}).get("path", "")
                if path_arg and path_arg not in files_changed:
                    files_changed.append(path_arg)

    next_step = step_index
    if response.tool_calls and plan and step_index < len(plan.steps) - 1:
        next_step = step_index + 1

    return {
        "messages": new_messages,
        "iteration": iteration + 1,
        "current_step_index": next_step,
        "files_changed": files_changed,
    }


# ---------------------------------------------------------------------------
# verify_node
# ---------------------------------------------------------------------------


async def verify_node(state: AgentState) -> dict[str, Any]:
    """Run the verifier on changed files and feed results back."""
    workspace = _project_root()
    files_changed = state.get("files_changed", [])

    if not files_changed:
        return {"verifier_failures": state.get("verifier_failures", 0)}

    result = run_verifier(files_changed, workspace)

    if result.passed:
        return {"verifier_failures": state.get("verifier_failures", 0)}

    failures = state.get("verifier_failures", 0) + 1
    feedback = VERIFY_FAILURE.format(
        verifier_output=result.output,
        files_changed=str(files_changed),
        retries_left=_MAX_VERIFIER_RETRIES - failures,
    )
    return {
        "messages": [HumanMessage(content=feedback)],
        "verifier_failures": failures,
    }


# ---------------------------------------------------------------------------
# commit_node
# ---------------------------------------------------------------------------


async def commit_node(state: AgentState) -> dict[str, Any]:
    """Finalise the task: write episode, produce summary, set outcome."""
    workspace = _project_root()
    agent_dir = _agent_dir(workspace)

    outcome = state.get("outcome", "pending")
    if outcome == "pending":
        outcome = "success"

    # Build summary via LLM (best-effort)
    from agent.llm.prompts import COMMIT

    files_list = ", ".join(state.get("files_changed", [])) or "none"
    commit_prompt = COMMIT.format(
        request=state["user_request"],
        action_count=state.get("iteration", 0),
        outcome=outcome,
        files_changed=files_list,
    )
    try:
        model = chat_model(temperature=0.2)
        summary_msg: AIMessage = await model.ainvoke(  # type: ignore[assignment]
            [HumanMessage(content=commit_prompt)]
        )
        summary = summary_msg.content if isinstance(summary_msg.content, str) else str(summary_msg.content)
    except Exception:
        summary = f"Task completed with outcome: {outcome}"

    # Persist episode (best-effort)
    try:
        actions = [
            {"type": "tool", "name": tc.get("name"), "args": tc.get("args")}
            for msg in state["messages"]
            if isinstance(msg, AIMessage)
            for tc in (msg.tool_calls or [])
        ]
        ep = Episode(
            id=state.get("task_id") or str(uuid.uuid4()),
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
            request=state["user_request"],
            plan=state["plan"].model_dump() if state.get("plan") else {},
            actions=actions,
            outcome=outcome,  # type: ignore[arg-type]
        )
        ep_store = EpisodeStore(agent_dir / "state.db")
        ep_store.save(ep)
    except Exception:
        pass

    return {
        "outcome": outcome,
        "messages": [AIMessage(content=summary)],
    }
