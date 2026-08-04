"""LangGraph StateGraph definition and public run_task() entrypoint."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from agent.orchestrator.nodes import (
    act_node,
    commit_node,
    plan_node,
    retrieve_node,
    verify_node,
)
from agent.orchestrator.routing import after_act, after_plan, after_verify
from agent.orchestrator.state import AgentState
from agent.safety.policy import TrustMode


class TaskResult(BaseModel):
    task_id: str
    outcome: str
    summary: str
    files_changed: list[str]
    iterations: int


async def _make_checkpointer(workspace: Path) -> tuple[Any, Any]:
    """Return (checkpointer, connection_or_None) for cleanup after the graph runs."""
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        import aiosqlite

        db_path = workspace / ".agent" / "checkpoints.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(db_path))
        return AsyncSqliteSaver(conn), conn
    except Exception:
        return MemorySaver(), None


def build_graph(checkpointer: Any | None = None) -> Any:
    """Assemble and compile the StateGraph."""
    builder: StateGraph = StateGraph(AgentState)  # type: ignore[type-arg]

    builder.add_node("planner", plan_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("act", act_node)
    builder.add_node("verify", verify_node)
    builder.add_node("commit", commit_node)

    builder.set_entry_point("planner")

    builder.add_conditional_edges("planner", after_plan, {"retrieve": "retrieve"})
    builder.add_edge("retrieve", "act")
    builder.add_conditional_edges(
        "act",
        after_act,
        {"act": "act", "verify": "verify", "commit": "commit"},
    )
    builder.add_conditional_edges(
        "verify",
        after_verify,
        {"act": "act", "commit": "commit"},
    )
    builder.add_edge("commit", END)

    return builder.compile(checkpointer=checkpointer)


async def run_task(
    request: str,
    *,
    workspace: Path,
    trust: TrustMode = "trusted",
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> TaskResult:
    """Run a task against the compiled graph and return a TaskResult."""
    import os

    # Resolve symlinks (e.g. /tmp → /private/tmp on macOS) so all nodes
    # that call _project_root() see the same absolute path.
    workspace = workspace.resolve()
    os.environ["AGENT_PROJECT_ROOT"] = str(workspace)
    os.environ["AGENT_TRUST_MODE"] = trust

    # Ensure .agent/ exists before nodes open SQLite databases inside it.
    (workspace / ".agent").mkdir(parents=True, exist_ok=True)

    task_id = str(uuid.uuid4())
    checkpointer, _conn = await _make_checkpointer(workspace)
    graph = build_graph(checkpointer)

    initial_state: AgentState = {  # type: ignore[misc]
        "messages": [HumanMessage(content=request)],
        "user_request": request,
        "plan": None,
        "current_step_index": 0,
        "retrieved": [],
        "iteration": 0,
        "verifier_failures": 0,
        "task_id": task_id,
        "outcome": "pending",
        "files_changed": [],
    }

    config = {"configurable": {"thread_id": task_id}}
    final_state: AgentState | None = None  # type: ignore[assignment]

    try:
        async for event in graph.astream_events(initial_state, config=config, version="v2"):
            if on_event:
                on_event(event)
            if event.get("event") == "on_chain_end" and event.get("name") == "LangGraph":
                data = event.get("data", {})
                final_state = data.get("output")

        if final_state is None:
            final_state = await graph.ainvoke(initial_state, config=config)  # type: ignore[assignment]
    finally:
        if _conn is not None:
            await _conn.close()

    outcome = final_state.get("outcome", "unknown") if final_state else "unknown"  # type: ignore[union-attr]
    files_changed = final_state.get("files_changed", []) if final_state else []  # type: ignore[union-attr]
    iterations = final_state.get("iteration", 0) if final_state else 0  # type: ignore[union-attr]

    summary = f"Task {task_id} completed: {outcome}"
    if final_state:
        for msg in reversed(final_state["messages"]):  # type: ignore[union-attr]
            from langchain_core.messages import AIMessage

            if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content:
                summary = msg.content
                break

    return TaskResult(
        task_id=task_id,
        outcome=outcome,
        summary=summary,
        files_changed=files_changed,
        iterations=iterations,
    )


async def resume_task(
    task_id: str,
    *,
    workspace: Path,
    trust: TrustMode = "trusted",
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> TaskResult:
    """Resume an interrupted task from its LangGraph checkpoint."""
    import os

    workspace = workspace.resolve()
    os.environ["AGENT_PROJECT_ROOT"] = str(workspace)
    os.environ["AGENT_TRUST_MODE"] = trust
    (workspace / ".agent").mkdir(parents=True, exist_ok=True)

    checkpointer, _conn = await _make_checkpointer(workspace)
    graph = build_graph(checkpointer)
    config = {"configurable": {"thread_id": task_id}}

    final_state: AgentState | None = None  # type: ignore[assignment]
    try:
        async for event in graph.astream_events(None, config=config, version="v2"):  # type: ignore[arg-type]
            if on_event:
                on_event(event)
            if event.get("event") == "on_chain_end" and event.get("name") == "LangGraph":
                data = event.get("data", {})
                final_state = data.get("output")
    finally:
        if _conn is not None:
            await _conn.close()

    outcome = final_state.get("outcome", "unknown") if final_state else "unknown"  # type: ignore[union-attr]
    files_changed = final_state.get("files_changed", []) if final_state else []  # type: ignore[union-attr]
    iterations = final_state.get("iteration", 0) if final_state else 0  # type: ignore[union-attr]
    summary = f"Task {task_id} resumed: {outcome}"

    return TaskResult(
        task_id=task_id,
        outcome=outcome,
        summary=summary,
        files_changed=files_changed,
        iterations=iterations,
    )
