"""Core ReAct agent loop.

Features:
  - forge.md: per-project custom instructions injected into every system prompt
  - Streaming: tokens emitted via on_content_token as they arrive
  - Parallel tool dispatch: all tool calls in one response run concurrently
  - Tool output truncation: large outputs capped to prevent context overflow
  - Auto-compact: LLM summarises old messages when history exceeds threshold
  - Context trimming: hard cap as a final safety net
  - Post-edit verification: runs lint/type/tests after changes; loops back on failure
  - Episodic memory: retrieves similar past tasks; saves episode on completion
  - Auto git checkpoint: rollback point before the first file write
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from agent.core.history import ConversationHistory
from agent.llm.chat import chat_model
from agent.memory.project import ProjectStore
from agent.safety.policy import TrustMode
from agent.tools.registry import ToolResult, all_tools

_MAX_ITERATIONS = 40
_MAX_HISTORY_MESSAGES = 80      # hard trim when auto-compact fails
_COMPACT_THRESHOLD = 60         # trigger auto-compact at this many messages
_MAX_TOOL_OUTPUT = 8_000        # truncate tool output beyond this many chars
_MAX_VERIFIER_RETRIES = 2       # max times to loop back for verification failures


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert AI coding assistant running in a local terminal — similar to Claude Code.
You help with any software engineering task: reading, writing, refactoring, debugging, testing.

WORKSPACE: {project_root}
TRUST MODE: {trust_mode}

## Tools available
### File operations
- read_file(path, start_line, end_line)  → read full file or a line range (adds line numbers)
- write_file(path, content)              → create or overwrite a file
- edit_file(path, old_string, new_string)         → replace one exact string
- edit_file_multi(path, old_strings, new_strings) → apply several edits in one call (faster)
- list_dir(path)                         → list directory entries
- find_files(pattern, path)              → glob-based file search ("**/*.py", "src/**/*.ts")

### Shell & search
- run_shell(command)                     → shell commands (pip, make, npm, cargo, …)
- search_code(query)                     → ripgrep full-text search
- run_tests(test_path, extra_args)       → run pytest

### Git
- git_status / git_diff / git_add / git_commit / git_log
- git_checkpoint / git_rollback          → safe undo point before risky changes

### Web (trust=yolo only)
- fetch_url(url)                         → HTTP GET a URL
- search_web(query, n)                   → DuckDuckGo search results

### Agent & task management
- run_subtask(description)               → fresh sub-agent for a well-scoped sub-task
- todo_write(items)                      → set the task checklist (mark done with "[done] ")
- todo_read()                            → read the current checklist

## Working strategy
1. EXPLORE FIRST: list_dir / find_files / search_code before assuming file locations.
2. READ BEFORE EDITING: read_file (with line range for large files) before edit_file.
3. MULTI-EDIT: use edit_file_multi when making several changes to one file.
4. COMPLEX TASKS: call todo_write at the start to outline steps; check off with "[done] ".
5. VERIFY ALWAYS: run tests or execute the code after every change.
6. DECOMPOSE: use run_subtask to delegate a well-defined chunk of work to a sub-agent.
7. RECOVER: read error output carefully; fix root cause; don't retry unchanged commands.

## Rules
- Work only within {project_root}.
- Do NOT ask for permission — just use the tools.
- Prefer edit_file / edit_file_multi over rewriting whole files.
- Always produce working, runnable code.

## Project conventions
{conventions}
{agentx_md}\
"""


def _load_forge_md(workspace: Path) -> str:
    """Read forge.md from project root (max 8 000 chars)."""
    md = workspace / "forge.md"
    if not md.exists():
        # Fallback: also accept legacy agentX.md for backwards compatibility
        md = workspace / "agentX.md"
        if not md.exists():
            return ""
    try:
        text = md.read_text(encoding="utf-8").strip()
        if len(text) > 8_000:
            text = text[:8_000] + "\n[...forge.md truncated at 8 000 chars...]"
        return text
    except Exception:
        return ""


def _build_system(
    workspace: Path,
    trust: TrustMode,
    episode_context: str = "",
) -> SystemMessage:
    agent_dir = workspace / ".agent"

    try:
        facts = ProjectStore(agent_dir / "state.db").all()
        conventions = "\n".join(f"  {f.key}: {f.value}" for f in facts) or "  none detected"
    except Exception:
        conventions = "  none detected"

    agentx_text = _load_forge_md(workspace)
    agentx_section = (
        f"\n## Custom instructions (forge.md)\n{agentx_text}"
        if agentx_text
        else ""
    )

    ep_section = (
        f"\n## Similar past tasks\n{episode_context}"
        if episode_context
        else ""
    )

    return SystemMessage(content=_SYSTEM.format(
        project_root=str(workspace),
        trust_mode=trust,
        conventions=conventions,
        agentx_md=agentx_section + ep_section,
    ))


# ---------------------------------------------------------------------------
# Turn result
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    text: str
    files_changed: list[str] = field(default_factory=list)
    tool_calls_made: int = 0
    iterations: int = 0


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------

async def _call_model(
    model: Any,
    messages: list[BaseMessage],
    on_token: Callable[[str], None] | None,
) -> AIMessage:
    """Call the model with streaming; fall back to ainvoke on failure."""
    if on_token is None:
        response = await model.ainvoke(messages)
        return AIMessage(
            content=response.content if isinstance(response.content, str) else "",
            tool_calls=list(response.tool_calls) if response.tool_calls else [],
        )

    full: Any = None
    try:
        async for chunk in model.astream(messages):
            if isinstance(chunk.content, str) and chunk.content:
                on_token(chunk.content)
            full = chunk if full is None else full + chunk

        if full is None:
            return AIMessage(content="")

        tool_calls = list(full.tool_calls) if full.tool_calls else []

        # Fallback: some providers put tool calls in additional_kwargs
        if not tool_calls and hasattr(full, "additional_kwargs"):
            import json as _json
            for raw in full.additional_kwargs.get("tool_calls", []):
                if isinstance(raw, dict) and "function" in raw:
                    try:
                        tool_calls.append({
                            "id": raw.get("id", ""),
                            "name": raw["function"]["name"],
                            "args": _json.loads(raw["function"].get("arguments", "{}")),
                        })
                    except Exception:
                        pass

        return AIMessage(
            content=full.content if isinstance(full.content, str) else "",
            tool_calls=tool_calls,
        )

    except Exception:
        response = await model.ainvoke(messages)
        text = response.content if isinstance(response.content, str) else ""
        if text and on_token:
            on_token(text)
        return AIMessage(
            content=text,
            tool_calls=list(response.tool_calls) if response.tool_calls else [],
        )


# ---------------------------------------------------------------------------
# Parallel tool dispatch with output truncation
# ---------------------------------------------------------------------------

def _truncate_output(content: str, name: str) -> str:
    """Cap tool output at _MAX_TOOL_OUTPUT to protect the context window."""
    if len(content) <= _MAX_TOOL_OUTPUT:
        return content
    kept = content[:_MAX_TOOL_OUTPUT]
    dropped = len(content) - _MAX_TOOL_OUTPUT
    hint = (
        "Use a more specific path or query to see less output."
        if name in {"search_code", "run_shell", "run_tests"}
        else "Use read_file with start_line/end_line to read specific sections."
    )
    return f"{kept}\n\n[...{dropped} chars truncated. {hint}]"


async def _dispatch_parallel(
    tool_calls: list[Any],
    tool_map: dict[str, Any],
    on_tool_start: Callable[[str, dict], None] | None,
    on_tool_end: Callable[[str, str, bool], None] | None,
) -> list[tuple[Any, ToolResult]]:
    """Dispatch all tool calls concurrently; return (tool_call_dict, result) pairs."""

    async def _one(tc: dict[str, Any]) -> tuple[dict[str, Any], ToolResult]:
        name: str = tc.get("name", "unknown")
        args: dict = tc.get("args") or {}

        if on_tool_start:
            on_tool_start(name, args)

        t = tool_map.get(name)
        if t is None:
            result = ToolResult(ok=False, output="", error=f"Unknown tool: {name!r}", duration_ms=0)
        else:
            start = time.monotonic()
            try:
                output = await asyncio.to_thread(t.invoke, args)
                result = ToolResult(
                    ok=True,
                    output=output,
                    duration_ms=int((time.monotonic() - start) * 1000),
                )
            except Exception as exc:
                result = ToolResult(
                    ok=False, output="", error=str(exc),
                    duration_ms=int((time.monotonic() - start) * 1000),
                )

        content = str(result.output) if result.ok else f"Error: {result.error}"
        content = _truncate_output(content, name)

        if on_tool_end:
            on_tool_end(name, content, result.ok)

        return tc, result

    return list(await asyncio.gather(*[_one(tc) for tc in tool_calls]))


# ---------------------------------------------------------------------------
# Auto-compact history
# ---------------------------------------------------------------------------

async def _maybe_compact(
    prior: list[BaseMessage],
    session_id: str,
    history: ConversationHistory,
) -> list[BaseMessage]:
    """Summarise old messages when history exceeds _COMPACT_THRESHOLD."""
    if len(prior) <= _COMPACT_THRESHOLD:
        return prior

    keep_n = 20
    to_summarise = prior[:-keep_n]
    recent = prior[-keep_n:]

    lines = []
    for m in to_summarise:
        role = type(m).__name__.replace("Message", "").lower()
        text = str(m.content)[:250].replace("\n", " ")
        lines.append(f"{role}: {text}")

    prompt = (
        "Summarise this conversation in 4-6 sentences. Preserve:\n"
        "- the original task\n"
        "- key decisions made\n"
        "- files created or modified\n"
        "- current state and what still needs doing\n\n"
        + "\n".join(lines[:50])
    )

    try:
        model = chat_model(temperature=0.2)
        result = await model.ainvoke([HumanMessage(content=prompt)])
        summary = result.content if isinstance(result.content, str) else str(result.content)

        summary_msg = SystemMessage(
            content=(
                f"[Auto-compacted: {len(to_summarise)} older messages replaced with summary]\n\n"
                f"{summary}"
            )
        )
        # Rewrite stored history
        history.clear_session(session_id)
        history.append(session_id, summary_msg)
        for m in recent:
            history.append(session_id, m)

        return [summary_msg] + list(recent)

    except Exception:
        return prior[-_MAX_HISTORY_MESSAGES:]


# ---------------------------------------------------------------------------
# Episodic memory helpers
# ---------------------------------------------------------------------------

async def _retrieve_episodes(user_message: str, agent_dir: Path) -> str:
    try:
        from agent.llm.embed import embed_texts
        from agent.memory.episodic import EpisodeStore

        vecs = await embed_texts([user_message])
        if not vecs:
            return ""
        store = EpisodeStore(agent_dir / "state.db")
        eps = store.search_semantic(vecs[0], k=2)
        store.close()
        return "\n".join(f"  [{e.outcome}] {e.request[:80]}" for e in eps)
    except Exception:
        return ""


def _save_episode(request: str, files_changed: list[str], tool_calls: int, agent_dir: Path) -> None:
    try:
        from datetime import datetime, timezone
        from agent.memory.episodic import Episode, EpisodeStore

        ep = Episode(
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
            request=request,
            plan={},
            actions=[{"tool_calls": tool_calls, "files": files_changed}],
            outcome="success",
        )
        store = EpisodeStore(agent_dir / "state.db")
        store.save(ep)
        store.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Post-edit verification
# ---------------------------------------------------------------------------

async def _run_verifier(files_changed: list[str], workspace: Path) -> Any:
    """Run verifier in a thread; return result or None on error."""
    try:
        from agent.verify.runner import run_verifier
        return await asyncio.to_thread(run_verifier, files_changed, workspace)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Auto git checkpoint
# ---------------------------------------------------------------------------

def _auto_checkpoint(workspace: Path) -> None:
    try:
        import subprocess
        subprocess.run(
            ["git", "tag", f"agent-cp-auto-{int(time.time())}"],
            capture_output=True, cwd=str(workspace), timeout=10,
        )
        (workspace / ".agent" / "checkpoints").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_turn(
    user_message: str,
    *,
    workspace: Path,
    session_id: str,
    history: ConversationHistory,
    trust: TrustMode = "trusted",
    on_content: Callable[[str], None] | None = None,
    on_content_token: Callable[[str], None] | None = None,
    on_tool_start: Callable[[str, dict[str, Any]], None] | None = None,
    on_tool_end: Callable[[str, str, bool], None] | None = None,
) -> TurnResult:
    """Run one user turn of the ReAct loop."""
    workspace = workspace.resolve()
    os.environ["AGENT_PROJECT_ROOT"] = str(workspace)
    os.environ["AGENT_TRUST_MODE"] = trust
    os.environ["AGENT_SESSION_ID"] = session_id  # used by todo tool

    agent_dir = workspace / ".agent"
    agent_dir.mkdir(parents=True, exist_ok=True)

    # Best-effort episodic retrieval (skipped if embed model unavailable)
    episode_context = await _retrieve_episodes(user_message, agent_dir)

    sys_msg = _build_system(workspace, trust, episode_context)

    # Load history, auto-compact if needed
    prior: list[BaseMessage] = history.load(session_id)
    prior = await _maybe_compact(prior, session_id, history)
    if len(prior) > _MAX_HISTORY_MESSAGES:
        prior = prior[-_MAX_HISTORY_MESSAGES:]

    human_msg = HumanMessage(content=user_message)
    messages: list[BaseMessage] = [sys_msg, *prior, human_msg]
    history.append(session_id, human_msg)

    tools = all_tools()
    tool_map = {t.name: t for t in tools}
    model = chat_model(temperature=0).bind_tools(tools)

    files_changed: list[str] = []
    tool_calls_made = 0
    final_text = ""
    checkpoint_made = False
    verifier_retries = 0
    iterations = 0

    for iterations in range(_MAX_ITERATIONS):
        response = await _call_model(model, messages, on_content_token)
        messages.append(response)

        text = response.content if isinstance(response.content, str) else ""
        if text:
            final_text = text
            if on_content and on_content_token is None:
                on_content(text)

        if not response.tool_calls:
            # ---- Post-edit verification --------------------------------
            if files_changed and verifier_retries < _MAX_VERIFIER_RETRIES:
                vr = await _run_verifier(files_changed, workspace)
                if vr is not None and not vr.passed:
                    verifier_retries += 1
                    retries_left = _MAX_VERIFIER_RETRIES - verifier_retries
                    feedback = (
                        f"Verification failed after your changes. "
                        f"Fix the issues before finishing "
                        f"({retries_left} attempt(s) remaining).\n\n"
                        f"Verifier output:\n{vr.output}\n\n"
                        f"Files changed: {', '.join(files_changed)}"
                    )
                    if on_tool_end:
                        on_tool_end("verifier", vr.output[:300], False)
                    vm = HumanMessage(content=feedback)
                    messages.append(vm)
                    history.append(session_id, response)
                    history.append(session_id, vm)
                    continue  # loop back so model can fix it
            # ---- Truly done --------------------------------------------
            history.append(session_id, response)
            break

        # Auto-checkpoint before first write
        if not checkpoint_made:
            write_tools = {"write_file", "edit_file", "edit_file_multi"}
            if any(tc.get("name") in write_tools for tc in response.tool_calls):
                _auto_checkpoint(workspace)
                checkpoint_made = True

        # Parallel dispatch
        pairs = await _dispatch_parallel(
            response.tool_calls, tool_map, on_tool_start, on_tool_end
        )

        tool_msgs: list[ToolMessage] = []
        for tc, result in pairs:
            name = tc.get("name", "unknown")
            args = tc.get("args") or {}
            tc_id = tc.get("id", "")
            tool_calls_made += 1

            if name in {"write_file", "edit_file", "edit_file_multi"} and result.ok:
                path_arg = args.get("path", "")
                if path_arg and path_arg not in files_changed:
                    files_changed.append(path_arg)

            content = str(result.output) if result.ok else f"Error: {result.error}"
            tool_msgs.append(ToolMessage(content=content, tool_call_id=tc_id, name=name))
            messages.append(tool_msgs[-1])

        history.append(session_id, response)
        history.append_many(session_id, tool_msgs)  # type: ignore[arg-type]

    _save_episode(user_message, files_changed, tool_calls_made, agent_dir)

    return TurnResult(
        text=final_text,
        files_changed=files_changed,
        tool_calls_made=tool_calls_made,
        iterations=iterations + 1,
    )
