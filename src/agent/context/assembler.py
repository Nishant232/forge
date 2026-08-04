"""Context assembler: builds the ordered message list sent to the LLM.

Priority order for budget trimming (drop lowest priority first):
  1. Episodes HumanMessage (past tasks, nice-to-have)
  2. Low-score code chunks (sorted ascending; pop from tail)
  3. Oldest history turns (pop from front of the tail slice)

A backtick symbol guarantee: any symbol name found in back-ticks in
*user_request* is looked up in the symbol index and its source file prepended
to retrieved_chunks if not already present.

Recency boost: chunks whose source file was modified in the last hour receive
a +0.3 score boost (capped at 1.0) before sorting.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent.code_index.search import CodeChunk
from agent.context.budgeter import Budgeter
from agent.llm.prompts import CONTEXT_BLOCK, EPISODE_BLOCK
from agent.memory.episodic import Episode

_BACKTICK_RE = re.compile(r"`([^`]+)`")
_RECENCY_WINDOW_SECS = 3600  # 1 hour


def _format_chunk(chunk: CodeChunk) -> str:
    header = f"# {chunk.path}:{chunk.start_line}-{chunk.end_line}"
    if chunk.symbol:
        header += f"  ({chunk.symbol})"
    return f"{header}\n{chunk.content}"


def _format_episode(ep: Episode) -> str:
    outcome = ep.outcome
    return (
        f"Task: {ep.request}\n"
        f"Outcome: {outcome}\n"
        f"Files: {[a.get('path', '') for a in ep.actions if 'path' in a]}"
    )


def _apply_recency_boost(
    chunks: list[CodeChunk],
    project_root: Path | None,
    now: float,
) -> list[CodeChunk]:
    """Return new list with recency boost applied; does not mutate originals."""
    result = []
    for chunk in chunks:
        path = Path(chunk.path)
        if project_root and not path.is_absolute():
            path = project_root / path
        try:
            mtime = path.stat().st_mtime
        except OSError:
            result.append(chunk)
            continue
        if now - mtime <= _RECENCY_WINDOW_SECS:
            boosted = min(chunk.score + 0.3, 1.0)
            result.append(chunk.model_copy(update={"score": boosted}))
        else:
            result.append(chunk)
    return result


def _backtick_symbols(user_request: str) -> list[str]:
    return _BACKTICK_RE.findall(user_request)


def _guarantee_backtick_chunks(
    user_request: str,
    chunks: list[CodeChunk],
    agent_dir: Path | None,
) -> list[CodeChunk]:
    """Prepend chunks for any backtick symbol not already present."""
    if agent_dir is None:
        return chunks

    symbols = _backtick_symbols(user_request)
    if not symbols:
        return chunks

    already_covered = {c.symbol for c in chunks if c.symbol}
    extra: list[CodeChunk] = []

    try:
        from agent.code_index.symbols import SymbolStore

        store = SymbolStore(agent_dir)
        for sym in symbols:
            if sym in already_covered:
                continue
            rows = store.lookup(sym)
            for row in rows:
                extra.append(
                    CodeChunk(
                        path=row.path,
                        start_line=row.start_line,
                        end_line=row.end_line,
                        symbol=row.name,
                        kind=row.kind,  # type: ignore[arg-type]
                        content=row.source or "",
                        score=1.0,
                    )
                )
                already_covered.add(sym)
    except Exception:
        pass  # symbol store not available — degrade gracefully

    return extra + chunks


def assemble(
    user_request: str,
    history: list[BaseMessage],
    retrieved_chunks: list[CodeChunk],
    retrieved_episodes: list[Episode],
    system_prompt: str,
    budget: int,
    project_root: Path | None = None,
    agent_dir: Path | None = None,
    max_history_turns: int = 20,
) -> list[BaseMessage]:
    """Build the ordered message list for the next LLM call.

    Args:
        user_request: The raw user task string.
        history: Prior conversation turns (HumanMessage / AIMessage pairs).
        retrieved_chunks: Code chunks from semantic/symbol search.
        retrieved_episodes: Past episodes from episodic memory.
        system_prompt: Already-formatted system prompt string.
        budget: Hard token ceiling in tokens.
        project_root: Absolute path to the project root (for file mtime checks).
        agent_dir: Absolute path to the .agent/ directory (for symbol lookups).
        max_history_turns: Soft cap on history entries before trimming starts.

    Returns:
        Ordered list of BaseMessage ready for model.invoke().
    """
    budgeter = Budgeter(budget)
    now = time.time()

    # --- Recency boost + backtick guarantee -----------------------------------
    chunks = _apply_recency_boost(retrieved_chunks, project_root, now)
    chunks = _guarantee_backtick_chunks(user_request, chunks, agent_dir)

    # Sort descending by score so high-value chunks stay longest.
    chunks = sorted(chunks, key=lambda c: c.score, reverse=True)

    # --- Build candidate messages ---------------------------------------------
    system_msg = SystemMessage(content=system_prompt)

    # Tail of history (most recent max_history_turns entries).
    history_tail = list(history[-max_history_turns:]) if history else []

    # Chunks block
    chunks_text = "\n\n".join(_format_chunk(c) for c in chunks)
    context_msg: HumanMessage | None = (
        HumanMessage(content=CONTEXT_BLOCK.format(chunks=chunks_text))
        if chunks_text.strip()
        else None
    )

    # Episodes block
    episodes_text = "\n\n".join(_format_episode(e) for e in retrieved_episodes)
    episode_msg: HumanMessage | None = (
        HumanMessage(content=EPISODE_BLOCK.format(episodes=episodes_text))
        if episodes_text.strip()
        else None
    )

    user_msg = HumanMessage(content=user_request)

    # --- Assemble and trim ----------------------------------------------------
    def _build(
        hist: list[BaseMessage],
        ctx: HumanMessage | None,
        ep: HumanMessage | None,
        cks: list[CodeChunk],
    ) -> list[BaseMessage]:
        parts: list[BaseMessage] = [system_msg] + hist
        if ctx is not None:
            # Rebuild context from current chunk list
            text = "\n\n".join(_format_chunk(c) for c in cks)
            if text.strip():
                parts.append(HumanMessage(content=CONTEXT_BLOCK.format(chunks=text)))
        if ep is not None:
            parts.append(ep)
        parts.append(user_msg)
        return parts

    messages = _build(history_tail, context_msg, episode_msg, chunks)

    # Drop episodes first
    if not budgeter.fits(messages) and episode_msg is not None:
        episode_msg = None
        messages = _build(history_tail, context_msg, None, chunks)

    # Drop lowest-score chunks one at a time
    while not budgeter.fits(messages) and chunks:
        chunks.pop()  # remove lowest-score chunk (list is sorted descending)
        messages = _build(history_tail, context_msg if chunks else None, episode_msg, chunks)

    # Drop oldest history turns one at a time
    while not budgeter.fits(messages) and history_tail:
        history_tail.pop(0)
        messages = _build(history_tail, context_msg if chunks else None, episode_msg, chunks)

    # If still over budget (system + user alone are too big), return just those
    if not budgeter.fits(messages):
        messages = [system_msg, user_msg]

    return messages
