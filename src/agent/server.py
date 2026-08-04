"""FastAPI server — exposes the ReAct agent loop as a streaming HTTP API.

Run with:  agent serve  (or uvicorn agent.server:app --host 0.0.0.0 --port 8080)
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Local Coding Agent", version="1.0.0")

_WORKSPACE = Path(os.environ.get("AGENT_PROJECT_ROOT", "/workspace"))
_TRUST = os.environ.get("AGENT_TRUST_MODE", "trusted")


def _make_history():
    from agent.core.history import ConversationHistory
    agent_dir = _WORKSPACE / ".agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    return ConversationHistory(agent_dir / "history.db")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "workspace": str(_WORKSPACE)}


@app.get("/sessions")
async def list_sessions() -> list[dict]:
    history = _make_history()
    try:
        sessions = history.list_sessions()
        return [
            {
                "session_id": s.get("id", ""),
                "title": s.get("title") or "Untitled",
                "created_at": s.get("created_at", ""),
                "updated_at": s.get("updated_at", ""),
            }
            for s in sessions
        ]
    finally:
        history.close()


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    from agent.core.loop import run_turn

    history = _make_history()
    session_id = req.session_id or str(uuid.uuid4())
    if not req.session_id:
        history.create_session(session_id)

    async def event_stream() -> AsyncIterator[str]:
        queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()

        def on_content(text: str) -> None:
            queue.put_nowait(("content", text))

        def on_content_token(token: str) -> None:
            queue.put_nowait(("token", token))

        def on_tool_start(name: str, args: dict) -> None:
            queue.put_nowait(("tool_start", {"name": name, "args": args}))

        def on_tool_end(name: str, output: str, ok: bool) -> None:
            queue.put_nowait(("tool_end", {"name": name, "output": output[-400:], "ok": ok}))

        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

        task = asyncio.create_task(
            run_turn(
                req.message,
                workspace=_WORKSPACE,
                session_id=session_id,
                history=history,
                trust=_TRUST,  # type: ignore[arg-type]
                on_content=on_content,
                on_content_token=on_content_token,
                on_tool_start=on_tool_start,
                on_tool_end=on_tool_end,
            )
        )

        while not task.done():
            try:
                event_type, data = await asyncio.wait_for(queue.get(), timeout=0.05)
                yield f"data: {json.dumps({'type': event_type, 'data': data})}\n\n"
            except asyncio.TimeoutError:
                pass

        while not queue.empty():
            event_type, data = queue.get_nowait()
            yield f"data: {json.dumps({'type': event_type, 'data': data})}\n\n"

        result = task.result()
        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'files_changed': result.files_changed, 'tool_calls_made': result.tool_calls_made})}\n\n"

        history.close()

    return StreamingResponse(event_stream(), media_type="text/event-stream")
