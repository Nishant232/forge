"""LLM provider — Ollama (local).

Configure via environment variables:
  OLLAMA_URL     Ollama server URL (default: http://localhost:11434)
  CHAT_MODEL     Model name        (default: qwen2.5:14b)
  OLLAMA_TIMEOUT Request timeout in seconds (default: 120)

Three public helpers cover every LLM interaction pattern:
  chat_model()  — plain chat client
  with_tools()  — binds tools so the model emits tool_calls
  structured()  — validates LLM output against a Pydantic schema
"""
from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel

# Single source of truth for the default model name.
# Recommended: any Ollama model with reliable tool-call support.
# qwen2.5:14b is a well-tested choice; qwen3-coder variants work too.
DEFAULT_MODEL: str = "qwen2.5:14b"

_CHAT_MODEL: str = os.environ.get("CHAT_MODEL", DEFAULT_MODEL)


def chat_model(temperature: float = 0) -> BaseChatModel:
    """Return a configured ChatOllama instance.

    Args:
        temperature: Sampling temperature (0 = deterministic for planning/tools,
                     0.2 for free-form text generation).
    """
    from langchain_ollama import ChatOllama

    model_name = os.environ.get("CHAT_MODEL", _CHAT_MODEL)
    # Disable qwen3's extended thinking — it produces <think>...</think> tokens
    # that break LangChain's tool-call parsing.
    extra: dict[str, Any] = {}
    if "qwen3" in model_name.lower():
        extra["think"] = False

    return ChatOllama(
        model=model_name,
        base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        temperature=temperature,
        timeout=int(os.environ.get("OLLAMA_TIMEOUT", "120")),
        **extra,
    )


def with_tools(model: BaseChatModel, tools: list[BaseTool]) -> Runnable:
    """Return the model with the given tools bound."""
    return model.bind_tools(tools)


def structured(model: BaseChatModel, schema: type[BaseModel]) -> Runnable:
    """Return a Runnable that validates LLM output against *schema*."""
    return model.with_structured_output(schema)
