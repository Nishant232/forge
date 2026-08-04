"""Token budgeter using tiktoken cl100k_base with a 15% safety margin.

The safety margin compensates for Qwen's tendency to count tokens slightly
differently from the OpenAI tokenizer used here as a proxy.
"""
from __future__ import annotations

import math
from typing import Sequence

import tiktoken
from langchain_core.messages import BaseMessage

_ENCODING = tiktoken.get_encoding("cl100k_base")

# Overhead that GPT-style chat completions add per message (role + delimiters).
_TOKENS_PER_MESSAGE = 4


class Budgeter:
    """Counts tokens and enforces a context budget.

    Args:
        budget: Hard token ceiling (e.g. 32_768 for a 32k context).
    """

    SAFETY_MARGIN: float = 0.85  # 15% buffer for Qwen tokeniser variance

    def __init__(self, budget: int) -> None:
        if budget <= 0:
            raise ValueError(f"budget must be positive, got {budget}")
        self._budget = budget

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def effective_budget(self) -> int:
        """Floor of budget * safety margin — the real ceiling to target."""
        return math.floor(self._budget * self.SAFETY_MARGIN)

    def count(self, text: str) -> int:
        """Return the tiktoken cl100k_base token count for *text*."""
        return len(_ENCODING.encode(text))

    def count_messages(self, messages: Sequence[BaseMessage]) -> int:
        """Return the total token count for a list of LangChain messages.

        Adds _TOKENS_PER_MESSAGE overhead per message to mirror the encoding
        format used by OpenAI-style models.
        """
        total = 0
        for msg in messages:
            total += _TOKENS_PER_MESSAGE
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            total += self.count(content)
        return total

    def fits(self, messages: Sequence[BaseMessage]) -> bool:
        """Return True if *messages* fit within the effective budget."""
        return self.count_messages(messages) <= self.effective_budget
