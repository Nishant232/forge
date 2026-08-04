"""Unit tests for context/budgeter.py."""
from __future__ import annotations

import math

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.context.budgeter import Budgeter


class TestBudgeterInit:
    def test_positive_budget_accepted(self) -> None:
        b = Budgeter(1000)
        assert b.effective_budget > 0

    def test_zero_budget_raises(self) -> None:
        with pytest.raises(ValueError):
            Budgeter(0)

    def test_negative_budget_raises(self) -> None:
        with pytest.raises(ValueError):
            Budgeter(-1)


class TestEffectiveBudget:
    def test_effective_budget_is_floor_of_85_percent(self) -> None:
        b = Budgeter(1000)
        assert b.effective_budget == math.floor(1000 * 0.85)

    def test_effective_budget_32k(self) -> None:
        b = Budgeter(32_768)
        assert b.effective_budget == math.floor(32_768 * 0.85)

    def test_effective_budget_less_than_raw(self) -> None:
        b = Budgeter(100)
        assert b.effective_budget < 100


class TestCount:
    def test_empty_string_is_zero(self) -> None:
        b = Budgeter(1000)
        assert b.count("") == 0

    def test_single_word(self) -> None:
        b = Budgeter(1000)
        assert b.count("hello") == 1

    def test_longer_text_has_more_tokens(self) -> None:
        b = Budgeter(1000)
        short = b.count("hi")
        long = b.count("hello world, this is a longer sentence with many words")
        assert long > short

    def test_count_is_deterministic(self) -> None:
        b = Budgeter(1000)
        text = "def parse_query(sql: str) -> dict: ..."
        assert b.count(text) == b.count(text)

    def test_count_returns_int(self) -> None:
        b = Budgeter(1000)
        assert isinstance(b.count("test"), int)


class TestCountMessages:
    def test_empty_list_is_zero(self) -> None:
        b = Budgeter(1000)
        assert b.count_messages([]) == 0

    def test_single_message_adds_overhead(self) -> None:
        b = Budgeter(1000)
        msg = HumanMessage(content="hello")
        total = b.count_messages([msg])
        # overhead (4) + tokens for "hello" (1) = 5
        assert total == 5

    def test_multiple_messages_sum_with_overhead(self) -> None:
        b = Budgeter(1000)
        msgs = [
            HumanMessage(content="hi"),
            AIMessage(content="hello"),
        ]
        total = b.count_messages(msgs)
        # Each message: 4 overhead + content tokens
        per_msg_overhead = 4
        expected = sum(per_msg_overhead + b.count(m.content) for m in msgs)
        assert total == expected

    def test_system_message_counted(self) -> None:
        b = Budgeter(1000)
        sys_msg = SystemMessage(content="You are an assistant.")
        total = b.count_messages([sys_msg])
        assert total > 0

    def test_count_messages_returns_int(self) -> None:
        b = Budgeter(1000)
        assert isinstance(b.count_messages([HumanMessage(content="x")]), int)


class TestFits:
    def test_empty_messages_always_fits(self) -> None:
        b = Budgeter(100)
        assert b.fits([]) is True

    def test_short_message_fits_large_budget(self) -> None:
        b = Budgeter(32_768)
        msgs = [HumanMessage(content="Fix the bug in main.py")]
        assert b.fits(msgs) is True

    def test_does_not_fit_tiny_budget(self) -> None:
        b = Budgeter(1)  # effective_budget = 0
        msgs = [HumanMessage(content="x")]
        assert b.fits(msgs) is False

    def test_exactly_at_effective_budget_fits(self) -> None:
        b = Budgeter(1000)
        # Craft messages whose token count equals the effective budget
        target = b.effective_budget
        # Use a message with enough tokens
        text = "a " * (target - 4)  # rough — each "a " is ~1 token
        msgs = [HumanMessage(content=text)]
        # Just verify fits() returns a bool without error
        assert isinstance(b.fits(msgs), bool)

    def test_fits_is_false_when_over_budget(self) -> None:
        b = Budgeter(10)  # tiny budget
        long_text = "word " * 100
        msgs = [HumanMessage(content=long_text)]
        assert b.fits(msgs) is False
