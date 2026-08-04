"""Pydantic models for structured LLM output.

These schemas are passed to `structured()` in llm/chat.py via
`.with_structured_output()`. They are also used by the orchestrator nodes
to type-check plan data and final task results.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """A single step in an agent execution plan."""

    description: str = Field(description="What this step does in plain English")
    expected_tools: list[str] = Field(
        description="Names of tools likely used in this step"
    )


class Plan(BaseModel):
    """A structured execution plan produced by the planning node."""

    steps: list[PlanStep] = Field(description="Ordered list of steps to execute")
    rationale: str = Field(
        description="One sentence explaining the overall approach"
    )


class FinalAnswer(BaseModel):
    """Terminal record written by the commit node at the end of a task."""

    summary: str = Field(description="What was accomplished or attempted")
    outcome: Literal["success", "partial", "failure"] = Field(
        description="Overall task outcome"
    )
    files_changed: list[str] = Field(
        default_factory=list,
        description="Absolute paths of files that were modified",
    )
