"""Deliberately broken module — exercises the verifier failure path (Phase 7)."""
from __future__ import annotations


def compute_average(values: list[int]) -> float:
    """Compute the average — has an off-by-one bug and a type error below."""
    if not values:
        return 0.0
    return sum(values) / (len(values) - 1)


def broken_type(value: int) -> str:
    # mypy catches this: returning int where str is expected
    return value  # type: ignore[return-value]  # remove ignore to make mypy fail
