"""FROZEN M2 contract. The Governor calls this and must never learn which
implementation is behind it.

    M2(state, reasoning_budget) -> M2Result

This is the seam that decides whether we built a general resource-allocation
layer or a synthetic benchmark policy. If swapping MathM2 for an LLM requires
touching the Governor, the abstraction failed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class M2Result:
    result: object                 # the reasoning output (answer, action, ...)
    reasoning_tokens: float = 0.0  # tokens spent on internal reasoning
    total_tokens: float = 0.0      # reasoning + output
    latency_s: float = 0.0
    cost_units: float = 1.0        # what the budget is denominated in
    ok: bool = True
    error: str = ""


class M2Engine(Protocol):
    """Any deep-reasoning arm. MathM2 today, an LLM tomorrow."""
    name: str
    def __call__(self, state: dict, reasoning_budget: float) -> M2Result: ...


class MathM2:
    """Reference implementation: Environment 6's deterministic deep mode.

    Kept as the regression anchor. If an LLM adapter is wired in and the
    Governor's held-out numbers move, MathM2 re-run must reproduce the frozen
    reference exactly -- otherwise the change is in shared code, not the engine.
    """
    name = "math_m2"

    def __call__(self, state: dict, reasoning_budget: float) -> M2Result:
        if reasoning_budget < 1.0:
            return M2Result(result=None, cost_units=0.0, ok=False,
                            error="insufficient budget")
        return M2Result(result="deep", cost_units=1.0)
