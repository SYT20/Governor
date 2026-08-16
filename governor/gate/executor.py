"""Gate 0 — the ONE canonical episode executor. Nothing may score a policy
except by running it through `run_episode`.

Run I failed because scoring and execution were separate code paths: it advanced
every trajectory under H, then scored each policy by summing local proxies from
that single trajectory. Making execution the only scoring path removes the
possibility rather than warning against it.

A policy is `fn(state, budget_left) -> mode`. The environment owns transitions,
utility and cost. The executor owns the loop and the budget accounting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol


class Env(Protocol):
    n_decisions: int
    def reset(self, ep: int): ...
    def observe(self, s) -> dict: ...
    def step(self, s, mode: str): ...          # -> (s2, cost)
    def utility(self, s) -> float: ...
    def modes(self) -> list[str]: ...


@dataclass(slots=True)
class Trace:
    modes: list[str] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    utility: float = 0.0
    spent: float = 0.0


def run_episode(env: Env, policy: Callable, ep: int, budget: float) -> Trace:
    """Execute a policy. The ONLY authoritative source of policy value."""
    s = env.reset(ep)
    tr = Trace()
    for _ in range(env.n_decisions):
        left = budget - tr.spent
        mode = policy(env.observe(s), left)
        if mode not in env.modes():
            raise ValueError(f"policy returned unknown mode {mode!r}")
        s, cost = env.step(s, mode)
        if tr.spent + cost > budget + 1e-9:      # budget is HARD
            raise RuntimeError(f"policy overspent: {tr.spent + cost} > {budget}")
        tr.modes.append(mode)
        tr.costs.append(cost)
        tr.spent += cost
    tr.utility = env.utility(s)
    return tr
