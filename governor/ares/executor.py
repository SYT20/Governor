"""Step 10 — Ares, the execution layer, with a frozen per-action interface.

    execute(action, state, budget) -> ExecResult(observation, utility,
                                                 consumed, state)

WHY A SECOND EXECUTOR AT ALL. `governor.gate.executor.run_episode` is the
canonical episode loop and is frozen; env6's reference numbers depend on it and
it must not move. What it does not expose is a single ACTION, which is what an
agent framework actually calls. Ares provides that, and `AresLoop.run` rebuilds
the episode from it.

The claim "these are the same execution path" is not asserted in prose. A test
runs both over the same environment and policies and requires byte-identical
modes, costs, spend and utility. If they ever diverge, the test fails rather
than two subtly different numbers appearing in two documents.

Ares imports nothing from `governor.phase4.policies` or the predictor, and the
test asserts that too: an executor that knows what a Governor is can be tuned
to flatter one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class Executable(Protocol):
    """What Ares needs from an environment. Deliberately smaller than `Env`."""
    n_decisions: int
    def reset(self, ep: int): ...
    def observe(self, s) -> dict: ...
    def step(self, s, action: str): ...          # -> (s2, cost)
    def utility(self, s) -> float: ...
    def modes(self) -> list[str]: ...


class BudgetExceeded(RuntimeError):
    """Raised instead of executing an action that does not fit."""


class UnknownAction(ValueError):
    pass


@dataclass(slots=True)
class ExecResult:
    observation: dict
    utility: float
    consumed: dict[str, float]
    state: Any
    ok: bool = True
    error: str = ""


@dataclass(slots=True)
class Episode:
    actions: list[str] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    spent: float = 0.0
    utility: float = 0.0
    observations: list[dict] = field(default_factory=list)


class Ares:
    """Executes one action at a time and accounts for what it cost.

    The budget check happens BEFORE the action, using the environment's own
    worst-case bound where it publishes one (`cap`). Checking afterwards would
    mean the resource was already consumed, which is not a budget.
    """

    def __init__(self, env: Executable):
        self.env = env

    def actions(self) -> list[str]:
        return list(self.env.modes())

    def affordable(self, action: str, state, budget_left: float) -> bool:
        cap = getattr(self.env, "cap", None)
        if cap is None:
            return True
        items_left = self.env.observe(state).get("items_left", 1)
        feasible = getattr(self.env, "feasible", None)
        if feasible is not None:
            return bool(feasible(action, budget_left, items_left))
        return budget_left >= cap(action)

    def execute(self, action: str, state, budget_left: float) -> ExecResult:
        if action not in self.env.modes():
            raise UnknownAction(f"unknown action {action!r}; "
                                f"available: {self.env.modes()}")
        if not self.affordable(action, state, budget_left):
            return ExecResult(observation=self.env.observe(state),
                              utility=self.env.utility(state),
                              consumed={"tokens": 0.0}, state=state,
                              ok=False, error="action does not fit the budget")
        s2, cost = self.env.step(state, action)
        if cost > budget_left + 1e-9:
            raise BudgetExceeded(f"{action} cost {cost} > {budget_left} left")
        return ExecResult(observation=self.env.observe(s2),
                          utility=self.env.utility(s2),
                          consumed={"tokens": float(cost)}, state=s2)


class AresLoop:
    """The episode loop rebuilt on `execute`. Asserted identical to
    `run_episode` by tests/test_ares.py."""

    def __init__(self, env: Executable):
        self.ares = Ares(env)
        self.env = env

    def run(self, policy: Callable[[dict, float], str], ep: int,
            budget: float) -> Episode:
        s = self.env.reset(ep)
        e = Episode()
        for _ in range(self.env.n_decisions):
            obs = self.env.observe(s)
            e.observations.append(obs)
            action = policy(obs, budget - e.spent)
            r = self.ares.execute(action, s, budget - e.spent)
            if not r.ok:
                raise BudgetExceeded(r.error)
            s = r.state
            e.actions.append(action)
            e.costs.append(r.consumed["tokens"])
            e.spent += r.consumed["tokens"]
        e.utility = self.env.utility(s)
        return e
