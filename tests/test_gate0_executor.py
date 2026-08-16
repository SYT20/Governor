"""Gate 0 regression tests — the executor must actually execute.

Run I scored policies by summing local proxies along a frozen H trajectory.
These tests make that class of bug fail loudly.
"""
from __future__ import annotations

import pytest

from governor.gate.executor import run_episode
from governor.gate.positive_control import PositiveControlEnv

ENV = PositiveControlEnv(seed=0, n=20)
ALL_H = lambda o, b: "H"          # noqa: E731
ALL_M2 = lambda o, b: "M2" if b >= 1.0 else "H"   # noqa: E731


def test_mode_changes_trajectory():
    """Different modes must produce different traces and utilities."""
    a = run_episode(ENV, ALL_H, 0, 2.0)
    b = run_episode(ENV, ALL_M2, 0, 2.0)
    assert a.modes != b.modes
    assert a.utility != b.utility


def test_compute_is_actually_charged():
    a = run_episode(ENV, ALL_H, 0, 2.0)
    b = run_episode(ENV, ALL_M2, 0, 2.0)
    assert a.spent == 0.0
    assert b.spent == 2.0                      # budget exhausted, not exceeded


def test_budget_is_hard():
    """A policy that overspends must raise, not silently truncate."""
    greedy = lambda o, b: "M2"                 # noqa: E731  ignores budget
    with pytest.raises(RuntimeError, match="overspent"):
        run_episode(ENV, greedy, 0, 1.0)


def test_utility_comes_from_execution_not_proxy():
    """Utility must equal the environment's terminal utility of the executed
    trajectory, not any precomputed score."""
    tr = run_episode(ENV, ALL_M2, 0, 2.0)
    s = ENV.reset(0)
    for m in tr.modes:
        s, _ = ENV.step(s, m)
    assert tr.utility == ENV.utility(s)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="unknown mode"):
        run_episode(ENV, lambda o, b: "TELEPORT", 0, 2.0)


def test_policy_sees_remaining_budget():
    seen = []
    def p(o, b):
        seen.append(b)
        return "M2" if b >= 1.0 else "H"
    run_episode(ENV, p, 0, 2.0)
    assert seen == [2.0, 1.0, 0.0, 0.0]        # budget decreases as spent
