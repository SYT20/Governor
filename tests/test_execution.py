"""ActionExecutor is tested INDEPENDENTLY of the Governor, and proven identical to the
frozen executor.

The point of a separate execution layer is that it can be wrong on its own. So:
one test asserts it imports nothing about deciding, and one runs both executors
over the same environment and requires identical traces. "They are the same
path" is otherwise a claim in a docstring, which is the category of claim this
project keeps having to retract.
"""
from __future__ import annotations

import itertools
import pathlib

import numpy as np
import pytest

from governor.execution.executor import (
    ActionExecutor, ExecutorLoop, BudgetExceeded, UnknownAction,
)
from governor.gate.env6 import Env6
from governor.gate.executor import run_episode
from governor.phase4.collect import CallRecord, ResponseCache
from governor.phase4.env import CHEAP, DEEP, P4Env, make_episodes
from governor.phase4.policies import all_cheap, fixed_schedule, greedy
from governor.phase4.tasks import make_pool

LOW, HIGH, PROMPT_CAP = 700, 2800, 128
BUDGET = 5200.0


@pytest.fixture
def p4(tmp_path):
    cache = ResponseCache(tmp_path / "a.sqlite", model="synthetic")
    pool = make_pool(seed=5, n=16)
    for j, it in enumerate(pool):
        for mt in (LOW, HIGH):
            correct = (j % 2 == 0) or mt == HIGH
            used = int(mt * (0.95 if mt == LOW else 0.29))
            cache.put(it, mt, CallRecord(
                it.item_id, mt, str(it.answer) if correct else "0",
                "stop" if correct else "length", 60, used, used - 10,
                60 + used, 0.3), attempts=1)
    return P4Env(cache, make_episodes(pool, 4, 3), LOW, HIGH, BUDGET, PROMPT_CAP)


# -- independence --------------------------------------------------------------

def test_ares_does_not_know_what_a_governor_is():
    """An executor that can see the controller can be tuned to flatter it.

    Checked against the IMPORT GRAPH, not the source text: the first version
    grepped the file and failed on the docstring sentence saying it imports
    nothing from `governor.phase4.policies`.
    """
    import ast
    import governor.execution.executor as mod
    tree = ast.parse(pathlib.Path(mod.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for m in imported:
        assert "policies" not in m and "predictor" not in m, m
        assert "phase4" not in m, m


# -- the frozen interface -------------------------------------------------------

def test_execute_returns_the_four_required_things(p4):
    a = ActionExecutor(p4)
    s = p4.reset(0)
    # DEEP at t=0 needs cap(DEEP) + 3*cap(CHEAP) reserved; the episode budget is
    # deliberately below that, which is the scarcity the experiment runs at.
    r = a.execute(DEEP, s, p4.cap(DEEP) + 3 * p4.cap(CHEAP))
    assert set(r.consumed) == {"tokens"} and r.consumed["tokens"] > 0
    assert 0.0 <= r.utility <= 1.0
    assert r.observation["t"] == 1
    assert r.state is not s


def test_unknown_action_raises(p4):
    with pytest.raises(UnknownAction):
        ActionExecutor(p4).execute("THINK_HARDER", p4.reset(0), p4.budget)


def test_unaffordable_action_is_refused_not_executed(p4):
    """Refused BEFORE the resource is spent. Checking afterwards is not a
    budget, it is a receipt."""
    a = ActionExecutor(p4)
    s = p4.reset(0)
    r = a.execute(DEEP, s, 10.0)
    assert not r.ok and r.consumed["tokens"] == 0.0
    assert r.state is s, "a refused action must not advance the state"


def test_consumed_is_measured_not_nominal(p4):
    a = ActionExecutor(p4)
    r = a.execute(DEEP, p4.reset(0), p4.cap(DEEP) + 3 * p4.cap(CHEAP))
    assert r.consumed["tokens"] < p4.cap(DEEP)
    assert r.consumed["tokens"] == pytest.approx(60 + int(HIGH * 0.29))


# -- equivalence with the frozen executor ---------------------------------------

def _policies(env):
    return {"cheap": all_cheap(env), "greedy": greedy(env),
            "sched01": fixed_schedule(env, {0, 1}),
            "sched23": fixed_schedule(env, {2, 3})}


def test_ares_reproduces_run_episode_exactly_on_phase4(p4):
    loop = ExecutorLoop(p4)
    for name, pol in _policies(p4).items():
        for ep in range(len(p4.episodes)):
            a = loop.run(pol, ep, p4.budget)
            b = run_episode(p4, pol, ep, p4.budget)
            assert a.actions == b.modes, (name, ep)
            assert a.costs == pytest.approx(b.costs), (name, ep)
            assert a.spent == pytest.approx(b.spent), (name, ep)
            assert a.utility == pytest.approx(b.utility), (name, ep)


def test_ares_reproduces_run_episode_exactly_on_env6():
    """Env 6 too: the equivalence must not be a property of one environment."""
    env = Env6(seed=20260817, n=40)
    loop = ExecutorLoop(env)
    pols = {f"s{s}": (lambda o, b, s=s: "M2" if (o["t"] in s and b >= 1.0) else "H")
            for k in range(3) for s in itertools.combinations(range(4), k)}
    pols["cue"] = lambda o, b: "M2" if (o["cue"] == 1 and b >= 1.0) else "H"
    for name, pol in pols.items():
        for ep in range(40):
            a = loop.run(pol, ep, 2.0)
            b = run_episode(env, pol, ep, 2.0)
            assert a.actions == b.modes and a.utility == pytest.approx(b.utility), name


def test_env6_reference_utilities_survive_the_ares_path():
    """The frozen numbers must be reachable through the new interface too, or
    the interface is a fork of the executor rather than a face on it."""
    env = Env6(seed=20260817, n=800)
    loop = ExecutorLoop(env)
    cheap = lambda o, b: "H"                                    # noqa: E731
    cue = lambda o, b: "M2" if (o["cue"] == 1 and b >= 1.0) else "H"  # noqa: E731
    u_h = float(np.mean([loop.run(cheap, e, 2.0).utility for e in range(800)]))
    u_c = float(np.mean([loop.run(cue, e, 2.0).utility for e in range(800)]))
    assert u_h == pytest.approx(0.6896875, abs=1e-12)
    assert u_c == pytest.approx(0.813125, abs=1e-12)


def test_loop_raises_rather_than_silently_skipping_an_action(p4):
    """A policy that asks for something unaffordable must fail loudly. Silently
    substituting the cheap mode would let a policy look budget-compliant while
    the executor was quietly rewriting its decisions."""
    always_deep = lambda o, b: DEEP                             # noqa: E731
    with pytest.raises(BudgetExceeded):
        ExecutorLoop(p4).run(always_deep, 0, 2000.0)


def test_budget_is_never_exceeded_across_every_policy_and_episode(p4):
    loop = ExecutorLoop(p4)
    for pol in _policies(p4).values():
        for ep in range(len(p4.episodes)):
            e = loop.run(pol, ep, p4.budget)
            assert e.spent <= p4.budget + 1e-9
            assert len(e.actions) == p4.n_decisions
            assert all(x in (CHEAP, DEEP) for x in e.actions)
