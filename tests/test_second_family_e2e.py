"""Second task family, end to end through the UNCHANGED interfaces.

The generalization claim under test is architectural, not numerical: the same
Governor, executor, ActionExecutor and budget accounting must run a family with a
different difficulty cue, a different surface form, and a CONTINUOUS
partial-credit reward -- without a line of the controller changing.

A synthetic response cache is used so this runs with no API key and no quota.
The LLM measurement on this family is a separate, quota-blocked experiment; what
is asserted here is that nothing in the stack is specialised to family one.
"""
from __future__ import annotations

import numpy as np
import pytest

from governor.execution.executor import ExecutorLoop
from governor.gate.executor import run_episode
from governor.phase4.collect import CallRecord, ResponseCache
from governor.phase4.config import PROMPT_CAP
from governor.phase4.env import DEEP, P4Env, make_episodes
from governor.phase4.evaluate import constant, execute
from governor.phase4.policies import all_cheap, clairvoyant, governor, greedy
from governor.phase4.predictor import OpportunityCostDP, ValuePredictor
from governor.phase4 import puzzles as P
from governor.phase4.family import PUZZLES

LOW, HIGH = 300, 700
BUDGET = 6 * (PROMPT_CAP + LOW) + 1.5 * (HIGH - LOW)
N_ITEMS = 6


def _reply(item: P.Puzzle, fraction: float) -> str:
    """A reply that places exactly `fraction` of the seats correctly."""
    names = list(item.answer)
    k = int(round(fraction * len(names)))
    good = {n: item.answer[n] for n in names[:k]}
    bad = {n: (item.answer[n] % len(names)) + 1 for n in names[k:]}
    merged = {**good, **bad}
    return "<think>\nreasoning\n</think>\n" + ", ".join(
        f"{n}={merged[n]}" for n in names)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Deep helps on clue-heavy puzzles, partially on the rest."""
    tmp = tmp_path_factory.mktemp("p2")
    cache = ResponseCache(tmp / "puz.sqlite", model="synthetic",
                          system_prompt=P.SYSTEM_PROMPT_PUZZLE)
    pool = P.make_pool(seed=99, n=120)
    for it in pool:
        hard = P.features(it.prompt)["n_clues"] >= 5
        for mt in (LOW, HIGH):
            frac = (1.0 if mt == HIGH else (0.25 if hard else 0.75))
            used = int(mt * (0.92 if mt == LOW else 0.78))
            cache.put(it, mt, CallRecord(
                it.item_id, mt, _reply(it, frac), "stop", 60, used, used - 12,
                60 + used, 0.02), attempts=1)
    return cache, pool


def _env(cache, items, seed=5):
    """The ONLY thing that changes between families is this argument."""
    n = len(items) // N_ITEMS
    return P4Env(cache, make_episodes(items, n, seed, n_items=N_ITEMS),
                 LOW, HIGH, BUDGET, PROMPT_CAP, family=PUZZLES), list(range(n))


def test_only_the_family_argument_changes(world):
    """The generalization claim, stated as code: the same env class, the same
    Governor, the same executor -- one argument different."""
    from governor.phase4.family import ARITHMETIC, PUZZLES as PZ
    assert PZ.feature_names != ARITHMETIC.feature_names
    assert PZ.grade is not None and ARITHMETIC.grade is None
    cache, pool = world
    env, _ = _env(cache, pool)
    assert env.family is PZ
    assert set(env.observe(env.reset(0))["features"]) == set(PZ.feature_names)


def test_reward_is_continuous_not_binary(world):
    """The structural difference from family one: gains are fractions."""
    cache, pool = world
    env, E = _env(cache, pool)
    gains = np.array([env.realised_gain(i) for i in pool], float)
    assert set(np.unique(gains)) - {-1.0, 0.0, 1.0}, "gains are still binary"
    assert 0.0 < gains.mean() < 1.0
    us = [run_episode(env, all_cheap(env), e, env.budget).utility for e in E]
    assert any(0.0 < u < 1.0 for u in us), "utility never lands between 0 and 1"


def test_the_unchanged_governor_runs_this_family(world):
    cache, pool = world
    cal_items, test_items = pool[:60], pool[60:]
    cal_env, C = _env(cache, cal_items, seed=5)
    test_env, T = _env(cache, test_items, seed=6)

    gains = np.array([cal_env.realised_gain(i) for i in cal_items], float)
    # No adapter, no monkeypatch: the family carries its own feature extractor.
    vp = ValuePredictor(kind="gbt", family=PUZZLES)
    rep = vp.fit(cal_items, gains)
    dp = OpportunityCostDP(vp.q_samples, n_items=N_ITEMS, max_k=N_ITEMS)

    gov = execute(test_env, "GOVERNOR",
                  constant(governor(test_env, vp, dp)), T)
    grd = execute(test_env, "greedy", constant(greedy(test_env)), T)
    chp = execute(test_env, "cheap", constant(all_cheap(test_env)), T)
    orc = execute(test_env, "oracle", lambda e: clairvoyant(test_env, e), T)

    assert rep.cv_r2 > 0.0, f"predictor learned nothing: {rep}"
    for r in (gov, grd, chp, orc):
        assert r.spent.max() <= test_env.budget + 1e-9
        assert all(len(m) == N_ITEMS for m in r.modes)
    assert chp.mean < orc.mean, "deep budget buys nothing in this world"
    assert gov.mean <= orc.mean + 1e-9, "Governor beat the exact optimum"
    m = gov.metrics(test_env.budget)
    assert 0.0 < m["budget_utilization"] <= 1.0
    assert m["deep_calls_per_episode"] <= N_ITEMS


def test_ares_is_trace_identical_on_the_second_family(world):
    """The executor equivalence must not be a property of family one."""
    cache, pool = world
    env, E = _env(cache, pool)
    loop = ExecutorLoop(env)
    for pol in (all_cheap(env), greedy(env)):
        for e in E[:6]:
            a = loop.run(pol, e, env.budget)
            b = run_episode(env, pol, e, env.budget)
            assert a.actions == b.modes
            assert a.costs == pytest.approx(b.costs)
            assert a.utility == pytest.approx(b.utility)


def test_partial_credit_flows_through_the_budget_accounting(world):
    cache, pool = world
    env, E = _env(cache, pool)
    tr = run_episode(env, greedy(env), E[0], env.budget)
    assert tr.spent == pytest.approx(sum(tr.costs))
    assert tr.spent < sum(env.cap(m) for m in tr.modes), "charged the cap"
    assert 0.0 <= tr.utility <= 1.0


def test_heuristic_features_come_from_the_family(world):
    """The third generalization defect of this kind: calibrate() iterated a
    hardcoded arithmetic feature tuple and raised KeyError on puzzles."""
    from governor.phase4.family import ARITHMETIC, PUZZLES as PZ
    from governor.phase4.pipeline import calibrate
    assert set(PZ.heuristic_features).isdisjoint(ARITHMETIC.heuristic_features)
    cache, pool = world
    env, E = _env(cache, pool[:60])
    cal = calibrate(env, pool[:60], E)
    assert cal.heuristic_feature in PZ.heuristic_features
