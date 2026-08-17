"""Phase 4 environment, budget accounting and allocation rule.

Uses a synthetic cache so the real code path (sqlite lookup, charge, executor)
is exercised without a network call. The thing being tested is the accounting,
not the model.
"""
from __future__ import annotations

import numpy as np
import pytest

from governor.gate.executor import run_episode
from governor.phase4.collect import CallRecord, ResponseCache
from governor.phase4.env import CHEAP, DEEP, P4Env, make_episodes
from governor.phase4.policies import (
    affordable_upgrades, all_cheap, clairvoyant, fixed_schedule, governor,
    greedy,
)
from governor.phase4.predictor import OpportunityCostDP, ValuePredictor
from governor.phase4.tasks import make_pool

LOW, HIGH, PROMPT_CAP = 300, 1400, 128
BUDGET = 4 * (PROMPT_CAP + LOW) + 2 * (HIGH - LOW)     # exactly two upgrades


@pytest.fixture
def env(tmp_path):
    """8 items: the odd ones need the deep budget, the even ones do not.

    Costs are deliberately well below the caps, so a policy that reserved
    nominal cost and one that reserves worst case behave differently -- which is
    what the budget-utilisation metric is about.
    """
    cache = ResponseCache(tmp_path / "c.sqlite", model="synthetic")
    pool = make_pool(seed=7, n=8)
    for j, it in enumerate(pool):
        needs_deep = j % 2 == 1
        for mt in (LOW, HIGH):
            correct = (not needs_deep) or mt == HIGH
            used = int(0.6 * mt)
            cache.put(it, mt, CallRecord(
                item_id=it.item_id, max_tokens=mt,
                content=str(it.answer) if correct else "0",
                finish_reason="stop" if correct else "length",
                prompt_tokens=60, completion_tokens=used,
                reasoning_tokens=used - 5, total_tokens=60 + used,
                latency_s=0.5), attempts=1)
    eps = make_episodes(pool, n_episodes=2, seed=1)
    return P4Env(cache, eps, low=LOW, high=HIGH, budget=BUDGET,
                 prompt_cap=PROMPT_CAP)


def test_charge_is_measured_not_nominal(env):
    """The cost must be the provider's reported total_tokens, never the cap."""
    tr = run_episode(env, greedy(env), 0, env.budget)
    assert tr.spent == pytest.approx(sum(tr.costs))
    assert tr.spent < sum(env.cap(m) for m in tr.modes), "charged the cap"
    for m, c in zip(tr.modes, tr.costs):
        assert c == pytest.approx(60 + 0.6 * env.tokens[m])


def test_budget_is_hard_for_every_policy(env):
    for name, pol in (("cheap", all_cheap(env)), ("greedy", greedy(env)),
                      ("sched", fixed_schedule(env, {0, 1, 2, 3}))):
        for ep in range(len(env.episodes)):
            tr = run_episode(env, pol, ep, env.budget)
            assert tr.spent <= env.budget + 1e-9, name


def test_every_item_is_always_answered(env):
    """Feasibility reserves a cheap call for each remaining item, so no policy
    can spend itself into skipping an item."""
    for pol in (greedy(env), fixed_schedule(env, {0, 1, 2, 3})):
        for ep in range(len(env.episodes)):
            tr = run_episode(env, pol, ep, env.budget)
            assert len(tr.modes) == 4


def test_budget_rule_guarantees_two_upgrades_in_the_worst_case(env, tmp_path):
    """With every call charging its cap, the frozen budget rule yields exactly
    the two upgrades it was designed for."""
    cache = ResponseCache(tmp_path / "worst.sqlite", model="synthetic")
    pool = make_pool(seed=7, n=4)
    for it in pool:
        for mt in (LOW, HIGH):
            cache.put(it, mt, CallRecord(it.item_id, mt, "1", "length",
                                         PROMPT_CAP, mt, mt, PROMPT_CAP + mt,
                                         0.1), attempts=1)
    worst = P4Env(cache, [pool], low=LOW, high=HIGH, budget=BUDGET,
                  prompt_cap=PROMPT_CAP)
    tr = run_episode(worst, greedy(worst), 0, worst.budget)
    assert sum(m == DEEP for m in tr.modes) == 2


def test_underspend_returns_slack_and_buys_extra_upgrades(env):
    """MEASURED accounting has a consequence the nominal version hides: calls
    that finish early hand budget back, so a policy affords MORE upgrades than
    the worst-case rule reserved for. Two is the floor, not the count.

    This is why the primary run reports realised deep calls per episode and why
    the robustness sweep varies the total budget: scarcity is an outcome of the
    engine's token use, not a number I set.
    """
    tr = run_episode(env, greedy(env), 0, env.budget)
    n_deep = sum(m == DEEP for m in tr.modes)
    assert n_deep == 3, n_deep
    assert tr.spent <= env.budget + 1e-9


def test_observation_carries_no_hidden_axis(env):
    s = env.reset(0)
    o = env.observe(s)
    assert set(o) == {"t", "prompt", "features", "items_left", "history"}
    for k in ("n_ops", "scale", "framing", "answer"):
        assert k not in o and k not in o["features"]


def test_history_never_reveals_correctness(env):
    """The controller may know what its earlier calls COST and whether they
    finished. It may not know whether they were right -- nothing reveals that at
    run time, and a state that carries it is an oracle feed."""
    s = env.reset(0)
    for _ in range(3):
        s, _ = env.step(s, DEEP if s.t == 0 else CHEAP)
    hist = env.observe(s)["history"]
    assert len(hist) == 3
    for h in hist:
        assert set(h) == {"mode", "total_tokens", "finish_reason", "answered",
                          "starved"}
        assert "correct" not in h and "parsed" not in h


def test_step_rejects_a_charge_above_the_cap(env, tmp_path):
    cache = ResponseCache(tmp_path / "bad.sqlite", model="synthetic")
    pool = make_pool(seed=8, n=4)
    for it in pool:
        for mt in (LOW, HIGH):
            cache.put(it, mt, CallRecord(it.item_id, mt, "1", "stop", 60,
                                         mt, mt, 10 ** 6, 0.1), attempts=1)
    bad = P4Env(cache, [pool], low=LOW, high=HIGH, budget=BUDGET,
                prompt_cap=PROMPT_CAP)
    with pytest.raises(RuntimeError, match="worst-case bound"):
        run_episode(bad, all_cheap(bad), 0, bad.budget)


def test_affordable_upgrades_returns_slack_as_it_is_freed(env):
    """Under-spent calls return budget to the pool; the count must reflect that."""
    assert affordable_upgrades(env, env.budget, 4) == 2
    assert affordable_upgrades(env, env.cap(CHEAP), 1) == 0
    assert affordable_upgrades(env, env.cap(DEEP), 1) == 1
    assert affordable_upgrades(env, 0.0, 3) == 0


# -- the dynamic program --------------------------------------------------------

def test_threshold_falls_as_items_run_out():
    """Opportunity cost drops when there is nothing left to save the budget for.
    A rule without this property is a fixed threshold wearing a schedule."""
    q = np.random.default_rng(0).normal(0.2, 0.3, 500)
    dp = OpportunityCostDP(q)
    assert dp.threshold(4, 1) > dp.threshold(3, 1) > dp.threshold(2, 1)
    assert dp.threshold(1, 1) == 0.0


def test_threshold_falls_as_budget_grows():
    q = np.random.default_rng(0).normal(0.2, 0.3, 500)
    dp = OpportunityCostDP(q)
    assert dp.threshold(4, 1) > dp.threshold(4, 2) > dp.threshold(4, 3)
    assert dp.threshold(4, 4) == 0.0


def test_no_budget_means_infinite_threshold():
    dp = OpportunityCostDP(np.zeros(10))
    assert dp.threshold(4, 0) == float("inf")


def test_governor_decisions_vary_with_state(env):
    """`constant_schedule` and `invariant_as_intelligence` in executable form:
    if the controller emits one pattern it is a schedule, not a controller."""
    pool = make_pool(seed=7, n=8)
    vp = ValuePredictor(kind="ridge")
    gains = np.array([env.realised_gain(it) for it in pool], float)
    vp.fit(pool, gains)
    dp = OpportunityCostDP(vp.q_samples)
    trace: list = []
    pats = set()
    for ep in range(len(env.episodes)):
        tr = run_episode(env, governor(env, vp, dp, trace), ep, env.budget)
        pats.add(tuple(tr.modes))
    assert any(r["thr"] is not None for r in trace)
    assert len(pats) >= 1 and any(m == DEEP for p in pats for m in p)


def test_clairvoyant_is_at_least_as_good_as_any_fixed_schedule(env):
    import itertools
    for ep in range(len(env.episodes)):
        u_or = run_episode(env, clairvoyant(env, ep), ep, env.budget).utility
        for k in range(5):
            for s in itertools.combinations(range(4), k):
                u = run_episode(env, fixed_schedule(env, set(s)), ep,
                                env.budget).utility
                assert u <= u_or + 1e-9, (ep, s, u, u_or)


def test_episodes_are_disjoint_and_pool_limited():
    pool = make_pool(seed=9, n=40)
    eps = make_episodes(pool, n_episodes=10, seed=3)
    seen = [i.item_id for e in eps for i in e]
    assert len(seen) == len(set(seen)) == 40
    with pytest.raises(ValueError):
        make_episodes(pool, n_episodes=11, seed=3)
