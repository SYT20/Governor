"""The whole Phase 4 pipeline, against a synthetic cache.

A pipeline that only runs when the network does gets validated by the same run
that produces its result — which is exactly how Run I shipped a scoring bug as a
finding. These tests exercise calibrate -> freeze -> evaluate -> trap-check on
responses whose structure is known, so a wrong answer here is a code defect and
not a model result.

The synthetic engine is built so the RIGHT answer is known in advance:
items whose text contains many numerals genuinely need the deep budget, and
nothing else does. A correct Governor must beat every fixed schedule; a broken
one cannot.
"""
from __future__ import annotations

import numpy as np
import pytest

from governor.phase4.collect import CallRecord, ResponseCache
from governor.phase4.config import PROMPT_CAP, episode_budget
from governor.phase4.env import DEEP, P4Env, make_episodes
from governor.phase4.evaluate import paired_ci
from governor.phase4.pipeline import calibrate, evaluate_heldout, summarise
from governor.phase4.tasks import features, make_pool

LOW, HIGH = 300, 1400
BUDGET = episode_budget(LOW, HIGH)


def build_cache(path, pool, rng, signal=True, noise=0.10):
    """Deep helps exactly on numeral-heavy items, plus a little label noise.

    `signal=False` builds the null world where the deep budget helps uniformly
    at random -- there the Governor MUST NOT beat a fixed schedule, and a test
    asserts it does not.
    """
    cache = ResponseCache(path, model="synthetic")
    for it in pool:
        f = features(it.prompt)
        needs = (f["numerals"] >= 3) if signal else (rng.random() < 0.5)
        if rng.random() < noise:
            needs = not needs
        for mt in (LOW, HIGH):
            correct = (not needs) or mt == HIGH
            used = int(mt * (0.95 if mt == LOW else 0.55))
            cache.put(it, mt, CallRecord(
                it.item_id, mt, str(it.answer) if correct else "0",
                "stop" if correct else "length", 60, used, used - 10,
                60 + used, 0.4), attempts=1)
    return cache


@pytest.fixture
def world(tmp_path):
    rng = np.random.default_rng(0)
    cal_pool = make_pool(1000, 240)
    test_pool = make_pool(20260817, 240)
    cache = build_cache(tmp_path / "s.sqlite", cal_pool + test_pool, rng)
    cal_env = P4Env(cache, make_episodes(cal_pool, 60, 11), LOW, HIGH, BUDGET,
                    PROMPT_CAP)
    test_env = P4Env(cache, make_episodes(test_pool, 60, 22), LOW, HIGH, BUDGET,
                     PROMPT_CAP)
    return cal_pool, cal_env, test_env


def test_pipeline_recovers_a_signal_it_is_supposed_to_recover(world):
    cal_pool, cal_env, test_env = world
    cal = calibrate(cal_env, cal_pool, range(60))
    assert cal.report.cv_r2 > 0.2, f"predictor learned nothing: {cal.report}"
    R, trace = evaluate_heldout(test_env, cal, range(60))
    s = summarise(R, cal, test_env, trace, commit="deadbeef",
                  froze_commit="cafebabe")
    d = s["deltas"][cal.base]
    assert d["mean"] > 0, f"Governor lost in a world built for it: {d}"
    assert R["GOVERNOR"].mean <= R["oracle"].mean + 1e-9


def test_governor_does_not_beat_a_schedule_when_there_is_no_signal(tmp_path):
    """The negative control. Nine environments in this project produced an
    apparent effect from an invariant; a pipeline that wins here is broken."""
    rng = np.random.default_rng(1)
    cal_pool, test_pool = make_pool(1000, 240), make_pool(20260817, 240)
    cache = build_cache(tmp_path / "n.sqlite", cal_pool + test_pool, rng,
                        signal=False)
    cal_env = P4Env(cache, make_episodes(cal_pool, 60, 11), LOW, HIGH, BUDGET,
                    PROMPT_CAP)
    test_env = P4Env(cache, make_episodes(test_pool, 60, 22), LOW, HIGH, BUDGET,
                     PROMPT_CAP)
    cal = calibrate(cal_env, cal_pool, range(60))
    R, trace = evaluate_heldout(test_env, cal, range(60))
    s = summarise(R, cal, test_env, trace, commit="a", froze_commit="b")
    assert not s["deltas"][cal.base]["beats"], (
        "Governor 'beat' the baseline on pure noise: " + str(s["deltas"]))


def test_every_policy_gets_the_same_budget_and_none_overspends(world):
    _, _, test_env = world
    cal_pool, cal_env, _ = world
    cal = calibrate(cal_env, cal_pool, range(60))
    R, _ = evaluate_heldout(test_env, cal, range(60))
    for k, r in R.items():
        assert r.spent.max() <= test_env.budget + 1e-9, k
        assert all(len(m) == 4 for m in r.modes), k


def test_charged_tokens_are_measured_not_nominal(world):
    cal_pool, cal_env, test_env = world
    cal = calibrate(cal_env, cal_pool, range(60))
    R, _ = evaluate_heldout(test_env, cal, range(60))
    m = R["greedy"].metrics(test_env.budget)
    nominal = 4 * test_env.cap("H")
    assert m["total_tokens_per_episode"] != pytest.approx(nominal)
    assert m["budget_utilization"] < 1.0


def test_traps_are_green_in_the_healthy_world(world):
    cal_pool, cal_env, test_env = world
    cal = calibrate(cal_env, cal_pool, range(60))
    R, trace = evaluate_heldout(test_env, cal, range(60))
    s = summarise(R, cal, test_env, trace, commit="run", froze_commit="prereg")
    assert s["red"] == [], {k: v for k, v in s["traps"].items() if not v[0]}


def test_trap_goes_red_when_the_prereg_commit_equals_the_run_commit(world):
    """Freezing evidence must be a real comparison, not a self-comparison."""
    cal_pool, cal_env, test_env = world
    cal = calibrate(cal_env, cal_pool, range(60))
    R, trace = evaluate_heldout(test_env, cal, range(60))
    s = summarise(R, cal, test_env, trace, commit="same", froze_commit="same")
    assert "frozen_before_heldout" in s["red"]


def test_oracle_bounds_every_implementable_policy(world):
    cal_pool, cal_env, test_env = world
    cal = calibrate(cal_env, cal_pool, range(60))
    R, _ = evaluate_heldout(test_env, cal, range(60))
    for k, r in R.items():
        if k == "oracle":
            continue
        assert r.mean <= R["oracle"].mean + 1e-9, k


def test_governor_spends_no_more_than_greedy(world):
    """If the Governor wins by spending more, the comparison is not at equal
    budget and the result means nothing."""
    cal_pool, cal_env, test_env = world
    cal = calibrate(cal_env, cal_pool, range(60))
    R, _ = evaluate_heldout(test_env, cal, range(60))
    g = R["GOVERNOR"].metrics(test_env.budget)
    gr = R["greedy"].metrics(test_env.budget)
    assert g["deep_calls_per_episode"] <= gr["deep_calls_per_episode"] + 1e-9


def test_paired_ci_is_paired():
    a = np.array([1.0, 0.0, 1.0, 0.0])
    assert paired_ci(a, a)["mean"] == 0.0
    assert paired_ci(a + 0.5, a)["mean"] == pytest.approx(0.5)
