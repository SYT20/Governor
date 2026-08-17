"""Reference lock. These pin Environment 6's validated numbers so later work
cannot silently move them.

If one of these fails, the reference changed. Either the change is intended --
in which case update the constants IN THE SAME COMMIT and say why -- or shared
code drifted and every downstream claim is suspect.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from governor.gate import env6 as E
from governor.gate.executor import run_episode
from governor.gate.m2_interface import M2Result, MathM2

BUDGET = 2.0
# Exact values, not the 4-dp figures printed in the run log. The first version
# of this file pinned 0.6897/0.8131/0.8387 with abs=1e-9 and failed immediately:
# a rounded report is not a reference. The lock must hold the real numbers.
TOL = 1e-12


def const_policy(sl):
    return lambda o, b: "M2" if (o["t"] in sl and b >= 1.0) else "H"


def cue(o, b):
    return "M2" if (o["cue"] == 1 and b >= 1.0) else "H"


@pytest.fixture(scope="module")
def env():
    assert E.CUE_NOISE == 0.15 and E.P_HARD == 0.5, "generative params drifted"
    return E.Env6(seed=20260817, n=800)


def _mean(env, pol):
    return float(np.mean([run_episode(env, pol, e, BUDGET).utility
                          for e in range(800)]))


def test_reference_baselines_unchanged(env):
    """Frozen held-out values from the Gate 5 run (seed 20260817, n=800)."""
    assert _mean(env, const_policy(set())) == pytest.approx(0.6896875, abs=TOL)
    assert _mean(env, cue) == pytest.approx(0.813125, abs=TOL)


def test_reference_oracle_unchanged(env):
    def oracle(ep):
        g = [0.35 if h else 0.05 for h in env.hard[ep]]
        return const_policy(set(sorted(range(4), key=lambda i: (-g[i], i))[:2]))
    u = float(np.mean([run_episode(env, oracle(e), e, BUDGET).utility
                       for e in range(800)]))
    assert u == pytest.approx(0.83875, abs=TOL)


def test_env6_structure_unchanged(env):
    assert env.n_decisions == 4
    assert set(env.modes()) == {"H", "M2"}
    assert E.ACC == {("H", 0): 0.90, ("H", 1): 0.50,
                     ("M2", 0): 0.95, ("M2", 1): 0.85}
    assert E.COST == {"H": 0.0, "M2": 1.0}


def test_difficulty_position_is_uninformative(env):
    """The property that makes constant schedules useless. If this breaks, the
    environment has silently become Env 5."""
    rate = env.hard.mean(axis=0)
    assert rate.max() - rate.min() < 0.06, f"position carries signal: {rate}"


def test_m2_interface_contract():
    m = MathM2()
    r = m({}, 1.0)
    assert isinstance(r, M2Result) and r.ok and r.cost_units == 1.0
    r0 = m({}, 0.5)
    assert not r0.ok and r0.cost_units == 0.0, "under-budget call must not charge"
