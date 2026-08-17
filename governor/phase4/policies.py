"""The seven Phase 4 policies. All are `fn(obs, budget_left) -> mode` and all
are scored by the canonical executor and nothing else.

Every policy computes affordability the same way, through `env.feasible`, so no
policy gets a budget advantage from being sloppier about reserves than another.
"""
from __future__ import annotations

import itertools
from typing import Callable

import numpy as np

from governor.phase4.env import CHEAP, DEEP, P4Env
from governor.phase4.predictor import OpportunityCostDP, ValuePredictor

Policy = Callable[[dict, float], str]


def affordable_upgrades(env: P4Env, budget_left: float, items_left: int) -> int:
    """How many of the remaining items could still get the deep budget.

    Recomputed every step from the ACTUAL remaining budget, so under-spent calls
    return their slack to the pool. That is the difference between planning
    against nominal costs and against measured ones.
    """
    if items_left <= 0:
        return 0
    slack = budget_left - items_left * env.cap(CHEAP)
    step = env.cap(DEEP) - env.cap(CHEAP)
    return int(max(0, min(items_left, np.floor(slack / step + 1e-9))))


# -- baselines -----------------------------------------------------------------

def all_cheap(env: P4Env) -> Policy:
    """H: never spend. The floor."""
    return lambda o, b: CHEAP


def greedy(env: P4Env) -> Policy:
    """Spend at the earliest affordable slots. The honest form of 'always deep'
    when always-deep does not fit in the budget."""
    def pol(o, b):
        return DEEP if env.feasible(DEEP, b, o["items_left"]) else CHEAP
    return pol


def fixed_schedule(env: P4Env, slots: set[int]) -> Policy:
    """Deep at fixed positions. The best such schedule is chosen on calibration."""
    def pol(o, b):
        if o["t"] in slots and env.feasible(DEEP, b, o["items_left"]):
            return DEEP
        return CHEAP
    return pol


def text_heuristic(env: P4Env, feature: str, threshold: float) -> Policy:
    """A single observable text rule. This is the baseline that matters: if a
    one-feature threshold does as well as the Governor, the learned value
    predictor and the dynamic program are decoration."""
    def pol(o, b):
        if (o["features"].get(feature, 0.0) >= threshold
                and env.feasible(DEEP, b, o["items_left"])):
            return DEEP
        return CHEAP
    return pol


def clairvoyant(env: P4Env, ep: int) -> Policy:
    """The exact optimum: the best feasible assignment of modes to the four
    items, found by enumerating all 2^4 of them and executing each.

    Two earlier versions of this were wrong in opposite directions.

    Env 6's first oracle maximised REALISED utility over patterns and inflated
    the headroom by 97%, because outcomes there were random GIVEN the mode, so
    the maximum was partly picking lucky coin flips. That objection does not
    apply here: a cached response is deterministic given (item, mode), so the
    maximum over assignments is the best an allocator could have done and
    nothing more.

    The version before this one ranked items by gain and took the top `k_max`
    computed from WORST-CASE costs. It was beaten by the best fixed schedule,
    because under-spend frees budget for a third upgrade and a ceiling that
    ignores that is not a ceiling. Enumeration has no such assumption: it uses
    the same executor and the same budget as every other policy.
    """
    best_u, best_s = -1.0, frozenset()
    for r in range(env.n_decisions + 1):
        for s in itertools.combinations(range(env.n_decisions), r):
            try:
                u = _run(env, fixed_schedule(env, set(s)), ep)
            except RuntimeError:                 # assignment does not fit
                continue
            if u > best_u:
                best_u, best_s = u, frozenset(s)
    return fixed_schedule(env, set(best_s))


def _run(env: P4Env, pol: Policy, ep: int) -> float:
    from governor.gate.executor import run_episode
    return run_episode(env, pol, ep, env.budget).utility


def myopic(env: P4Env, vp: ValuePredictor, thr: float = 0.0) -> Policy:
    """Spend whenever the predicted gain clears a FIXED threshold.

    The ablation that matters most. In Env 6 exactly this rule collapsed into
    greedy: predicted gain was positive for both cue values, so `q > 0` fired on
    every item and the budget simply truncated. If the DP does no better than
    this here, the dynamic program is decoration and should be deleted.
    """
    def pol(o, b):
        if not env.feasible(DEEP, b, o["items_left"]):
            return CHEAP
        return DEEP if vp.predict_one(o["features"]) > thr else CHEAP
    return pol


# -- the Governor ---------------------------------------------------------------

def governor(env: P4Env, vp: ValuePredictor, dp: OpportunityCostDP,
             trace: list | None = None) -> Policy:
    """Predicted gain vs the opportunity cost of the budget it would consume."""
    def pol(o, b):
        m = o["items_left"]
        k = affordable_upgrades(env, b, m)
        if k <= 0 or not env.feasible(DEEP, b, m):
            if trace is not None:
                trace.append({"t": o["t"], "k": k, "q": None, "thr": None,
                              "mode": CHEAP, "reason": "infeasible"})
            return CHEAP
        q = vp.predict_one(o["features"])
        thr = dp.threshold(m, k)
        mode = DEEP if q >= thr else CHEAP
        if trace is not None:
            trace.append({"t": o["t"], "k": k, "q": round(q, 5),
                          "thr": round(thr, 5), "mode": mode,
                          "reason": "q>=thr" if mode == DEEP else "reserve"})
        return mode
    return pol


def governor_state(env: P4Env, sp, dp: OpportunityCostDP, components,
                   ens=None) -> Policy:
    """The Governor over a composed cognitive state (Step 9 ablation).

    Identical decision rule; only the predictor's input changes. Keeping the
    rule fixed is what makes the ablation about the STATE rather than about two
    different controllers.
    """
    from governor.phase4.graft import state_features

    def pol(o, b):
        m = o["items_left"]
        k = affordable_upgrades(env, b, m)
        if k <= 0 or not env.feasible(DEEP, b, m):
            return CHEAP
        sd = ens.spread(o["features"]) if ens else 0.0
        q = sp.predict_one(state_features(env, o, b, components, sd))
        return DEEP if q >= dp.threshold(m, k) else CHEAP
    return pol
