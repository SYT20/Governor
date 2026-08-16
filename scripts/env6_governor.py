#!/usr/bin/env python3
"""Gates 4-5: held-out replication on a FRESH SEED, then the learned Governor.

Everything is frozen before this runs: environment, cost model, budget, episode
structure, oracle definition, constant baseline, EPSILON, metric.

The Governor sees ONLY observable state -- t, cue, remaining budget. It never
sees hidden difficulty, the oracle's choice, the outcome, or the episode index.
It is trained on CALIBRATION episodes and evaluated on a fresh-seed test set.

Baselines, all through the canonical executor:
  H            never deep
  constant     best fixed schedule, frozen on calibration
  greedy       spend the budget at the earliest decisions (budget-limited
               always-deep -- the honest version, since always-deep is
               infeasible at budget 2 of 4)
  cue          observable heuristic: deep iff cue says hard
  GOVERNOR     learned from observable state
  oracle       difficulty oracle, upper bound
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from governor.gate.env6 import Env6  # noqa: E402
from governor.gate.executor import run_episode  # noqa: E402

EPSILON, BUDGET = 0.02, 2.0
N_CAL, N_TEST = 800, 800
TRAIN_SEED, TEST_SEED = 0, 20260817        # fresh seed, never touched


def const_policy(slots):
    return lambda o, b: "M2" if (o["t"] in slots and b >= 1.0) else "H"


def greedy(o, b):
    return "M2" if b >= 1.0 else "H"


def cue(o, b):
    return "M2" if (o["cue"] == 1 and b >= 1.0) else "H"


def oracle_policy(env, ep):
    hard = env.hard[ep]
    g = [0.35 if h else 0.05 for h in hard]
    return const_policy(set(sorted(range(4), key=lambda i: (-g[i], i))[:2]))


def mean_u(env, pol, eps):
    return np.array([run_episode(env, pol, e, BUDGET).utility for e in eps])


def build_training(env, eps):
    """Teacher: per-decision realised gain of M2 over H, matched by CRN.

    Legitimate as a TARGET (it uses hidden difficulty offline); the FEATURES the
    Governor sees are observable only.
    """
    X, y = [], []
    for e in eps:
        s = env.reset(e)
        for t in range(4):
            gain = (env.ACC_gain(e, t) if hasattr(env, "ACC_gain") else None)
            from governor.gate.env6 import ACC
            h = int(env.hard[e][t]); r = env.roll[e][t]
            gain = int(r < ACC[("M2", h)]) - int(r < ACC[("H", h)])
            X.append([t, int(env.cue[e][t])]); y.append(gain)
    return np.array(X, float), np.array(y, float)


def make_governor(model):
    """Allocate by predicted gain RELATIVE TO REMAINING OPPORTUNITIES.

    The first version tested `pred > 0` per decision. That collapsed into
    greedy: M2's realised gain is positive on average for BOTH cue values
    (0.95 vs 0.90 even on easy items), so the threshold fired everywhere and the
    budget simply truncated at t=0,1 -- 2.00 deep calls/episode, identical to
    greedy on every episode, delta exactly +0.0000.

    Spending is a CHOICE AMONG the episode's four items, so the decision rule
    must compare this item against what is still to come. Reserve the budget
    unless this item's predicted gain beats the expected best of the remaining
    slots -- a one-step lookahead over the controller's own predictions, using
    only observable features.
    """
    from governor.gate.env6 import CUE_NOISE, P_HARD
    def q(t, c):
        return float(model.predict(np.array([[t, c]], float))[0])
    # expected gain of a FUTURE decision, marginalising the unseen cue
    p_cue_hard = P_HARD * (1 - CUE_NOISE) + (1 - P_HARD) * CUE_NOISE
    def pol(o, b):
        if b < 1.0:
            return "H"
        t, here = o["t"], q(o["t"], o["cue"])
        remaining = 4 - t - 1
        if remaining <= 0:
            return "M2" if here > 0 else "H"
        fut = p_cue_hard * q(t + 1, 1) + (1 - p_cue_hard) * q(t + 1, 0)
        # spend only if this slot is at least as good as a typical future slot,
        # OR if there are no more slots than remaining budget (use it or lose it)
        must_spend = remaining <= b - 1e-9
        return "M2" if (here >= fut or must_spend) and here > 0 else "H"
    return pol


def main() -> int:
    print("=" * 76)
    print("ENV 6 — GATE 4 (fresh seed) + GATE 5 (learned Governor)")
    print("=" * 76)
    tr_env = Env6(seed=TRAIN_SEED, n=N_CAL)
    te_env = Env6(seed=TEST_SEED, n=N_TEST)
    cal, test = range(N_CAL), range(N_TEST)

    scheds = [s for k in range(3) for s in itertools.combinations(range(4), k)]
    best_s = max(scheds, key=lambda s: mean_u(tr_env, const_policy(set(s)), cal).mean())
    print(f"\nGATE 1  frozen best constant (calibration): {sorted(best_s)}")

    X, y = build_training(tr_env, cal)
    model = HistGradientBoostingRegressor(max_depth=3, max_iter=150,
                                          random_state=0).fit(X, y)
    gov = make_governor(model)
    print(f"GATE 5  Governor trained on {len(X)} observable decisions "
          f"(features: t, cue)")

    U = {
        "H": mean_u(te_env, const_policy(set()), test),
        "constant": mean_u(te_env, const_policy(set(best_s)), test),
        "greedy": mean_u(te_env, greedy, test),
        "cue": mean_u(te_env, cue, test),
        "GOVERNOR": mean_u(te_env, gov, test),
        "oracle": np.array([run_episode(te_env, oracle_policy(te_env, e), e,
                                        BUDGET).utility for e in test]),
    }
    print(f"\nGATE 4  HELD-OUT, seed {TEST_SEED}, {N_TEST} episodes")
    for k in ("H", "constant", "greedy", "cue", "GOVERNOR", "oracle"):
        v = U[k]
        print(f"    {k:<10} {v.mean():.4f} +- {1.96*v.std(ddof=1)/np.sqrt(len(v)):.4f}")

    print("\n  PRIMARY COMPARISONS (paired, same episodes)")
    res = {}
    for base in ("constant", "greedy", "cue"):
        d = U["GOVERNOR"] - U[base]
        se = d.std(ddof=1) / np.sqrt(len(d))
        ok = d.mean() - 1.96 * se > 0
        res[base] = [float(d.mean()), float(d.mean()-1.96*se), float(d.mean()+1.96*se)]
        print(f"    GOVERNOR - {base:<9} {d.mean():+.4f} "
              f"[{d.mean()-1.96*se:+.4f}, {d.mean()+1.96*se:+.4f}]"
              f"{'  BEATS' if ok else '  not separable'}")
    head = (U["GOVERNOR"].mean() - U["constant"].mean()) / \
           max(U["oracle"].mean() - U["constant"].mean(), 1e-9)
    print(f"\n    oracle headroom captured: {head:.0%}")

    # compute accounting
    n_deep = np.mean([sum(1 for m in run_episode(te_env, gov, e, BUDGET).modes
                          if m == "M2") for e in test])
    n_deep_g = np.mean([sum(1 for m in run_episode(te_env, greedy, e, BUDGET).modes
                            if m == "M2") for e in test])
    print(f"    deep calls/episode: GOVERNOR {n_deep:.2f}  greedy {n_deep_g:.2f}"
          f"  (budget 2)")

    passed = (res["constant"][1] > 0 and res["greedy"][1] > 0)
    print(f"\n  GATE 5: {'PASS -- Governor beats constant AND budget-limited greedy'
                        if passed else 'FAIL'}")
    Path("results").mkdir(exist_ok=True)
    Path("results/env6_governor.json").write_text(json.dumps(
        {"means": {k: float(v.mean()) for k, v in U.items()},
         "deltas": res, "headroom_captured": float(head),
         "deep_calls": {"governor": float(n_deep), "greedy": float(n_deep_g)},
         "pass": bool(passed)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
