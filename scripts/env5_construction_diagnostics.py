#!/usr/bin/env python3
"""Three construction diagnostics for Environment 5, before H1-H5.

Required by review after the mode-construction sub-gate passed. The sub-gate
established that C(M0)=0 < C(M1) < C(M2) with a shared primitive and that M1's
output varies. None of that shows M1 is USEFUL:

> A mode does not earn its place because it is computationally expensive or
> because its output varies. It earns its place only if that computation
> produces information that improves the resource-allocation decision.

DIAG 1  does M1's assessment predict Delta = U(M2) - U(M0)?
DIAG 2  is the myopic-optimal action inside M1's candidate pool?
DIAG 3  does M0's cached prior dispersion leak instance or regime information?

Recorded outcome, 2026-08-16:
  DIAG 3  PASS
  DIAG 2  FAIL, 81.3% recall -- the pool is first-16-BY-INDEX, not by gain
  DIAG 1  INVALIDATED -- m2_plan does not actually look ahead (see below), so
          the quantity M1 is asked to predict is not the quantity intended
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.stats import spearmanr  # noqa: E402

from governor.envs.env5_modes import (  # noqa: E402
    InstrumentedBayes,
    ModeRunner,
    m0_reflex,
    m2_plan,
)
from governor.envs.probe_family import (  # noqa: E402
    ObservableProbeBayes,
    ProbeTask,
    make_config,
)

REGIMES = (0.10, 0.35, 0.60, 1.50)


def advance(ib, x, steps, budget=6.0):
    logL, avail = ib.prior_logL(), list(range(ib.n_groups))
    for _ in range(steps):
        g = ib.b.myopic_step(logL, avail, budget)
        avail.remove(g)
        logL = logL + ib.b.loglik_cols(x, ib.group_cols[g])
    return logL, avail


def rollout(ib, x, logL, avail, budget, first):
    logL, avail, spent = logL.copy(), list(avail), 0.0
    if first is not None:
        avail.remove(first)
        spent += float(ib.cost[first])
        logL = logL + ib.b.loglik_cols(x, ib.group_cols[first])
    while True:
        g = ib.b.myopic_step(logL, avail, budget - spent)
        if g is None:
            break
        avail.remove(g)
        spent += float(ib.cost[g])
        logL = logL + ib.b.loglik_cols(x, ib.group_cols[g])
    return int(np.argmax(ib.b.label_posterior(logL)))


def diag3():
    sp = {}
    for so in REGIMES:
        t = ProbeTask(cfg=make_config(so, 1.0, 0.05), n_samples=4, seed=1)
        sp[so] = ObservableProbeBayes(t)._prior_spread.copy()
    return all(np.allclose(sp[k], sp[REGIMES[0]]) for k in sp)


def diag2(n=25):
    rec = []
    for so in (0.10, 0.60, 1.50):
        t = ProbeTask(cfg=make_config(so, 1.0, 0.05), n_samples=n, seed=4)
        ib = InstrumentedBayes(ObservableProbeBayes(t))
        for i in range(n):
            x = t.features[i]
            logL, avail = advance(ib, x, 2)
            afford = [g for g in avail if g != ib.probe_group]
            full = ib.b.gains(logL, afford)
            best = max(afford, key=lambda a: full[a] / ib.cost[a])
            base = ib.b.gains(logL, afford[:16])
            pool = sorted(afford[:16], key=lambda a: -base[a] / ib.cost[a])[:8]
            rec.append(best in pool)
    return float(np.mean(rec)), len(rec)


def diag1(n=60):
    A, D = [], []
    for so in REGIMES:
        t = ProbeTask(cfg=make_config(so, 1.0, 0.05), n_samples=n, seed=7)
        ib = InstrumentedBayes(ObservableProbeBayes(t))
        for i in range(n):
            x, y = t.features[i], int(t.labels[i])
            logL, avail = advance(ib, x, 2)
            r = ModeRunner(ib=ib, b_tool=6.0, b_compute=1e9)
            r.tool_spent = 2.0
            r.invoke("M1", logL, avail, x, "candidate_evals")
            a1 = r.trace[-1]["assessment"]
            a0, a2 = m0_reflex(ib, logL, avail, 4.0), m2_plan(ib, logL, avail, 4.0)
            if a1 is None or a0 is None or a2 is None:
                continue
            A.append(a1)
            D.append(int(rollout(ib, x, logL, avail, 4.0, a2) == y)
                     - int(rollout(ib, x, logL, avail, 4.0, a0) == y))
    return np.array(A), np.array(D)


def m2_is_really_planning(n=30):
    t = ProbeTask(cfg=make_config(0.60, 1.0, 0.05), n_samples=n, seed=4)
    ib = InstrumentedBayes(ObservableProbeBayes(t))
    same = 0
    for i in range(n):
        x = t.features[i]
        logL, avail = advance(ib, x, 2)
        same += int(ib.b.myopic_step(logL, avail, 4.0)
                    == m2_plan(ib, logL, avail, 4.0))
    return same, n


def main() -> int:
    print("=" * 78)
    print("ENVIRONMENT 5 — CONSTRUCTION DIAGNOSTICS (before H1-H5)")
    print("=" * 78)

    ok3 = diag3()
    print(f"\nDIAG 3  M0 preprocessing audit")
    print(f"  _prior_spread identical across all true regimes: {ok3}")
    print(f"  built from MU only; no instance observation enters")
    print(f"  -> model-initialisation cost, declared outside the episode budget")
    print(f"  {'PASS' if ok3 else 'FAIL'}")

    r, n = diag2()
    print(f"\nDIAG 2  M1 candidate-pool recall")
    print(f"  P(myopic-optimal action in pool) = {r:.1%}  (n={n})")
    print(f"  {'PASS' if r > 0.9 else 'FAIL -- pool is first-16-BY-INDEX, not by gain'}")

    A, D = diag1()
    rho = float(spearmanr(A, D).statistic)
    lo, hi = A <= np.median(A), A > np.median(A)
    diff = float(D[lo].mean() - D[hi].mean())
    se = float(np.sqrt(D[lo].var(ddof=1) / lo.sum() + D[hi].var(ddof=1) / hi.sum()))
    print(f"\nDIAG 1  does M1 predict the value of escalating M0 -> M2?")
    print(f"  n={len(A)}  Spearman(M1, Delta) = {rho:+.3f}")
    print(f"  E[Delta | M1 low ] = {D[lo].mean():+.3f}")
    print(f"  E[Delta | M1 high] = {D[hi].mean():+.3f}")
    print(f"  low - high = {diff:+.3f} +- {1.96*se:.3f}")

    same, tot = m2_is_really_planning()
    print(f"\n  BEFORE reading DIAG 1: is M2 actually planning?")
    print(f"  M2 picks the same action as plain myopic in {same}/{tot} "
          f"({same/tot:.0%}) of states")
    if same / tot > 0.7:
        print("  -> DIAG 1 IS INVALIDATED. m2_plan's continuation term is")
        print("     gains(logL, nxt) at the CURRENT logL: the posterior is never")
        print("     updated for having taken a, so the term is the same for every")
        print("     candidate bar exclusion. It is myopic plus a near-constant")
        print("     bonus, not a 2-step lookahead. M1 cannot be judged on its")
        print("     ability to predict the value of a planner that does not plan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
