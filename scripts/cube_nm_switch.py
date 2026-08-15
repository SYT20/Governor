#!/usr/bin/env python3
"""Is the myopic/strategic switch STATE-dependent, or only budget-dependent?

THE ARCHITECTURAL CLAIM UNDER TEST. The proposed Governor is a meta-controller
that inspects the current state and asks "is this one of those states where
thinking ahead is worth paying for?", switching to strategic mode when

    Delta_NM(s, b) = V_nonmyopic(s, b) - V_myopic(s, b) > C_metareason.

That is only a learning problem if Delta_NM actually varies with s. If it varies
only with the remaining budget b, then the optimal Governor is a one-dimensional
lookup table over b, learnable from eight numbers, and calling it metacognition
would be a category error.

There is a structural reason to doubt state-dependence here, and it is worth
stating before measuring: AT STEP 0 EVERY INSTANCE IS IDENTICAL. Nothing has
been observed, so the posterior is the uniform prior for every row in the
dataset. A step-0 "should I go strategic?" decision therefore CANNOT condition
on anything except b -- not because the model is weak, but because there is no
state to condition on. Cell t=0 below is included precisely to make that visible
as a measurement rather than an assertion.

The real question is whether the deviation stays worth taking at t > 0, once
observations differentiate the instances, and whether WHICH instances benefit is
predictable from the state.

METHOD. Branched counterfactual, the protocol already used for SynthBug:

    roll the exact myopic policy t steps            -> state s_t
    arm A: continue myopic to budget B
    arm B: force the context acquisition at step t, then myopic to budget B
    delta = 1[B correct] - 1[A correct]

Then, WITHIN each (t, B) cell, try to predict delta > 0 from state features
alone, with grouped cross-validation. AUC at chance means the switch carries no
state signal in that cell and a budget lookup is the whole policy.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import GroupKFold, cross_val_predict  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from governor.envs.cube_nm_bayes import CubeNMBayes  # noqa: E402
from governor.envs.cube_nm_repro import N_LABELS, CubeNMRepro  # noqa: E402

SEEDS = [1, 2, 3, 4, 5]
N_TEST = 400
BUDGETS = [3, 4, 5]
FORKS = [0, 1, 2]
KMAX = max(BUDGETS)


def state_features(bayes, logL, t, acquired):
    """What a Governor could legitimately see at this state.

    Posterior-derived only: no latent, no label, no dataset-level identity. The
    group INDEX is deliberately excluded -- a feature like "group 0 not yet
    acquired" would let a model memorise the benchmark's fixed layout, which is
    the failure mode the next experiment is meant to remove anyway.
    """
    p = np.exp(logL - logL.max())
    p /= p.sum()
    py = p.reshape(bayes.K, N_LABELS).sum(axis=0)
    pc = p.reshape(bayes.K, N_LABELS).sum(axis=1)
    sy = np.sort(py)[::-1]
    sc = np.sort(pc)[::-1]
    return {
        "step": float(t),
        "H_y": float(-(py * np.log(np.maximum(py, 1e-300))).sum()),
        "H_c": float(-(pc * np.log(np.maximum(pc, 1e-300))).sum()),
        "max_py": float(sy[0]),
        "gap_py": float(sy[0] - sy[1]),
        "max_pc": float(sc[0]),
        "gap_pc": float(sc[0] - sc[1]),
        "n_blocks_touched": float(len({(g - 1) // 10 for g in acquired if g})),
    }


def rollout(bayes, x, logL, available, n_steps, forced_first=None):
    """Continue exact-myopic play from a state; return the final MAP label."""
    logL = logL.copy()
    available = list(available)
    for s in range(n_steps):
        g = forced_first if (s == 0 and forced_first is not None) \
            else bayes.myopic_step_exact(logL, available)
        available.remove(g)
        logL = logL + bayes.loglik_cols(x, bayes.group_cols[g])
    return int(np.argmax(bayes.label_posterior(logL)))


def main() -> int:
    print("=" * 88)
    print("IS THE MYOPIC/STRATEGIC SWITCH STATE-DEPENDENT, OR ONLY BUDGET-DEPENDENT?")
    print("=" * 88)

    rows = []
    for seed in SEEDS:
        ds = CubeNMRepro(n_samples=6000, seed=seed)
        bayes = CubeNMBayes(ds)
        rng = np.random.default_rng(seed)
        test = rng.choice(ds.n_samples, N_TEST, replace=False)

        for i in test:
            x = ds.features[i]
            y = int(ds.labels[i])
            logL = np.zeros(bayes.H)
            available = list(range(bayes.n_groups))
            acquired: list[int] = []

            for t in range(max(FORKS) + 1):
                if t in FORKS and 0 in available:
                    feats = state_features(bayes, logL, t, acquired)
                    # CONFOUND CONTROL. Myopic buys the context at step 2 in 64%
                    # of rows, so at t=1 the "strategic" action is very often the
                    # action myopic was going to take anyway -- the two arms are
                    # then literally identical and delta is 0 by construction.
                    # A classifier predicting delta>0 across all rows could score
                    # well merely by detecting whether the arms differ, which is
                    # a deterministic function of the posterior and says nothing
                    # about whether DEVIATING pays. Recorded so predictability
                    # can be re-measured on the differing subset alone.
                    differ = int(bayes.myopic_step_exact(logL, available) != 0)
                    for B in BUDGETS:
                        rem = B - t
                        if rem < 1:
                            continue
                        a = rollout(bayes, x, logL, available, rem)
                        b = rollout(bayes, x, logL, available, rem, forced_first=0)
                        rows.append({**feats, "seed": seed, "i": int(i), "B": B,
                                     "t": t, "differ": differ,
                                     "delta": int(b == y) - int(a == y)})
                g = bayes.myopic_step_exact(logL, available)
                available.remove(g)
                acquired.append(g)
                logL = logL + bayes.loglik_cols(x, bayes.group_cols[g])
        print(f"    seed {seed}: done")

    print(f"\n[1] Mean value of forcing the strategic action, by (fork step, budget)")
    print(f"    {'t':>3} {'B':>3} {'n':>6} {'mean delta':>11} {'95% CI':>18} "
          f"{'help':>6} {'hurt':>6}")
    cells = defaultdict(list)
    for r in rows:
        cells[(r["t"], r["B"])].append(r)
    for key in sorted(cells):
        c = cells[key]
        d = np.array([r["delta"] for r in c], dtype=float)
        h = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        print(f"    {key[0]:>3} {key[1]:>3} {len(c):>6} {d.mean():>+11.3f} "
              f"{f'[{d.mean()-h:+.3f}, {d.mean()+h:+.3f}]':>18} "
              f"{np.mean(d > 0):>6.1%} {np.mean(d < 0):>6.1%}")

    print(f"\n[2] Fraction of rows where the strategic action IS the myopic action")
    print("    (arms identical -> delta = 0 by construction, not by measurement)")
    for key in sorted(cells):
        c = cells[key]
        same = 1.0 - float(np.mean([r["differ"] for r in c]))
        print(f"    t={key[0]} B={key[1]}: {same:>6.1%} identical")

    names = [k for k in rows[0]
             if k not in ("seed", "i", "B", "t", "delta", "differ")]
    out = {}

    def predictability(c, label):
        yb = np.array([1 if r["delta"] > 0 else 0 for r in c])
        X = np.array([[r[n] for n in names] for r in c])
        var = float(X.var(axis=0).sum())
        if len(c) < 100 or yb.sum() < 25 or yb.sum() == len(yb) or var < 1e-12:
            reason = ("no state variance" if var < 1e-12
                      else f"too few positives ({int(yb.sum())})")
            return None, var, float(yb.mean()), reason
        groups = np.array([r["seed"] * 100003 + r["i"] for r in c])
        p = cross_val_predict(
            HistGradientBoostingClassifier(max_depth=3, max_iter=120, random_state=0),
            X, yb, cv=GroupKFold(n_splits=5).split(X, yb, groups),
            method="predict_proba")[:, 1]
        return float(roc_auc_score(yb, p)), var, float(yb.mean()), label

    print(f"\n[3] Is delta>0 PREDICTABLE from state features?")
    print("    ALL = every forked row.  DIFFER = only rows where the strategic")
    print("    action differs from what myopic would have done. Only the second")
    print("    column measures whether DEVIATING pays; the first can score well")
    print("    merely by detecting that the two arms coincide.")
    print(f"\n    {'t':>3} {'B':>3} {'n(all)':>7} {'AUC all':>8} "
          f"{'n(diff)':>8} {'base':>7} {'AUC diff':>9}  verdict")
    for key in sorted(cells):
        c = cells[key]
        a_auc, var, _, a_note = predictability(c, "ok")
        d = [r for r in c if r["differ"]]
        d_auc, _, d_base, d_note = predictability(d, "ok")
        a_s = f"{a_auc:>8.3f}" if a_auc is not None else f"{'n/a':>8}"
        d_s = f"{d_auc:>9.3f}" if d_auc is not None else f"{'n/a':>9}"
        if d_auc is None:
            verdict = d_note
        elif d_auc > 0.60:
            verdict = "STATE SIGNAL survives"
        elif d_auc > 0.55:
            verdict = "weak state signal"
        else:
            verdict = "no state signal -- budget lookup suffices"
        print(f"    {key[0]:>3} {key[1]:>3} {len(c):>7} {a_s} "
              f"{len(d):>8} {d_base:>7.3f} {d_s}  {verdict}")
        out[str(key)] = {"auc_all": a_auc, "auc_differ": d_auc,
                         "n_differ": len(d), "base_differ": d_base,
                         "feat_var": var}

    print(f"\n[4] Effect size among rows where the arms actually differ")
    print(f"    {'t':>3} {'B':>3} {'n':>7} {'mean delta':>11} {'95% CI':>18}")
    for key in sorted(cells):
        d = [r["delta"] for r in cells[key] if r["differ"]]
        if len(d) < 30:
            continue
        arr = np.array(d, dtype=float)
        h = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr))
        print(f"    {key[0]:>3} {key[1]:>3} {len(arr):>7} {arr.mean():>+11.3f} "
              f"{f'[{arr.mean()-h:+.3f}, {arr.mean()+h:+.3f}]':>18}")

    Path("results").mkdir(exist_ok=True)
    Path("results/cube_nm_switch.json").write_text(json.dumps(
        {"cells": {str(k): {"n": len(v),
                            "mean_delta": float(np.mean([r["delta"] for r in v]))}
                   for k, v in cells.items()},
         "predictability": out, "features": names,
         "seeds": SEEDS, "n_test": N_TEST}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
