#!/usr/bin/env python3
"""Is the surviving uplift signal STATE, or just the budget lookup again?

The uplift model reached +8.3% precision over base at top-10%, which is real at
~8 SE. But its feature set contains `step` and `rem_budget`, and the earlier
switch experiment already established that a budget/step lookup is worth a lot
here. If those two columns carry the whole effect, the "learned controller" is
the same three-number table with a gradient booster wrapped around it.

Three feature sets, identical model and identical leave-one-seed-out protocol:

    regime_only   step, rem_budget                 -- the lookup table
    state_only    posterior + option scores        -- no step, no budget
    all           both

regime_only ~ all  =>  no state contribution; the controller is a lookup.
state_only  ~ base =>  the state carries nothing on its own.

Reads the cached rows from the uplift run, so this costs seconds rather than
re-running the branched counterfactual collection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

TOPK = [1, 5, 10, 20, 50]

# The first split called only step and rem_budget "regime". That is too weak a
# control: `n_blocks_touched` counts acquisitions and is a PROGRESS variable, not
# a cognitive one, so crediting it to "state" inflates the residual. Every group
# costs 1 here, so total spend == step and there is no separate cost trajectory
# to add; n_blocks_touched is the one misfiled column.
REGIME = {"step", "rem_budget"}
REGIME_PLUS = {"step", "rem_budget", "n_blocks_touched"}


def loso(X, y, seeds):
    p = np.zeros(len(y))
    for s in np.unique(seeds):
        te = seeds == s
        if X.shape[1] == 0:
            p[te] = y[~te].mean()
            continue
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                           random_state=0).fit(X[~te], y[~te])
        P = m.predict_proba(X[te])
        out = np.zeros(te.sum())
        for j, c in enumerate(m.classes_):
            out += c * P[:, j]
        p[te] = out
    return p


def main() -> int:
    f = np.load("results/cube_nm_uplift_rows.npz", allow_pickle=True)
    X, y, seeds = f["X"], f["y"], f["seed"]
    names = [str(n) for n in f["names"]]
    base_d, base_p = float(y.mean()), float(np.mean(y > 0))

    print("=" * 84)
    print("ABLATION — does the uplift signal come from STATE or from the budget lookup?")
    print("=" * 84)
    print(f"\n  {len(y)} rows, {len(np.unique(seeds))} seeds; "
          f"intervene-everywhere delta {base_d:+.3f}, precision {base_p:.1%}")
    print(f"  regime features: {sorted(REGIME)}")
    print(f"  state  features: {[n for n in names if n not in REGIME]}")

    sets = {
        "regime_only": [i for i, n in enumerate(names) if n in REGIME],
        "regime_plus": [i for i, n in enumerate(names) if n in REGIME_PLUS],
        "cognitive_only": [i for i, n in enumerate(names) if n not in REGIME_PLUS],
        "all": list(range(len(names))),
    }

    print(f"\n  {'feature set':<14} " +
          " ".join(f"{'top' + str(k) + '%':>15}" for k in TOPK))
    print("  " + "-" * (14 + 16 * len(TOPK)))
    out = {}
    for tag, cols in sets.items():
        p = loso(X[:, cols], y, seeds)
        cells, rec = [], {}
        for k in TOPK:
            m = max(1, int(len(y) * k / 100))
            idx = np.argsort(-p)[:m]
            d, pr = float(y[idx].mean()), float(np.mean(y[idx] > 0))
            cells.append(f"{d:>+8.3f}({pr:>4.0%})")
            rec[k] = {"delta": d, "precision": pr}
        out[tag] = rec
        print(f"  {tag:<14} " + " ".join(f"{c:>15}" for c in cells))
    print(f"  {'(everywhere)':<14} " +
          " ".join(f"{f'{base_d:>+8.3f}({base_p:>4.0%})':>15}" for _ in TOPK))

    print("\n  Verdict at top-10%:")
    for t in sets:
        print(f"    {t:<15} {out[t][10]['delta']:+.3f}")
    print(f"    {'(everywhere)':<15} {base_d:+.3f}")

    # PAIRED, CLUSTERED BY SEED. Both models rank the SAME 19020 rows, so an
    # unpaired binomial SE overstates the evidence. Held-out seeds are the
    # independent unit under leave-one-seed-out, so the honest test computes
    # each model's top-10% value within each held-out seed and pairs across the
    # ten of them.
    print("\n  Paired by held-out seed (the independent unit under LOSO):")
    per = {t: [] for t in sets}
    preds = {t: loso(X[:, cols], y, seeds) for t, cols in sets.items()}
    for sd in np.unique(seeds):
        m = seeds == sd
        n10 = max(1, int(m.sum() * 0.10))
        for t in sets:
            idx = np.argsort(-preds[t][m])[:n10]
            per[t].append(float(y[m][idx].mean()))
    for t in sets:
        v = np.array(per[t])
        print(f"    {t:<14} {v.mean():+.3f} +- {1.96*v.std(ddof=1)/np.sqrt(len(v)):.3f}")
    out["paired_top10"] = {t: [float(x) for x in per[t]] for t in sets}
    for base_set, label in (("regime_only", "beyond step+budget"),
                            ("regime_plus", "beyond ALL progress vars")):
        d = np.array(per["all"]) - np.array(per[base_set])
        h = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        verdict = ("survives" if d.mean() - h > 0
                   else "NOT separable from zero")
        print(f"    all - {base_set:<12} {d.mean():+.3f} "
              f"[{d.mean()-h:+.3f}, {d.mean()+h:+.3f}]   {label} -> {verdict}")
        out[f"contribution_vs_{base_set}"] = {
            "mean": float(d.mean()), "lo": float(d.mean() - h),
            "hi": float(d.mean() + h)}
    print("\n    Only the second line is evidence for a COGNITIVE signal; the")
    print("    first still credits acquisition-count progress to 'state'.")

    Path("results").mkdir(exist_ok=True)
    Path("results/cube_nm_uplift_ablation.json").write_text(json.dumps(
        {"base_delta": base_d, "base_precision": base_p, "curves": out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
