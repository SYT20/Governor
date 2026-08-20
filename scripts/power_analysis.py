#!/usr/bin/env python3
"""E0022 — how many items would settle the real-LLM claim? Computed, not guessed.

I estimated 2000-3000 items last iteration. That was wrong by an order of
magnitude, and it was wrong because I scaled from a CI half-width instead of
from the observed per-item paired variance.

The paired difference between Governor and myopic is bimodal-ish over items:
most items agree (d=0) and the disagreements contribute +-1. That gives a small
mean over a large standard deviation, which is the worst possible ratio for
detection.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.harness.ledger import ExperimentRun, ExperimentSpec  # noqa: E402
from governor.harness.traps import exact_token_counts, secret_scan  # noqa: E402
from governor.phase4.s1data import feature_vector, load  # noqa: E402
from governor.phase4.softbudget import (  # noqa: E402
    enforced_alloc, fit_predictors, governor_alloc, myopic_alloc, tune,
)

B_STAR, TOKEN_SOURCE = 846.0, "simplescaling/s1-32B tokenizer (exact)"


def main() -> int:
    data = pickle.load(open("results/s1_caps_exact.pkl", "rb"))["math"]
    levels = sorted(data)
    ids = sorted(data[levels[0]])
    items, _ = load("math", budgets=[levels[0]])
    pr = {i.item_id: i.prompt for i in items}
    C = np.array([[data[b][i]["correct"] for b in levels] for i in ids], float)
    T = np.array([[data[b][i]["tokens"] for b in levels] for i in ids], float)
    X = np.array([feature_vector(pr[i]) for i in ids], float)
    cal = np.array([int(i[-5:]) % 2 == 0 for i in ids])
    ev = ~cal
    lp = fit_predictors(X[cal], C[cal], T[cal], levels, kind="logistic",
                        calibrate="isotonic")
    Qc, Tc = lp.predict(X[cal])
    Qe, Te = lp.predict(X[ev])
    lam = tune(lambda k: governor_alloc(Qc, Tc, k),
               np.concatenate([[0.0], np.geomspace(1e-7, 5e-2, 400)]), T[cal], B_STAR)
    tau = tune(lambda k: myopic_alloc(Qc, k), np.linspace(0, 1, 401), T[cal], B_STAR)
    Ce, Tt = C[ev], T[ev]
    n = len(Ce)
    r = np.arange(n)
    res = np.percentile(T[cal], 99, axis=0)
    gi = enforced_alloc(range(n), lambda i: governor_alloc(Qe, Te, lam)[i], Tt,
                        levels, B_STAR, reserve=res)
    mi = enforced_alloc(range(n), lambda i: myopic_alloc(Qe, tau)[i], Tt,
                        levels, B_STAR, reserve=res)
    d = (Ce[r, gi] - Ce[r, mi]).astype(float)
    dis = gi != mi
    print("=" * 80)
    print("E0022  POWER ANALYSIS — what sample would settle the real-LLM claim?")
    print("=" * 80)
    rows = []
    for label, v in (("Governor - myopic, all items", d),
                     ("Governor - myopic, disagreements only", d[dis])):
        m, s = float(v.mean()), float(v.std(ddof=1))
        N = (1.96 * s / m) ** 2 if m > 0 else float("inf")
        rows.append({"comparison": label, "n_observed": int(len(v)),
                     "mean": m, "sd": s, "n_required": float(N)})
        print(f"\n  {label}")
        print(f"    n={len(v)}  mean={m:+.4f}  sd={s:.4f}")
        print(f"    N for a 95% CI lower bound above zero: "
              f"{'unattainable (effect <= 0)' if m <= 0 else f'{N:,.0f} items'}")
    frac = float(dis.mean())
    print(f"\n  disagreement rate {frac:.0%}; MATH-500 supplies 500 items total, "
          f"{n} in evaluation")
    need = rows[0]["n_required"]
    print(f"\n  CONCLUSION: the required sample is ~{need/500:,.0f}x the entire "
          f"MATH-500 benchmark.")
    print("  The claim cannot be established OR refuted on this dataset. That is")
    print("  a property of the effect size relative to the per-item variance,")
    print("  not something more compute or a better tokenizer would fix.")
    print("\n  My previous estimate of 2000-3000 items was wrong by ~10x: it")
    print("  scaled from a bootstrap CI half-width rather than from the observed")
    print("  per-item paired variance. Most items agree (d=0) and disagreements")
    print("  contribute +-1, so the mean is small over a large sd.")

    spec = ExperimentSpec(
        exp_id="E0022-power", title="Power analysis for the real-LLM claim",
        model="s1-32B via simplescaling/results",
        budget={"B_star": B_STAR, "charged": TOKEN_SOURCE},
        seeds={"split": "doc_id parity"},
        split={"evaluation_items": n, "benchmark_total": 500},
        metric="items required for a 95% CI lower bound above zero, from the "
               "observed per-item paired mean and standard deviation",
        params={"lambda": float(lam), "tau": float(tau)},
        notes="Corrects a previous 2000-3000 item estimate that scaled from a "
              "CI half-width instead of per-item variance.")
    run = ExperimentRun(spec, overwrite=True)
    for row in rows:
        run.append(row)
    run.finalize(summary={"n_required_all_items": rows[0]["n_required"],
                          "n_required_disagreements": rows[1]["n_required"],
                          "benchmark_size": 500,
                          "multiple_of_benchmark": rows[0]["n_required"] / 500,
                          "disagreement_rate": frac,
                          "verdict": "SAMPLE-UNATTAINABLE"},
                 metrics={"rows": rows},
                 traps={"exact_token_counts": exact_token_counts(TOKEN_SOURCE),
                        "secret_scan": secret_scan()},
                 verdict="SAMPLE-UNATTAINABLE")
    print("\n  recorded: experiments/E0022-power/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
