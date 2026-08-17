#!/usr/bin/env python3
"""E0017 — the learned Governor under SOFT_EXPECTED_BUDGET. External data.

PRIMARY    U(Governor) - U(best fixed)      does adaptive allocation pay?
SECONDARY  U(Governor) - U(myopic)          does PRICING the resource pay,
                                            beyond predicting difficulty?

The secondary comparison is the point. Phase 5 found the opportunity-cost DP
statistically indistinguishable from a `q>0` rule; this asks the same question
where the ceiling is large enough to answer it.

Everything is fitted and tuned on the frozen CALIBRATION half. Held-out items
are touched once, to report.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.harness.ledger import ExperimentRun, ExperimentSpec  # noqa: E402
from governor.harness.traps import (  # noqa: E402
    exact_token_counts, oracle_leakage, progress_as_cognition, secret_scan,
    split_leakage,
)
from governor.phase4.s1data import FEATURE_NAMES, feature_vector  # noqa: E402
from governor.phase4.softbudget import (  # noqa: E402
    fit_predictors, governor_alloc, myopic_alloc, realised, tune,
)

TOKEN_SOURCE = "simplescaling/s1-32B tokenizer (exact)"
SPLIT_RULE = "doc_id parity: even=calibration, odd=evaluation (declared in E0013)"


def envelope_fixed(C, T, budget):
    """Strongest fixed baseline: randomise between adjacent levels to hit B."""
    pts = sorted((float(T[:, j].mean()), float(C[:, j].mean()), j)
                 for j in range(C.shape[1]))
    best = None
    for (c0, u0, _), (c1, u1, _) in zip(pts, pts[1:]):
        if c0 <= budget <= c1 and c1 > c0:
            w = (budget - c0) / (c1 - c0)
            best = max(best or -1, u0 + w * (u1 - u0))
    for c, u, _ in pts:
        if c <= budget + 1e-9:
            best = max(best if best is not None else -1.0, u)
    if best is None:
        # A bootstrap resample can push every level's mean cost above B. The
        # cheapest level is then the only thing a fixed policy can do, and it is
        # the honest comparator rather than a crash or a silent skip.
        best = pts[0][1]
    return float(best)


def oracle_at(C, T, budget):
    """Lagrangian optimum with TRUE per-item outcomes: the ceiling."""
    best = -1.0
    for lam in np.concatenate([[0.0], np.geomspace(1e-7, 1e-1, 600)]):
        idx = np.argmax(C - lam * T, axis=1)
        r = np.arange(len(idx))
        if float(T[r, idx].mean()) <= budget + 1e-9:
            best = max(best, float(C[r, idx].mean()))
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="math", choices=("math", "gpqa"))
    ap.add_argument("--boot", type=int, default=600)
    ap.add_argument("--exp", default=None)
    a = ap.parse_args()
    exp = a.exp or f"E0017-soft-governor-{a.bench}"

    data = pickle.load(open("results/s1_caps_exact.pkl", "rb"))[a.bench]
    levels = sorted(data)
    ids = sorted(data[levels[0]])
    from governor.phase4.s1data import load
    items, _ = load(a.bench)
    prompt = {i.item_id: i.prompt for i in items}

    C = np.array([[data[b][i]["correct"] for b in levels] for i in ids], float)
    T = np.array([[data[b][i]["tokens"] for b in levels] for i in ids], float)
    X = np.array([feature_vector(prompt[i]) for i in ids], float)
    cal = np.array([int(i[-5:]) % 2 == 0 for i in ids])
    ev = ~cal

    print("=" * 92)
    print(f"E0017  SOFT-BUDGET GOVERNOR — {a.bench.upper()} ({len(ids)} items)")
    print("=" * 92)
    print(f"  contract E[sum actual tokens] <= B | tokens: {TOKEN_SOURCE}")
    print(f"  split: {cal.sum()} calibration / {ev.sum()} evaluation ({SPLIT_RULE})")

    lp = fit_predictors(X[cal], C[cal], T[cal], levels)
    print(f"  predictors (calibration CV R2): "
          f"correctness {min(lp.cv_r2_q.values()):+.3f}..{max(lp.cv_r2_q.values()):+.3f}, "
          f"cost {min(lp.cv_r2_t.values()):+.3f}..{max(lp.cv_r2_t.values()):+.3f}")

    Qc, Tc = lp.predict(X[cal])
    Qe, Te = lp.predict(X[ev])

    # Budget grid spans the useful region: cheapest level up to saturation.
    lo, hi = float(T[:, 0].mean()), float(T[:, -1].mean())
    grid = np.linspace(lo * 1.15, hi * 0.75, 7)
    lam_grid = np.concatenate([[0.0], np.geomspace(1e-7, 5e-2, 500)])
    tau_grid = np.linspace(0.0, 1.0, 501)

    rows = []
    print(f"\n  {'B':>7}{'fixed':>8}{'myopic':>8}{'GOV':>8}{'oracle':>8}"
          f"{'G-fix':>8}{'G-myo':>8}{'tok_G':>8}{'dev':>7}")
    for B in grid:
        lam = tune(lambda k: governor_alloc(Qc, Tc, k), lam_grid, T[cal], B)
        tau = tune(lambda k: myopic_alloc(Qc, k), tau_grid, T[cal], B)
        gi, mi = governor_alloc(Qe, Te, lam), myopic_alloc(Qe, tau)
        g, m = realised(gi, C[ev], T[ev]), realised(mi, C[ev], T[ev])
        fx, orc = envelope_fixed(C[ev], T[ev], B), oracle_at(C[ev], T[ev], B)
        rows.append({"budget": float(B), "lam": float(lam), "tau": float(tau),
                     "fixed": fx, "oracle": orc, "governor": g, "myopic": m,
                     "gov_minus_fixed": g["utility"] - fx,
                     "gov_minus_myopic": g["utility"] - m["utility"],
                     "budget_deviation": g["mean_tokens"] - B})
        print(f"  {B:>7.0f}{fx:>8.4f}{m['utility']:>8.4f}{g['utility']:>8.4f}"
              f"{orc:>8.4f}{g['utility']-fx:>+8.4f}"
              f"{g['utility']-m['utility']:>+8.4f}{g['mean_tokens']:>8.0f}"
              f"{g['mean_tokens']-B:>+7.0f}")

    # bootstrap over EVALUATION items, predictors and knobs frozen
    rng = np.random.default_rng(0)
    n_ev = int(ev.sum())
    Ce, Te_true = C[ev], T[ev]
    best_row = max(rows, key=lambda r: r["gov_minus_fixed"])
    B = best_row["budget"]
    lam, tau = best_row["lam"], best_row["tau"]
    gi, mi = governor_alloc(Qe, Te, lam), myopic_alloc(Qe, tau)
    dgf, dgm = [], []
    for _ in range(a.boot):
        s = rng.integers(0, n_ev, n_ev)
        r = np.arange(len(s))
        ug = float(Ce[s, gi[s]].mean()); um = float(Ce[s, mi[s]].mean())
        dgf.append(ug - envelope_fixed(Ce[s], Te_true[s], B))
        dgm.append(ug - um)
    def ci(v):
        v = np.array(v)
        return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    mf, lf, hf = ci(dgf); mm, lm, hm = ci(dgm)
    print(f"\n  BOOTSTRAP over {n_ev} evaluation items ({a.boot} resamples) at B={B:.0f}")
    print(f"    PRIMARY   GOVERNOR - best fixed : {mf:+.4f} [{lf:+.4f}, {hf:+.4f}]"
          f"  {'BEATS' if lf > 0 else 'LOSES' if hf < 0 else 'not separable'}")
    print(f"    SECONDARY GOVERNOR - myopic     : {mm:+.4f} [{lm:+.4f}, {hm:+.4f}]"
          f"  {'BEATS' if lm > 0 else 'LOSES' if hm < 0 else 'not separable'}")

    traps = {"oracle_leakage": oracle_leakage(FEATURE_NAMES),
             "progress_as_cognition": progress_as_cognition(FEATURE_NAMES),
             "split_leakage": split_leakage([i for i, c in zip(ids, cal) if c],
                                            [i for i, e in zip(ids, ev) if e]),
             "exact_token_counts": exact_token_counts(TOKEN_SOURCE),
             "secret_scan": secret_scan()}
    red = [k for k, (ok, _) in traps.items() if not ok]
    passed = lf > 0 and not red
    strong = passed and lm > 0
    print(f"\n  traps: {'all green' if not red else 'RED ' + str(red)}")
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}"
          f"{' + ARCHITECTURE (pricing beats prediction alone)' if strong else ''}")
    if passed and not strong:
        print("    The learned predictor is valuable; PRICING the resource is not")
        print("    yet justified beyond it.")

    spec = ExperimentSpec(
        exp_id=exp, title=f"Soft-budget learned Governor on external {a.bench}",
        model="s1-32B via simplescaling/results",
        budget={"contract": "SOFT_EXPECTED_BUDGET", "grid": [float(x) for x in grid],
                "reported_at": float(B), "charged": TOKEN_SOURCE},
        seeds={"split": SPLIT_RULE, "bootstrap": 0},
        split={"calibration_items": int(cal.sum()),
               "evaluation_items": n_ev, "rule": SPLIT_RULE},
        metric="mean correctness at matched expected tokens; primary = Governor "
               "minus the randomised-envelope fixed baseline; secondary = "
               "Governor minus myopic; 95% CI from a bootstrap over evaluation "
               "items with predictors and knobs frozen",
        params={"features": list(FEATURE_NAMES), "levels": levels,
                "n_bootstrap": a.boot},
        notes="Predictors and both knobs fitted/tuned on calibration only.")
    run = ExperimentRun(spec, overwrite=True)
    for r in rows:
        run.append({"kind": "budget_point", **r})
    run.finalize(summary={"benchmark": a.bench, "reported_budget": float(B),
                          "primary_mean": mf, "primary_lo": lf, "primary_hi": hf,
                          "primary_beats": bool(lf > 0),
                          "secondary_mean": mm, "secondary_lo": lm,
                          "secondary_hi": hm, "secondary_beats": bool(lm > 0),
                          "verdict": "PASS" if passed else "FAIL",
                          "architecture_justified": bool(strong)},
                 metrics={"budget_points": rows,
                          "cv_r2_correctness": lp.cv_r2_q,
                          "cv_r2_cost": lp.cv_r2_t},
                 traps=traps, verdict="PASS" if passed else "FAIL")
    print(f"\n  recorded: experiments/{exp}/")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
