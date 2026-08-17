#!/usr/bin/env python3
"""E0019 — does fixing the predictor's LOSS and CALIBRATION fix the allocation?

E0017 fitted ridge to a binary correctness target and its Governor captured 4-6%
of a demonstrated ceiling. E0018 showed the features do carry signal (AUC up to
0.741) and retracted the "no signal" diagnosis. This changes ONLY the predictor
and nothing else: same data, split, budget grid, features, allocator, baseline,
and soft-budget contract.

RANKING IS NOT PRICING. The Lagrangian computes qhat - lambda*that, so the SCALE
of qhat prices tokens, not just its order. AUC cannot see that; Brier and the
calibration curve can. Four predictor variants are compared, all fitted on
calibration only:

    ridge                 what E0017 used
    logistic (raw)        correct loss, uncalibrated probabilities
    logistic + Platt      sigmoid calibration
    logistic + isotonic   non-parametric calibration

No probe. The 500-token probe is deliberately NOT introduced here: it must earn
back its own token cost, and that question is only meaningful once the no-probe
allocator is understood.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.harness.ledger import ExperimentRun, ExperimentSpec  # noqa: E402
from governor.harness.traps import (  # noqa: E402
    exact_token_counts, oracle_leakage, secret_scan, split_leakage,
)
from governor.phase4.s1data import FEATURE_NAMES, feature_vector, load  # noqa: E402
from governor.phase4.softbudget import (  # noqa: E402
    fit_predictors, governor_alloc, myopic_alloc, realised, tune,
)

TOKEN_SOURCE = "simplescaling/s1-32B tokenizer (exact)"
VARIANTS = [("ridge", None), ("logistic", None),
            ("logistic", "sigmoid"), ("logistic", "isotonic")]


def envelope_fixed(C, T, budget):
    pts = sorted((float(T[:, j].mean()), float(C[:, j].mean()))
                 for j in range(C.shape[1]))
    best = None
    for (c0, u0), (c1, u1) in zip(pts, pts[1:]):
        if c0 <= budget <= c1 and c1 > c0:
            w = (budget - c0) / (c1 - c0)
            best = max(best if best is not None else -1.0, u0 + w * (u1 - u0))
    for c, u in pts:
        if c <= budget + 1e-9:
            best = max(best if best is not None else -1.0, u)
    return float(best if best is not None else pts[0][1])


def oracle_at(C, T, budget):
    best = -1.0
    for lam in np.concatenate([[0.0], np.geomspace(1e-7, 1e-1, 400)]):
        idx = np.argmax(C - lam * T, axis=1)
        r = np.arange(len(idx))
        if float(T[r, idx].mean()) <= budget + 1e-9:
            best = max(best, float(C[r, idx].mean()))
    return best


def ece(p, y, bins=10):
    """Expected calibration error: |confidence - accuracy| averaged over bins."""
    e, n = 0.0, len(y)
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (p >= lo) & (p < lo + 1.0 / bins)
        if m.sum():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="math", choices=("math", "gpqa"))
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    exp = f"E0019-predictor-loss-{a.bench}"

    data = pickle.load(open("results/s1_caps_exact.pkl", "rb"))[a.bench]
    levels = sorted(data)
    ids = sorted(data[levels[0]])
    items, _ = load(a.bench, budgets=[levels[0]])
    pr = {i.item_id: i.prompt for i in items}
    C = np.array([[data[b][i]["correct"] for b in levels] for i in ids], float)
    T = np.array([[data[b][i]["tokens"] for b in levels] for i in ids], float)
    X = np.array([feature_vector(pr[i]) for i in ids], float)
    cal = np.array([int(i[-5:]) % 2 == 0 for i in ids])
    ev = ~cal

    print("=" * 94)
    print(f"E0019  PREDICTOR LOSS AND CALIBRATION — {a.bench.upper()}")
    print("=" * 94)
    print(f"  {int(cal.sum())} calibration / {int(ev.sum())} evaluation items; "
          f"no probe; contract and allocator unchanged")

    lo, hi = float(T[:, 0].mean()), float(T[:, -1].mean())
    grid = np.linspace(lo * 1.15, hi * 0.75, 7)
    lam_grid = np.concatenate([[0.0], np.geomspace(1e-7, 5e-2, 400)])
    tau_grid = np.linspace(0.0, 1.0, 401)

    rows, summary = [], {}
    for kind, calib in VARIANTS:
        name = f"{kind}" + (f"+{calib}" if calib else "")
        lp = fit_predictors(X[cal], C[cal], T[cal], levels, kind=kind,
                            calibrate=calib)
        Qc, Tc = lp.predict(X[cal])
        Qe, Te = lp.predict(X[ev])
        # calibration quality on the CALIBRATION half, per level
        briers, eces = [], []
        for j, b in enumerate(levels):
            y = C[cal][:, j]
            if len(np.unique(y)) > 1:
                briers.append(float(np.mean((Qc[:, j] - y) ** 2)))
                eces.append(ece(Qc[:, j], y))
        best = None
        for B in grid:
            lam = tune(lambda k: governor_alloc(Qc, Tc, k), lam_grid, T[cal], B)
            tau = tune(lambda k: myopic_alloc(Qc, k), tau_grid, T[cal], B)
            gi, mi = governor_alloc(Qe, Te, lam), myopic_alloc(Qe, tau)
            g, m = realised(gi, C[ev], T[ev]), realised(mi, C[ev], T[ev])
            fx = envelope_fixed(C[ev], T[ev], B)
            row = {"variant": name, "budget": float(B),
                   "fixed": fx, "myopic": m["utility"],
                   "governor": g["utility"], "gov_tokens": g["mean_tokens"],
                   "gov_minus_fixed": g["utility"] - fx,
                   "gov_minus_myopic": g["utility"] - m["utility"],
                   "lam": float(lam), "tau": float(tau)}
            rows.append(row)
            if best is None or row["gov_minus_fixed"] > best["gov_minus_fixed"]:
                best = row
        summary[name] = {"brier": float(np.mean(briers)) if briers else float("nan"),
                         "ece": float(np.mean(eces)) if eces else float("nan"),
                         "best": best}
        print(f"\n  {name:<20} mean Brier {summary[name]['brier']:.4f}  "
              f"mean ECE {summary[name]['ece']:.4f}")
        print(f"    best point: B={best['budget']:.0f}  fixed {best['fixed']:.4f}"
              f"  myopic {best['myopic']:.4f}  GOV {best['governor']:.4f}"
              f"  G-fix {best['gov_minus_fixed']:+.4f}"
              f"  G-myo {best['gov_minus_myopic']:+.4f}")

    # bootstrap the best variant by CALIBRATION-side Brier (never by outcome)
    pick = min(summary, key=lambda k: summary[k]["brier"])
    print(f"\n  variant selected by calibration Brier (not by outcome): {pick}")
    kind, calib = VARIANTS[[f"{k}" + (f"+{c}" if c else "") for k, c in VARIANTS].index(pick)]
    lp = fit_predictors(X[cal], C[cal], T[cal], levels, kind=kind, calibrate=calib)
    Qc, Tc = lp.predict(X[cal]); Qe, Te = lp.predict(X[ev])
    B = summary[pick]["best"]["budget"]
    lam = tune(lambda k: governor_alloc(Qc, Tc, k), lam_grid, T[cal], B)
    tau = tune(lambda k: myopic_alloc(Qc, k), tau_grid, T[cal], B)
    gi, mi = governor_alloc(Qe, Te, lam), myopic_alloc(Qe, tau)
    Ce, Tt = C[ev], T[ev]
    rng = np.random.default_rng(0); n = int(ev.sum())
    dgf, dgm = [], []
    for _ in range(a.boot):
        s = rng.integers(0, n, n)
        ug = float(Ce[s, gi[s]].mean())
        dgf.append(ug - envelope_fixed(Ce[s], Tt[s], B))
        dgm.append(ug - float(Ce[s, mi[s]].mean()))
    def ci(v):
        v = np.array(v)
        return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    mf, lf, hf = ci(dgf); mm, lm, hm = ci(dgm)
    orc = oracle_at(Ce, Tt, B)
    print(f"\n  BOOTSTRAP ({n} evaluation items, {a.boot} resamples) at B={B:.0f}")
    print(f"    ceiling (oracle - fixed): {orc - envelope_fixed(Ce, Tt, B):+.4f}")
    print(f"    PRIMARY   GOV - fixed  : {mf:+.4f} [{lf:+.4f}, {hf:+.4f}]  "
          f"{'BEATS' if lf > 0 else 'LOSES' if hf < 0 else 'not separable'}")
    print(f"    SECONDARY GOV - myopic : {mm:+.4f} [{lm:+.4f}, {hm:+.4f}]  "
          f"{'BEATS' if lm > 0 else 'LOSES' if hm < 0 else 'not separable'}")

    traps = {"oracle_leakage": oracle_leakage(FEATURE_NAMES),
             "exact_token_counts": exact_token_counts(TOKEN_SOURCE),
             "split_leakage": split_leakage([i for i, c in zip(ids, cal) if c],
                                            [i for i, e in zip(ids, ev) if e]),
             "secret_scan": secret_scan()}
    red = [k for k, (ok, _) in traps.items() if not ok]
    passed = lf > 0 and not red
    print(f"\n  traps: {'all green' if not red else red}")
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")

    spec = ExperimentSpec(
        exp_id=exp, title=f"Predictor loss and calibration, no probe ({a.bench})",
        model="s1-32B via simplescaling/results",
        budget={"contract": "SOFT_EXPECTED_BUDGET", "reported_at": float(B),
                "charged": TOKEN_SOURCE},
        seeds={"split": "doc_id parity", "bootstrap": 0},
        split={"calibration_items": int(cal.sum()), "evaluation_items": n},
        metric="mean correctness at matched expected tokens; variants differ "
               "ONLY in the correctness predictor's loss and calibration; the "
               "variant is chosen by calibration-side Brier, never by outcome",
        params={"variants": [f"{k}" + (f"+{c}" if c else "") for k, c in VARIANTS],
                "selected": pick, "features": list(FEATURE_NAMES)},
        notes="No probe. The probe must earn its own token cost and that is a "
              "separate question.")
    run = ExperimentRun(spec, overwrite=True)
    for r in rows:
        run.append(r)
    run.finalize(summary={"benchmark": a.bench, "selected_variant": pick,
                          "brier_by_variant": {k: v["brier"] for k, v in summary.items()},
                          "ece_by_variant": {k: v["ece"] for k, v in summary.items()},
                          "primary_mean": mf, "primary_lo": lf, "primary_hi": hf,
                          "secondary_mean": mm, "secondary_lo": lm, "secondary_hi": hm,
                          "ceiling": orc - envelope_fixed(Ce, Tt, B),
                          "verdict": "PASS" if passed else "FAIL"},
        metrics={"rows": rows,
                 "summary": {k: {"brier": v["brier"], "ece": v["ece"],
                                 "best": v["best"]} for k, v in summary.items()}},
        traps=traps, verdict="PASS" if passed else "FAIL")
    print(f"\n  recorded: experiments/{exp}/")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
