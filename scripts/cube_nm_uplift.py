#!/usr/bin/env python3
"""Can a controller learn WHEN overriding the myopic policy is worth it?

This is the uplift formulation, not the classification one. The simulator gives
counterfactual outcomes for both policies at the same state, so the target is
the realised treatment effect

    delta(s, b) = V(s, pi_strategic, b) - V(s, pi_myopic, b)

and Governor's rule is `override iff delta_hat(s, b) > C_meta`.

FOUR CHANGES FROM THE AUC EXPERIMENT.

1. DISAGREEMENT-FILTERED BY CONSTRUCTION. Only states where the myopic action
   and the strategic action actually differ are collected. Elsewhere delta = 0
   for structural reasons and a model scores well by detecting arm coincidence
   rather than by predicting benefit. Filtering at collection time also buys
   ~3x the useful rows per unit of compute.

2. THE LEARNER GETS A FAIR FEATURE SET. The AUC run gave it posterior summaries
   only. A real metareasoning controller can also compute what its options are
   worth right now, so the myopic scores of the best action and of the strategic
   action -- and the gap between them -- are included. Withholding those would
   handicap the learner and make a negative result meaningless. Group INDEX is
   still excluded: it would let a model memorise this benchmark's fixed layout.

3. LEAVE-ONE-SEED-OUT, NOT POOLED CV. Each seed is an independently generated
   dataset. Training and testing within the same generative realisation is the
   weaker question.

4. POLICY-VALUE CURVES, NOT AUC. AUC 0.58 on the disagreement subset says the
   global ranking is weak, but a weak ranker can still be precise at the very
   top, which is the only region an override policy actually uses. Reported:
   RMSE / MAE / Spearman / R^2 against the constant predictor, plus realised
   mean delta and precision when intervening on the top k%.

Sample size is the reason this is a separate script: the AUC run had 672
disagreement rows, where the top 1% is seven rows. 10 seeds x 1500 instances
brings that into a range where a top-k curve means something.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from governor.envs.cube_nm_bayes import CubeNMBayes  # noqa: E402
from governor.envs.cube_nm_repro import N_LABELS, CubeNMRepro  # noqa: E402

SEEDS = list(range(1, 11))
N_INST = 1500
BUDGETS = [3, 4, 5]
FORKS = [1, 2]
TOPK = [1, 2, 5, 10, 20, 50, 100]


def features(bayes, logL, t, acquired, scores, rem):
    p = np.exp(logL - logL.max())
    p /= p.sum()
    M = p.reshape(bayes.K, N_LABELS)
    py, pc = M.sum(axis=0), M.sum(axis=1)
    sy, sc = np.sort(py)[::-1], np.sort(pc)[::-1]
    best = max(v for g, v in scores.items() if g != 0)
    ctx = scores[0]
    return {
        "step": float(t),
        "rem_budget": float(rem),
        "H_y": float(-(py * np.log(np.maximum(py, 1e-300))).sum()),
        "H_c": float(-(pc * np.log(np.maximum(pc, 1e-300))).sum()),
        "max_py": float(sy[0]), "gap_py": float(sy[0] - sy[1]),
        "max_pc": float(sc[0]), "gap_pc": float(sc[0] - sc[1]),
        "n_blocks_touched": float(len({(g - 1) // 10 for g in acquired if g})),
        # what the controller can compute about its own options right now
        "score_best": float(best),
        "score_ctx": float(ctx),
        "score_gap": float(best - ctx),
        "H_y_minus_best": float(-(py * np.log(np.maximum(py, 1e-300))).sum() + best),
    }


def rollout(bayes, x, logL, available, n, forced_first=None):
    logL = logL.copy()
    available = list(available)
    for s in range(n):
        g = forced_first if (s == 0 and forced_first is not None) \
            else bayes.myopic_step_exact(logL, available)
        available.remove(g)
        logL = logL + bayes.loglik_cols(x, bayes.group_cols[g])
    return int(np.argmax(bayes.label_posterior(logL)))


def collect():
    rows = []
    for seed in SEEDS:
        ds = CubeNMRepro(n_samples=N_INST + 200, seed=seed)
        bayes = CubeNMBayes(ds)
        for i in range(N_INST):
            x = ds.features[i]
            y = int(ds.labels[i])
            logL = np.zeros(bayes.H)
            available = list(range(bayes.n_groups))
            acquired: list[int] = []
            for t in range(max(FORKS) + 1):
                sc = bayes.myopic_scores_exact(logL, available)
                g = max(sc, key=lambda k: sc[k])
                # collect ONLY where the strategic action differs from myopic
                if t in FORKS and 0 in available and g != 0:
                    for B in BUDGETS:
                        rem = B - t
                        if rem < 1:
                            continue
                        a = rollout(bayes, x, logL, available, rem)
                        b = rollout(bayes, x, logL, available, rem,
                                    forced_first=0)
                        rows.append({
                            **features(bayes, logL, t, acquired, sc, rem),
                            "seed": seed, "B": B, "t": t,
                            "delta": float(int(b == y) - int(a == y))})
                available.remove(g)
                acquired.append(g)
                logL = logL + bayes.loglik_cols(x, bayes.group_cols[g])
        print(f"    seed {seed}: {len(rows)} rows so far")
    return rows


def fit_predict(kind, Xtr, ytr, Xte):
    """delta in {-1,0,1}: fit P(delta=+1) - P(delta=-1) as the effect estimate."""
    if kind == "constant":
        return np.full(len(Xte), ytr.mean())
    if kind == "ridge":
        m = Ridge(alpha=1.0).fit(Xtr, ytr)
        return m.predict(Xte)
    cls = np.array([-1.0, 0.0, 1.0])
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                       random_state=0).fit(Xtr, ytr)
    P = m.predict_proba(Xte)
    lut = {c: j for j, c in enumerate(m.classes_)}
    out = np.zeros(len(Xte))
    for c in cls:
        if c in lut:
            out += c * P[:, lut[c]]
    return out


def main() -> int:
    print("=" * 88)
    print("UPLIFT: can a controller learn WHEN to override the myopic policy?")
    print("=" * 88)
    print(f"\n[1] Collecting disagreement states "
          f"({len(SEEDS)} seeds x {N_INST} instances)")
    rows = collect()

    Path("results").mkdir(exist_ok=True)
    names = [k for k in rows[0] if k not in ("seed", "B", "t", "delta")]
    np.savez_compressed(
        "results/cube_nm_uplift_rows.npz",
        X=np.array([[r[n] for n in names] for r in rows]),
        y=np.array([r["delta"] for r in rows]),
        seed=np.array([r["seed"] for r in rows]),
        B=np.array([r["B"] for r in rows]), t=np.array([r["t"] for r in rows]),
        names=np.array(names))
    X = np.array([[r[n] for n in names] for r in rows])
    y = np.array([r["delta"] for r in rows])
    seed_of = np.array([r["seed"] for r in rows])
    print(f"\n    {len(rows)} disagreement rows; "
          f"delta>0 {np.mean(y > 0):.1%}, delta<0 {np.mean(y < 0):.1%}, "
          f"mean {y.mean():+.4f}")

    print("\n[2] Leave-one-seed-out effect prediction")
    print(f"    {'model':<10} {'RMSE':>8} {'MAE':>8} {'Spearman':>10} "
          f"{'R2 vs const':>12}")
    preds = {}
    for kind in ("constant", "ridge", "gbm"):
        p = np.zeros(len(y))
        for s in SEEDS:
            te = seed_of == s
            p[te] = fit_predict(kind, X[~te], y[~te], X[te])
        preds[kind] = p
        rmse = float(np.sqrt(np.mean((p - y) ** 2)))
        mae = float(np.mean(np.abs(p - y)))
        rho = float(spearmanr(p, y).statistic) if p.std() > 0 else 0.0
        r2 = 1 - np.sum((p - y) ** 2) / np.sum((preds["constant"] - y) ** 2)
        print(f"    {kind:<10} {rmse:>8.4f} {mae:>8.4f} {rho:>10.4f} "
              f"{r2:>+12.4f}")

    # A CEILING THAT A STATE FUNCTION COULD ACTUALLY REACH.
    # Ranking by realised delta ("oracle_realised") is NOT achievable by any
    # state-based ranker: delta is stochastic given s, because the posterior
    # summary does not determine the full feature vector, so two identical
    # states can produce different outcomes. That column tops out at +1.000 and
    # makes every model look hopeless for a reason no model can fix.
    # Fitting IN-SAMPLE on the evaluation fold overestimates the achievable
    # ceiling (it fits the noise too), so LOSO and in-sample bracket the true
    # best-state-function value from below and above.
    p_in = np.zeros(len(y))
    for s in SEEDS:
        te = seed_of == s
        p_in[te] = fit_predict("gbm", X[te], y[te], X[te])
    preds["gbm_insample"] = p_in

    print("\n[3] Policy value: intervene on the top k% by predicted uplift")
    print("    all           = intervene everywhere (the no-controller policy)")
    print("    gbm_insample  = OPTIMISTIC bound on the best state function")
    print("    oracle_real   = top k% by REALISED delta; NOT achievable by any")
    print("                    state-based ranker, shown only to bound the noise")
    cols = ("gbm", "ridge", "gbm_insample", "oracle_real")
    print(f"\n    {'top k%':>7} {'n':>6} " + " ".join(f"{m:>17}" for m in cols))
    curve = {}
    for k in TOPK:
        m = max(1, int(len(y) * k / 100))
        cells = []
        for kind in cols:
            score = y if kind == "oracle_real" else preds[kind]
            idx = np.argsort(-score)[:m]
            d, prec = float(y[idx].mean()), float(np.mean(y[idx] > 0))
            cells.append(f"{d:>+9.3f}({prec:>4.0%})")
            curve.setdefault(kind, {})[k] = {"delta": d, "precision": prec}
        print(f"    {k:>7} {m:>6} " + " ".join(f"{c:>17}" for c in cells))
    print(f"\n    {'all':>7} {len(y):>6} {y.mean():>+9.3f}({np.mean(y > 0):>4.0%})"
          f"   <- intervening everywhere")

    print("\n[4] Verdict")
    a = float(y.mean())
    g10 = curve["gbm"][10]["delta"]
    in10 = curve["gbm_insample"][10]["delta"]
    n10 = max(1, int(len(y) * 0.10))
    # binomial SE on the precision lift at top-10%
    p0, p1 = float(np.mean(y > 0)), curve["gbm"][10]["precision"]
    se = float(np.sqrt(p1 * (1 - p1) / n10 + p0 * (1 - p0) / len(y)))
    print(f"    top-10%: delta {g10:+.3f} vs intervene-everywhere {a:+.3f} "
          f"-> lift {g10 - a:+.3f}")
    print(f"    precision {p1:.1%} vs base {p0:.1%}; lift {p1 - p0:+.1%} "
          f"+- {1.96 * se:.1%} (95%)")
    print(f"    optimistic state-function ceiling (in-sample) {in10:+.3f}; "
          f"LOSO captures {(g10 - a) / max(in10 - a, 1e-9):.1%} of it")
    print(f"    monotonicity: top-1% {curve['gbm'][1]['delta']:+.3f} vs "
          f"top-20% {curve['gbm'][20]['delta']:+.3f} -- a real ranker is "
          f"strictly better at 1%")
    if g10 - a < 0.02:
        print("    NO USABLE TARGETING SIGNAL: selecting states is no better than")
        print("    intervening everywhere, so the controller adds nothing here.")

    Path("results").mkdir(exist_ok=True)
    Path("results/cube_nm_uplift.json").write_text(json.dumps(
        {"n_rows": len(rows), "mean_delta": float(y.mean()),
         "features": names, "curve": curve,
         "seeds": SEEDS, "n_inst": N_INST}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
