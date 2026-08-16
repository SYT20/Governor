#!/usr/bin/env python3
"""Gate A — replace the realised outcome label with an expected-value target.

The selector was trained on D = 1[M2 correct] - 1[H correct] from ONE realised
episode. Both policies are deterministic given the instance, so the noise is not
policy stochasticity: it is that a single posterior state s is consistent with
many instances x, and each s was labelled by the one x that produced it. The
Governor needs Delta*(s) = E[U(M2) - U(H) | s], which is what it will have to
estimate at decision time.

Delta*(s) is computable from the agent's OWN posterior and needs no ground
truth: draw h ~ P(regime, context, label | s), synthesise the unobserved
features from N(MU[h,.], SD[h,.]), run both policies against that synthetic
completion, and average. Observed columns are held FIXED at their real values,
so the replicates are genuine completions of this state rather than fresh
episodes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from governor.envs.env5_modes import (  # noqa: E402
    InstrumentedBayes, h_gate_first, m2_plan)
from governor.envs.gated_family import N_LABELS  # noqa: E402
from governor.envs.probe_family import (  # noqa: E402
    ObservableProbeBayes, ProbeTask, make_config)

SIG = [0.10, 0.35, 0.60, 1.50]
N_STATES, N_REP, B = 60, 8, 6.0
NAMES = ["H_y", "max_py", "gap_py", "H_c", "max_pc",
         "H_regime", "max_pr", "gap_pr", "n_acq"]


def feats(ib, logL, n):
    p = ib.b._norm(logL)
    M = p.reshape(ib.b.R, ib.b.K, N_LABELS)
    py, pc, pr = M.sum(axis=(0, 1)), M.sum(axis=(0, 2)), M.sum(axis=(1, 2))
    sy, sc, sr = np.sort(py)[::-1], np.sort(pc)[::-1], np.sort(pr)[::-1]
    e = lambda q: -float((q * np.log(np.maximum(q, 1e-300))).sum())  # noqa: E731
    return [e(py), float(sy[0]), float(sy[0] - sy[1]), e(pc), float(sc[0]),
            e(pr), float(sr[0]), float(sr[0] - sr[1]), float(n)]


def roll(ib, x, logL, av, budget, first):
    logL, av, sp = logL.copy(), list(av), 0.0
    if first is not None:
        av.remove(first)
        sp += float(ib.cost[first])
        logL = logL + ib.b.loglik_cols(x, ib.group_cols[first])
    while True:
        g = ib.b.myopic_step(logL, av, budget - sp)
        if g is None:
            break
        av.remove(g)
        sp += float(ib.cost[g])
        logL = logL + ib.b.loglik_cols(x, ib.group_cols[g])
    return int(np.argmax(ib.b.label_posterior(logL)))


def delta_star(ib, x, logL, av, rem, seen, ah, am, rng, reps=N_REP):
    """E[U(M2) - U(H) | s] by posterior-predictive completion of THIS state."""
    p = ib.b._norm(logL)
    d = 0.0
    for _ in range(reps):
        h = int(rng.choice(len(p), p=p))
        xs = ib.b.MU[h] + ib.b.SD[h] * rng.standard_normal(ib.b.nf)
        xs[seen] = x[seen]                      # observed columns stay real
        y = h % N_LABELS                        # the label under this hypothesis
        d += int(roll(ib, xs, logL, av, rem, am) == y) \
            - int(roll(ib, xs, logL, av, rem, ah) == y)
    return d / reps


def main() -> int:
    print("=" * 76)
    print("GATE A — expected-value target vs realised-outcome target")
    print("=" * 76)
    X, DS, DR, G = [], [], [], []
    for gi, so in enumerate(SIG):
        t = ProbeTask(cfg=make_config(so, 1.0, 0.05), n_samples=N_STATES, seed=7)
        ib = InstrumentedBayes(ObservableProbeBayes(t))
        rng = np.random.default_rng(100 + gi)
        for i in range(N_STATES):
            x, y = t.features[i], int(t.labels[i])
            logL, av, seen = ib.prior_logL(), list(range(ib.n_groups)), []
            for _ in range(2):
                g = ib.b.myopic_step(logL, av, B)
                av.remove(g)
                seen += ib.group_cols[g]
                logL = logL + ib.b.loglik_cols(x, ib.group_cols[g])
            rem = B - 2.0
            ah = h_gate_first(ib, logL, av, rem, acquired=False)
            am = m2_plan(ib, logL, av, rem)
            if ah is None or am is None:
                continue
            X.append(feats(ib, logL, 2))
            DR.append(int(roll(ib, x, logL, av, rem, am) == y)
                      - int(roll(ib, x, logL, av, rem, ah) == y))
            DS.append(delta_star(ib, x, logL, av, rem, seen, ah, am, rng))
            G.append(gi)
        print(f"    sigma={so} done")
    X, DR, DS, G = map(np.array, (X, DR, DS, G))

    print(f"\n  n={len(DR)}   realised D: std {DR.std():.3f}   "
          f"Delta*: std {DS.std():.3f}")
    print(f"  corr(D, Delta*) = {np.corrcoef(DR, DS)[0,1]:+.3f}  "
          f"-- how much of D is signal")

    out = {}
    for tag, tgt in (("realised D", DR), ("Delta* (expected)", DS)):
        pred = np.zeros(len(tgt))
        for g in np.unique(G):
            te = G == g
            m = HistGradientBoostingRegressor(max_depth=3, max_iter=200,
                                              random_state=0).fit(X[~te], tgt[~te])
            pred[te] = m.predict(X[te])
        # value is always scored against the REALISED outcome, whichever target
        # was used for training -- otherwise the comparison is circular
        val = float(DR[pred > 0].sum() / len(DR)) if (pred > 0).any() else 0.0
        const = max(0.0, float(DR.mean()))
        oracle = float(np.maximum(DR, 0).mean())
        out[tag] = {"value": val, "const": const, "oracle": oracle,
                    "escalated": float(np.mean(pred > 0))}
        print(f"\n  trained on {tag}")
        print(f"    best constant        {const:+.4f}")
        print(f"    state-aware selector {val:+.4f}   escalated "
              f"{np.mean(pred > 0):.0%}")
        print(f"    oracle               {oracle:+.4f}")
        print(f"    residual captured    "
              f"{(val - const) / max(oracle - const, 1e-9):.0%}")

    Path("results").mkdir(exist_ok=True)
    Path("results/env5_gate_a.json").write_text(json.dumps(
        {"n": len(DR), "corr_D_Dstar": float(np.corrcoef(DR, DS)[0, 1]),
         "results": out, "n_rep": N_REP, "budget": B}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
