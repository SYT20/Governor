#!/usr/bin/env python3
"""Recheck the MIXED cell (sigma=0.35, B=4) storing per-state draws.

Run E reported credM2 12 / credH 6 / straddle 22 at UNCORRECTED 95% CIs across
40 states. At 5% roughly 2 false positives per side are expected by chance, so
12 and 6 clear that comfortably -- but the honest test is a multiple-testing
correction, and Run E saved only summary statistics. This rerun is protocol-
identical and stores the draws so corrected counts can be computed.

MIXED must survive Bonferroni to justify a Governor. It is the first cell in the
project to reach this bar and the first result worth defending rather than
retracting.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scipy.stats import norm
from governor.envs.env5_modes import InstrumentedBayes, h_gate_first, m2_plan
from governor.envs.gated_family import N_LABELS
from governor.envs.probe_family import ObservableProbeBayes, ProbeTask, make_config

SIGMA, B, K_MAX, N_STATES = 0.35, 4.0, 256, 40


def roll(ib, x, logL, av, budget, first):
    logL, av, sp = logL.copy(), list(av), 0.0
    if first is not None:
        av.remove(first); sp += float(ib.cost[first])
        logL = logL + ib.b.loglik_cols(x, ib.group_cols[first])
    while True:
        g = ib.b.myopic_step(logL, av, budget - sp)
        if g is None: break
        av.remove(g); sp += float(ib.cost[g])
        logL = logL + ib.b.loglik_cols(x, ib.group_cols[g])
    return int(np.argmax(ib.b.label_posterior(logL)))


def main() -> int:
    t = ProbeTask(cfg=make_config(SIGMA, 1.0, 0.05), n_samples=N_STATES, seed=7)
    ib = InstrumentedBayes(ObservableProbeBayes(t)); rem = B - 2.0
    dr = []
    for i in range(N_STATES):
        x = t.features[i]
        logL, av, seen = ib.prior_logL(), list(range(ib.n_groups)), []
        for _ in range(2):
            g = ib.b.myopic_step(logL, av, B); av.remove(g)
            seen += ib.group_cols[g]
            logL = logL + ib.b.loglik_cols(x, ib.group_cols[g])
        ah = h_gate_first(ib, logL, av, rem, acquired=False)
        am = m2_plan(ib, logL, av, rem)
        if ah is None or am is None: continue
        p = ib.b._norm(logL); rng = np.random.default_rng(9000 + i); d = []
        for _ in range(K_MAX):
            h = int(rng.choice(len(p), p=p))
            xs = ib.b.MU[h] + ib.b.SD[h] * rng.standard_normal(ib.b.nf)
            xs[seen] = x[seen]; y = h % N_LABELS
            d.append(int(roll(ib, xs, logL, av, rem, am) == y)
                     - int(roll(ib, xs, logL, av, rem, ah) == y))
        dr.append(d)
        if (i + 1) % 10 == 0:
            print(f"  recheck state {i+1}/{N_STATES}", flush=True)
    D = np.array(dr, dtype=float)
    m = D.mean(axis=1); se = D.std(axis=1, ddof=1) / np.sqrt(K_MAX)
    z95, zb = 1.96, float(norm.ppf(1 - 0.025 / len(m)))
    res = {
        "n": len(m), "mean": float(m.mean()), "sd": float(m.std(ddof=1)),
        "min": float(m.min()), "max": float(m.max()),
        "uncorrected": {"credM2": int((m - z95 * se > 0).sum()),
                        "credH": int((m + z95 * se < 0).sum())},
        "bonferroni": {"z": zb, "credM2": int((m - zb * se > 0).sum()),
                       "credH": int((m + zb * se < 0).sum())},
        "draws": [list(map(int, r)) for r in dr]}
    res["verdict_bonferroni"] = (
        "MIXED" if res["bonferroni"]["credM2"] and res["bonferroni"]["credH"]
        else "not MIXED under correction")
    Path("results/env5_mixed_recheck.json").write_text(json.dumps(res, indent=2))
    print(f"\n  sigma={SIGMA} B={B:.0f}  mean {m.mean():+.4f}  SD {m.std(ddof=1):.4f}"
          f"  range [{m.min():+.3f},{m.max():+.3f}]", flush=True)
    print(f"  uncorrected 95%: credM2 {res['uncorrected']['credM2']}  "
          f"credH {res['uncorrected']['credH']}", flush=True)
    print(f"  BONFERRONI z={zb:.2f}: credM2 {res['bonferroni']['credM2']}  "
          f"credH {res['bonferroni']['credH']}  -> {res['verdict_bonferroni']}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
