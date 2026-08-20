#!/usr/bin/env python3
"""The held-out Governor decision test. First one justified in this project.

Per-state significance is NOT the criterion. A controller does not need state
#37 to clear a Bonferroni threshold; it needs its DECISIONS to improve utility
on states it has never seen.

PREREGISTERED before running:
  train states  seed 7      (discovery, already characterised)
  test states   seed 31337  (fresh, never touched)
  configurations sigma in {0.35, 0.60, 1.50} at B=4 -- FROZEN, the replicated
                 candidate region; not reselected
  features      posterior-derived only: label/context/regime entropy and
                margin, acquisitions made. No sigma, no config id, no budget
                (it is constant at B=4 by construction).

  policies      always-H, always-M2, oracle (switches on true Delta*),
                Governor (observable state only)
  metric        U_G - U_H on held-out states, and the fraction of oracle
                headroom captured: (U_G - U_H) / (U_oracle - U_H)

COMPUTE COST is NOT subtracted from accuracy. That would repeat the
dimensional error that killed NetVDI -- accuracy and primitive operations are
different units. Under the Option A formulation compute enters as a hard
constraint via B_compute, and is reported alongside utility rather than folded
into it.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.ensemble import HistGradientBoostingRegressor
from governor.envs.env5_modes import InstrumentedBayes, h_gate_first, m2_plan
from governor.envs.gated_family import N_LABELS
from governor.envs.probe_family import ObservableProbeBayes, ProbeTask, make_config

CELLS = ((0.35, 4.0), (0.60, 4.0), (1.50, 4.0))
TRAIN_SEED, TEST_SEED = 7, 31337
N_STATES, K = 40, 256


def roll(ib, x, logL, av, B, first):
    logL, av, sp = logL.copy(), list(av), 0.0
    if first is not None:
        av.remove(first); sp += float(ib.cost[first])
        logL = logL + ib.b.loglik_cols(x, ib.group_cols[first])
    while True:
        g = ib.b.myopic_step(logL, av, B - sp)
        if g is None: break
        av.remove(g); sp += float(ib.cost[g])
        logL = logL + ib.b.loglik_cols(x, ib.group_cols[g])
    return int(np.argmax(ib.b.label_posterior(logL)))


def feats(ib, logL):
    p = ib.b._norm(logL); M = p.reshape(ib.b.R, ib.b.K, N_LABELS)
    py, pc, pr = M.sum(axis=(0,1)), M.sum(axis=(0,2)), M.sum(axis=(1,2))
    sy, sc, sr = np.sort(py)[::-1], np.sort(pc)[::-1], np.sort(pr)[::-1]
    e = lambda q: -float((q*np.log(np.maximum(q,1e-300))).sum())
    return [e(py), float(sy[0]), float(sy[0]-sy[1]), e(pc), float(sc[0]),
            e(pr), float(sr[0]), float(sr[0]-sr[1])]


def collect(seed, tag):
    X, DS, C = [], [], []
    for so, B in CELLS:
        t = ProbeTask(cfg=make_config(so,1.0,0.05), n_samples=N_STATES, seed=seed)
        ib = InstrumentedBayes(ObservableProbeBayes(t)); rem = B-2.0
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
            p = ib.b._norm(logL); rng = np.random.default_rng(seed*1000+i); d = []
            for _ in range(K):
                h = int(rng.choice(len(p), p=p))
                xs = ib.b.MU[h] + ib.b.SD[h]*rng.standard_normal(ib.b.nf)
                xs[seen] = x[seen]; y = h % N_LABELS
                d.append(int(roll(ib,xs,logL,av,rem,am)==y)
                         - int(roll(ib,xs,logL,av,rem,ah)==y))
            X.append(feats(ib, logL)); DS.append(float(np.mean(d))); C.append(so)
        print(f"  {tag}: sigma={so} done", flush=True)
    return np.array(X), np.array(DS), np.array(C)


def main() -> int:
    print("HELD-OUT GOVERNOR TEST — train seed 7, test seed 31337", flush=True)
    Xtr, Dtr, _ = collect(TRAIN_SEED, "train")
    Xte, Dte, Cte = collect(TEST_SEED, "test")
    m = HistGradientBoostingRegressor(max_depth=3, max_iter=200,
                                      random_state=0).fit(Xtr, Dtr)
    pred = m.predict(Xte)
    # utility is measured in the SAME unit throughout: expected accuracy.
    uH = 0.0
    uM2 = float(Dte.mean())
    uG = float(Dte[pred > 0].sum() / len(Dte))
    uO = float(np.maximum(Dte, 0).mean())
    print(f"\n  held-out states: {len(Dte)}   escalated {np.mean(pred>0):.0%}")
    print(f"    always-H            {uH:+.4f}")
    print(f"    always-M2           {uM2:+.4f}")
    print(f"    GOVERNOR            {uG:+.4f}")
    print(f"    oracle (true Delta*){uO:+.4f}")
    d = np.where(pred > 0, Dte, 0.0); se = d.std(ddof=1)/np.sqrt(len(d))
    print(f"\n    U_G - U_H = {uG:+.4f} [{uG-1.96*se:+.4f}, {uG+1.96*se:+.4f}]")
    print(f"    oracle headroom captured: {uG/max(uO,1e-9):.0%}")
    print(f"    beats always-M2 by {uG-uM2:+.4f}")
    ok = uG - 1.96*se > 0
    print(f"\n    {'GOVERNOR GATE PASSES' if ok else 'does NOT beat always-H with CI excluding zero'}")
    Path("results/env5_governor_test.json").write_text(json.dumps(
        {"n_test": len(Dte), "uH": uH, "uM2": uM2, "uG": uG, "uOracle": uO,
         "ci": [uG-1.96*se, uG+1.96*se], "escalated": float(np.mean(pred>0)),
         "passes": bool(ok)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
