#!/usr/bin/env python3
"""Allocation gate: is WHERE the two M2 calls go actually a choice?

The adequacy gate established that H and M2 differ 2-4 times per episode. That
is "something differs", not "compute allocation is worth learning". It recorded
|Delta*| only at disagreements, so it could not distinguish:

    Delta* = [+0.08, +0.07, -0.02, +0.01]   spend both early; allocation forced
    Delta* = [+0.01, +0.02, +0.12, -0.06]   save compute for later; real problem

Both look identical under |Delta*| and disagreement counts.

MEASURED HERE, signed and at EVERY decision point:
  A  Delta*_t distribution by t
  B  sign rates P(Delta*_t > 0) by t
  C  THE TEST: how often the best two opportunities are NOT the first two
        P( Top2(Delta*) != {0,1} )
  D  the advantage forgone by always spending early:
        sum of Top2(Delta*)  -  sum of Delta*_{0,1}

ACCEPTANCE, fixed before running: P(Top2 != {0,1}) materially positive AND the
mean forgone advantage nontrivial. Otherwise the budget binds technically but
the allocation is irrelevant, and the sequential executor is not worth building.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from governor.envs.env5_modes import InstrumentedBayes, h_gate_first, m2_plan
from governor.envs.gated_family import N_LABELS
from governor.envs.probe_family import ObservableProbeBayes, ProbeTask, make_config

CELLS = ((0.35, 6.0), (0.60, 6.0), (1.50, 6.0))
N_EP, N_DEC, K = 25, 4, 64


def _roll(ib, x, logL, av, B, first):
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


def main() -> int:
    print("ALLOCATION GATE — is WHERE the two M2 calls go a real choice?", flush=True)
    rows = []
    for so, B in CELLS:
        t_ = ProbeTask(cfg=make_config(so, 1.0, 0.05), n_samples=N_EP, seed=31337)
        ib = InstrumentedBayes(ObservableProbeBayes(t_))
        for i in range(N_EP):
            x = t_.features[i]
            logL, av, spent, seen = ib.prior_logL(), list(range(ib.n_groups)), 0.0, []
            ds = []
            for t in range(N_DEC):
                rem = B - spent
                ah = h_gate_first(ib, logL, av, rem, acquired=bool(seen))
                am = m2_plan(ib, logL, av, rem)
                if ah is None or am is None: break
                p = ib.b._norm(logL); rng = np.random.default_rng(11); d = 0.0
                for _ in range(K):
                    h = int(rng.choice(len(p), p=p))
                    xs = ib.b.MU[h] + ib.b.SD[h] * rng.standard_normal(ib.b.nf)
                    if seen: xs[seen] = x[seen]
                    y = h % N_LABELS
                    d += int(_roll(ib, xs, logL, av, rem, am) == y) \
                        - int(_roll(ib, xs, logL, av, rem, ah) == y)
                ds.append(d / K)
                g = ah
                av.remove(g); spent += float(ib.cost[g]); seen += ib.group_cols[g]
                logL = logL + ib.b.loglik_cols(x, ib.group_cols[g])
            if len(ds) == N_DEC: rows.append(ds)
        print(f"  sigma={so} done ({len(rows)} eps)", flush=True)

    D = np.array(rows)
    print(f"\n[A] Delta*_t by decision point ({len(D)} episodes)", flush=True)
    for t in range(N_DEC):
        print(f"    t={t}: mean {D[:,t].mean():+.4f}  median {np.median(D[:,t]):+.4f}"
              f"  SD {D[:,t].std(ddof=1):.4f}", flush=True)
    print(f"\n[B] sign rates", flush=True)
    for t in range(N_DEC):
        print(f"    t={t}: P(>0) {np.mean(D[:,t]>0):.0%}   P(<0) {np.mean(D[:,t]<0):.0%}",
              flush=True)
    top2 = [set(np.argsort(-r)[:2].tolist()) for r in D]
    early = set([0, 1])
    ne = np.array([s != early for s in top2])
    gain_best = np.array([sum(sorted(r, reverse=True)[:2]) for r in D])
    gain_early = D[:, 0] + D[:, 1]
    forg = gain_best - gain_early
    print(f"\n[C] THE TEST — best two opportunities are NOT the first two:", flush=True)
    print(f"    P(Top2 != {{0,1}}) = {ne.mean():.0%}", flush=True)
    print(f"\n[D] advantage forgone by always spending early", flush=True)
    se = forg.std(ddof=1)/np.sqrt(len(forg))
    print(f"    mean {forg.mean():+.4f} [{forg.mean()-1.96*se:+.4f}, "
          f"{forg.mean()+1.96*se:+.4f}]   max {forg.max():+.4f}", flush=True)
    print(f"    P(forgone > 0.02) = {np.mean(forg > 0.02):.0%}", flush=True)
    ok = ne.mean() > 0.25 and forg.mean() - 1.96*se > 0
    print(f"\n  ALLOCATION GATE: {'PASS -- build the executor' if ok else 'FAIL -- budget binds but allocation is irrelevant'}",
          flush=True)
    Path("results/env5_allocation.json").write_text(json.dumps(
        {"delta_by_t": D.tolist(), "p_top2_not_early": float(ne.mean()),
         "forgone_mean": float(forg.mean()), "pass": bool(ok)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
