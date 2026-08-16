#!/usr/bin/env python3
"""Structural adequacy gate: can Env 5 support a 4-decision / 2-M2-call problem?

Run BEFORE building the sequential executor, not after. The prior claim that
"the allocation problem may be thin" rested on 2/12 disagreement at t=1 alone,
which does not generalise -- disagreement plausibly rises with depth.

Four measurements, preregistered:
  1. P(a_H != a_M2) at EACH decision point t
  2. distribution of N_diff = actionable opportunities per episode
  3. |Delta*| at disagreeing states -- structural variation is not decision
     value, the failure mode this project has hit repeatedly
  4. re-convergence: after a disagreement at t, do the branches still disagree
     at t+1? If they re-converge, later opportunities vanish.

DECISION RULE, fixed before running:
  most episodes N_diff = 0, or |Delta*| negligible  -> STOP, environment
                                                       cannot support it
  episodes commonly N_diff >= 2 with material |Delta*| -> build the executor
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
EPS = 0.02          # preregistered materiality threshold on |Delta*|


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


def dstar(ib, x, logL, av, rem, seen, ah, am):
    p = ib.b._norm(logL); rng = np.random.default_rng(11); d = 0.0
    for _ in range(K):
        h = int(rng.choice(len(p), p=p))
        xs = ib.b.MU[h] + ib.b.SD[h] * rng.standard_normal(ib.b.nf)
        if seen: xs[seen] = x[seen]
        y = h % N_LABELS
        d += int(_roll(ib, xs, logL, av, rem, am) == y) \
            - int(_roll(ib, xs, logL, av, rem, ah) == y)
    return d / K


def main() -> int:
    print("STRUCTURAL ADEQUACY GATE — before building the sequential executor",
          flush=True)
    by_t = {t: [] for t in range(N_DEC)}
    ndiff, mags, reconv = [], [], []
    for so, B in CELLS:
        t_ = ProbeTask(cfg=make_config(so, 1.0, 0.05), n_samples=N_EP, seed=31337)
        ib = InstrumentedBayes(ObservableProbeBayes(t_))
        for i in range(N_EP):
            x = t_.features[i]
            logL, av, spent, seen = ib.prior_logL(), list(range(ib.n_groups)), 0.0, []
            n = 0
            for t in range(N_DEC):
                rem = B - spent
                ah = h_gate_first(ib, logL, av, rem, acquired=bool(seen))
                am = m2_plan(ib, logL, av, rem)
                if ah is None or am is None: break
                diff = int(ah != am)
                by_t[t].append(diff)
                if diff:
                    n += 1
                    mags.append(abs(dstar(ib, x, logL, av, rem, seen, ah, am)))
                    # re-convergence: branch on M2, does t+1 still disagree?
                    lm = logL + ib.b.loglik_cols(x, ib.group_cols[am])
                    am2 = [g for g in av if g != am]
                    r2 = rem - float(ib.cost[am])
                    a2h = h_gate_first(ib, lm, am2, r2, acquired=True)
                    a2m = m2_plan(ib, lm, am2, r2)
                    if a2h is not None and a2m is not None:
                        reconv.append(int(a2h == a2m))
                g = ah
                av.remove(g); spent += float(ib.cost[g]); seen += ib.group_cols[g]
                logL = logL + ib.b.loglik_cols(x, ib.group_cols[g])
            ndiff.append(n)
        print(f"  sigma={so} done", flush=True)

    print(f"\n[1] P(a_H != a_M2) by decision point")
    for t in range(N_DEC):
        v = by_t[t]
        if v: print(f"    t={t}: {np.mean(v):.1%}  (n={len(v)})", flush=True)
    print(f"\n[2] actionable opportunities per episode (N_diff), {len(ndiff)} episodes")
    for k in range(N_DEC + 1):
        print(f"    N_diff={k}: {np.mean(np.array(ndiff)==k):.0%}", flush=True)
    print(f"    mean {np.mean(ndiff):.2f}   P(N_diff >= 2) = {np.mean(np.array(ndiff)>=2):.0%}")
    print(f"\n[3] |Delta*| at DISAGREEING states  (n={len(mags)})")
    if mags:
        m = np.array(mags)
        print(f"    mean {m.mean():.4f}  median {np.median(m):.4f}  "
              f"max {m.max():.4f}")
        print(f"    P(|Delta*| > {EPS}) = {np.mean(m > EPS):.0%}")
    print(f"\n[4] re-convergence after a disagreement  (n={len(reconv)})")
    if reconv:
        print(f"    P(branches agree again at t+1) = {np.mean(reconv):.0%}")
    ok = (np.mean(np.array(ndiff) >= 2) > 0.3 and mags
          and np.mean(np.array(mags) > EPS) > 0.3)
    print(f"\n  ADEQUACY: {'PASS -- build the executor' if ok else 'FAIL -- environment cannot support it'}")
    Path("results/env5_adequacy.json").write_text(json.dumps(
        {"by_t": {str(k): float(np.mean(v)) for k, v in by_t.items() if v},
         "ndiff": ndiff, "mags": [float(x) for x in mags],
         "reconv": reconv, "pass": bool(ok)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
