#!/usr/bin/env python3
"""Run E — middle cells. The last open question for Environment 5.

Runs A/C/D applied the 256-draw Delta* protocol only to the two ENDPOINTS of the
observed sign reversal, then I generalised to the whole environment. That was an
overreach: a configuration whose aggregate Delta sits near zero could contain
states on both sides of it while the endpoints do not.

These three cells are chosen because their configuration-level Delta was nearest
zero in the 12-cell sweep -- 0.35/B4 +0.067, 0.60/B4 +0.000, 1.50/B4 +0.022 --
so they are where a within-configuration crossing is most plausible. Protocol is
byte-identical to C and D; nothing is tuned.

    MIXED   credible positive AND negative states in one configuration
            -> Environment 5 has a real selective-escalation problem
    other   -> Environment 5 closes
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from governor.envs.env5_modes import InstrumentedBayes, h_gate_first, m2_plan
from governor.envs.gated_family import N_LABELS
from governor.envs.probe_family import ObservableProbeBayes, ProbeTask, make_config

K_MAX, N_STATES = 256, 40
CELLS = ((0.35, 4.0), (0.60, 4.0), (1.50, 4.0))
OUT = Path("results/env5_middle_cells.json")


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


def main() -> int:
    out, t0 = {}, time.time()
    print("RUN E — middle cells, 40 states x 256 CRN draws", flush=True)
    for so, B in CELLS:
        t = ProbeTask(cfg=make_config(so, 1.0, 0.05), n_samples=N_STATES, seed=7)
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
                print(f"  s={so} B={B:.0f}  state {i+1}/{N_STATES}  "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)
        D = np.array(dr, dtype=float)
        m = D.mean(axis=1); se = D.std(axis=1, ddof=1) / np.sqrt(K_MAX)
        lo, hi = m - 1.96 * se, m + 1.96 * se
        cm, ch, st = int((lo > 0).sum()), int((hi < 0).sum()), int(((lo <= 0) & (hi >= 0)).sum())
        v = "MIXED" if (cm and ch) else ("strong M2" if cm else ("strong H" if ch else "inconclusive"))
        out[f"{so}|{B}"] = {"n": len(m), "mean": float(m.mean()), "sd": float(m.std(ddof=1)),
                            "min": float(m.min()), "max": float(m.max()),
                            "credM2": cm, "credH": ch, "straddle": st, "verdict": v}
        OUT.write_text(json.dumps(out, indent=2))
        print(f"  DONE s={so} B={B:.0f}: mean {m.mean():+.4f} SD {m.std(ddof=1):.4f} "
              f"range [{m.min():+.3f},{m.max():+.3f}] credM2 {cm} credH {ch} "
              f"straddle {st} -> {v}", flush=True)
    print(f"\n  ANY MIXED cell: {any(v['verdict']=='MIXED' for v in out.values())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
