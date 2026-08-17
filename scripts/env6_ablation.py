#!/usr/bin/env python3
"""Policy-equivalence ablation: is the learned Governor doing anything a
hand-derived observable-optimal allocator does not?

The learned Governor beat the cue heuristic by +0.0116. That could mean it
learned something, or it could mean the problem is analytically easy and any
correct observable rule would do. This separates the two.

  cue_greedy   deep iff cue says hard (the simple heuristic)
  handcoded    SAME opportunity-cost rule as the Governor, but with gains
               computed ANALYTICALLY from the known environment rather than
               learned. This is the observable-optimal reference.
  GOVERNOR     learned gains from calibration data

  handcoded ~= GOVERNOR  -> architecture validated, learning adds nothing
  GOVERNOR  <  handcoded -> the learner is lossy; report the gap
  GOVERNOR  >  handcoded -> the learner found something the analysis missed
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.ensemble import HistGradientBoostingRegressor
from governor.gate import env6 as E
from governor.gate.executor import run_episode

B, N_CAL, N_TEST, SEEDS = 2.0, 800, 500, [101, 202, 303]


def const_policy(sl): return lambda o, b: "M2" if (o["t"] in sl and b >= 1.0) else "H"
def cue_greedy(o, b): return "M2" if (o["cue"] == 1 and b >= 1.0) else "H"


def analytic_gain(cue_val):
    """E[acc(M2) - acc(H) | cue] by Bayes on the known generative model.
    ORACLE-DERIVED: legitimate for a reference policy, NOT given to the learner."""
    ph, nz = E.P_HARD, E.CUE_NOISE
    p_h_given = (ph * (1 - nz)) / (ph * (1 - nz) + (1 - ph) * nz) if cue_val == 1 \
        else (ph * nz) / (ph * nz + (1 - ph) * (1 - nz))
    return p_h_given * 0.35 + (1 - p_h_given) * 0.05


def opportunity_rule(q, pch):
    def pol(o, b):
        if b < 1.0: return "H"
        t, here = o["t"], q(o["t"], o["cue"])
        rem = 4 - t - 1
        if rem <= 0: return "M2" if here > 0 else "H"
        fut = pch * q(t + 1, 1) + (1 - pch) * q(t + 1, 0)
        return "M2" if (here >= fut or rem <= b - 1e-9) and here > 0 else "H"
    return pol


def main() -> int:
    cal = E.Env6(seed=0, n=N_CAL)
    X = np.array([[t, int(cal.cue[e][t])] for e in range(N_CAL) for t in range(4)], float)
    y = np.array([int(cal.roll[e][t] < E.ACC[("M2", int(cal.hard[e][t]))])
                  - int(cal.roll[e][t] < E.ACC[("H", int(cal.hard[e][t]))])
                  for e in range(N_CAL) for t in range(4)], float)
    m = HistGradientBoostingRegressor(max_depth=3, max_iter=150, random_state=0).fit(X, y)
    tab = m.predict(np.array([[t, c] for t in range(4) for c in (0, 1)], float))
    pch = float(X[:, 1].mean())
    gov = opportunity_rule(lambda t, c: float(tab[t * 2 + c]), pch)
    hand = opportunity_rule(lambda t, c: analytic_gain(c), pch)
    scheds = [s for k in range(3) for s in itertools.combinations(range(4), k)]
    best = max(scheds, key=lambda s: np.mean(
        [run_episode(cal, const_policy(set(s)), e, B).utility for e in range(N_CAL)]))

    print("POLICY-EQUIVALENCE ABLATION  (budget 2, 3 fresh seeds x 500 episodes)")
    print(f"  {'seed':>5} {'constant':>9} {'cue':>8} {'handcoded':>10} {'GOVERNOR':>9}"
          f" | {'GOV-hand':>18}")
    rows = []
    for sd in SEEDS:
        te = E.Env6(seed=sd, n=N_TEST); ix = range(N_TEST)
        U = {k: np.array([run_episode(te, p, e, B).utility for e in ix])
             for k, p in (("constant", const_policy(set(best))), ("cue", cue_greedy),
                          ("hand", hand), ("gov", gov))}
        d = U["gov"] - U["hand"]; se = d.std(ddof=1) / np.sqrt(len(d))
        rows.append({k: float(v.mean()) for k, v in U.items()} |
                    {"seed": sd, "d_gov_hand": float(d.mean()),
                     "lo": float(d.mean() - 1.96 * se)})
        print(f"  {sd:>5} {U['constant'].mean():>9.4f} {U['cue'].mean():>8.4f} "
              f"{U['hand'].mean():>10.4f} {U['gov'].mean():>9.4f} | "
              f"{f'{d.mean():+.4f} [{d.mean()-1.96*se:+.4f}]':>18}")
    dm = np.mean([r["d_gov_hand"] for r in rows])
    print(f"\n  mean GOVERNOR - handcoded: {dm:+.4f}")
    print("  -> " + ("learner MATCHES the observable-optimal reference; the"
                     " architecture is what works, not the model class"
                     if abs(dm) < 0.005 else
                     ("learner BEATS the analytic reference" if dm > 0 else
                      "learner is LOSSY relative to the analytic reference")))
    Path("results/env6_ablation.json").write_text(json.dumps(rows, indent=2))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
