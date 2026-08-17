#!/usr/bin/env python3
"""Robustness sweep. Preregistered BEFORE running:
  budgets B in {1, 2, 3}
  CUE_NOISE in {0.10, 0.15, 0.25, 0.35}
  three fresh test seeds per cell
Governor is retrained from calibration in every cell. Nothing is hand-tuned.
PASS per cell: Governor - constant AND Governor - greedy both > 0, CI excl 0.
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.ensemble import HistGradientBoostingRegressor
from governor.gate import env6 as E
from governor.gate.executor import run_episode

N_CAL, N_TEST = 800, 500
BUDGETS, NOISES, SEEDS = [1.0, 2.0, 3.0], [0.10, 0.15, 0.25, 0.35], [101, 202, 303]


def const_policy(sl): return lambda o, b: "M2" if (o["t"] in sl and b >= 1.0) else "H"
def greedy(o, b): return "M2" if b >= 1.0 else "H"


def train(env, eps):
    X, y = [], []
    for e in eps:
        for t in range(4):
            h = int(env.hard[e][t]); r = env.roll[e][t]
            X.append([t, int(env.cue[e][t])])
            y.append(int(r < E.ACC[("M2", h)]) - int(r < E.ACC[("H", h)]))
    X, y = np.array(X, float), np.array(y, float)
    m = HistGradientBoostingRegressor(max_depth=3, max_iter=150, random_state=0).fit(X, y)
    return m, float(X[:, 1].mean())


def governor(model, pch, B):
    # EXACT precompute: the model has only 8 distinct inputs (t in 0..3, cue in
    # {0,1}). Calling model.predict on 1-row arrays inside the episode loop made
    # the sweep time out. Tabulating is identical arithmetic, ~1000x faster.
    TAB = model.predict(np.array([[t, c] for t in range(4) for c in (0, 1)], float))
    q = lambda t, c: float(TAB[t * 2 + c])
    def pol(o, b):
        if b < 1.0: return "H"
        t, here = o["t"], q(o["t"], o["cue"])
        rem = 4 - t - 1
        if rem <= 0: return "M2" if here > 0 else "H"
        fut = pch * q(t + 1, 1) + (1 - pch) * q(t + 1, 0)
        return "M2" if (here >= fut or rem <= b - 1e-9) and here > 0 else "H"
    return pol


def u(env, pol, eps, B):
    return np.array([run_episode(env, pol, e, B).utility for e in eps])


def main() -> int:
    print("ROBUSTNESS SWEEP — 3 budgets x 4 noise levels x 3 fresh seeds")
    print(f"{'B':>3} {'noise':>6} {'seed':>5} | {'const':>7} {'greedy':>7} "
          f"{'GOV':>7} | {'G-const':>17} {'G-greedy':>17}  verdict")
    rows, npass = [], 0
    for B, nz in itertools.product(BUDGETS, NOISES):
        E.CUE_NOISE = nz                      # module-level, read at construction
        cal_env = E.Env6(seed=0, n=N_CAL)
        model, pch = train(cal_env, range(N_CAL))
        scheds = [s for k in range(int(B) + 1) for s in itertools.combinations(range(4), k)]
        best = max(scheds, key=lambda s: u(cal_env, const_policy(set(s)), range(N_CAL), B).mean())
        gov = governor(model, pch, B)
        for sd in SEEDS:
            te = E.Env6(seed=sd, n=N_TEST); ix = range(N_TEST)
            uc, ug, uv = (u(te, const_policy(set(best)), ix, B),
                          u(te, greedy, ix, B), u(te, gov, ix, B))
            def ci(d):
                se = d.std(ddof=1) / np.sqrt(len(d)); return d.mean(), d.mean() - 1.96 * se
            dc, lc = ci(uv - uc); dg, lg = ci(uv - ug)
            ok = lc > 0 and lg > 0; npass += ok
            rows.append({"B": B, "noise": nz, "seed": sd, "const": float(uc.mean()),
                         "greedy": float(ug.mean()), "gov": float(uv.mean()),
                         "d_const": float(dc), "d_greedy": float(dg), "pass": bool(ok)})
            print(f"{B:>3.0f} {nz:>6.2f} {sd:>5} | {uc.mean():>7.4f} {ug.mean():>7.4f} "
                  f"{uv.mean():>7.4f} | {f'{dc:+.4f} [{lc:+.4f}]':>17} "
                  f"{f'{dg:+.4f} [{lg:+.4f}]':>17}  {'PASS' if ok else 'fail'}")
    print(f"\n  {npass}/{len(rows)} cells pass both comparisons")
    Path("results/env6_robust.json").write_text(json.dumps(rows, indent=2))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
