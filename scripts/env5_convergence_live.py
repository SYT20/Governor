#!/usr/bin/env python3
"""Run C — Delta* convergence by exact variance decomposition, checkpointed.

WHY THIS REPLACES THE FLOOR SUBTRACTION. Run A estimated the signal as
sqrt(Var(Dhat_k) - (0.473/sqrt(k))^2), using a GLOBAL noise constant. Var(D|s)
differs by state, so that constant is an approximation and the subtraction is
biased. Here the decomposition is computed from the replicates themselves:

    Var(Dhat_k) = Var_s(Delta*(s))  +  E_s[ Var(D|s) / k ]

Both terms are estimable per state, so no global constant is assumed.

WHY IT IS ~4x CHEAPER THAN RUN A. Run A recomputed every k from scratch:
40*(8+32+128+512) = 27,200 completions. Here each state draws 256 completions
ONCE and the smaller k are read off as prefixes. That is common random numbers
by construction -- the k=16 estimate is literally the first 16 draws of the
k=256 estimate -- and costs 40*256 = 10,240 completions total.

CHECKPOINTS every few states to results/env5_convergence.json, so a kill at any
point loses at most a few states rather than the whole run.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.envs.env5_modes import (  # noqa: E402
    InstrumentedBayes, h_gate_first, m2_plan)
from governor.envs.gated_family import N_LABELS  # noqa: E402
from governor.envs.probe_family import (  # noqa: E402
    ObservableProbeBayes, ProbeTask, make_config)

SIGMA, BUDGET, N_STATES, K_MAX = 0.10, 3.0, 40, 256
K_REPORT = [16, 64, 256]
OUT = Path("results/env5_convergence_live.json")


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


def main() -> int:
    print("=" * 74, flush=True)
    print(f"RUN C — Delta* convergence, sigma={SIGMA}, B={BUDGET}, "
          f"{N_STATES} states x {K_MAX} completions", flush=True)
    print("=" * 74, flush=True)
    t = ProbeTask(cfg=make_config(SIGMA, 1.0, 0.05), n_samples=N_STATES, seed=7)
    ib = InstrumentedBayes(ObservableProbeBayes(t))
    rem = BUDGET - 2.0
    draws: list[list[int]] = []
    t0 = time.time()

    for i in range(N_STATES):
        x = t.features[i]
        logL, av, seen = ib.prior_logL(), list(range(ib.n_groups)), []
        for _ in range(2):
            g = ib.b.myopic_step(logL, av, BUDGET)
            av.remove(g)
            seen += ib.group_cols[g]
            logL = logL + ib.b.loglik_cols(x, ib.group_cols[g])
        ah = h_gate_first(ib, logL, av, rem, acquired=False)
        am = m2_plan(ib, logL, av, rem)
        if ah is None or am is None:
            continue

        p = ib.b._norm(logL)
        rng = np.random.default_rng(9000 + i)     # CRN: fixed per state
        d = []
        for _ in range(K_MAX):
            h = int(rng.choice(len(p), p=p))
            xs = ib.b.MU[h] + ib.b.SD[h] * rng.standard_normal(ib.b.nf)
            xs[seen] = x[seen]                    # observed columns stay real
            y = h % N_LABELS
            d.append(int(roll(ib, xs, logL, av, rem, am) == y)
                     - int(roll(ib, xs, logL, av, rem, ah) == y))
        draws.append(d)

        if (i + 1) % 5 == 0 or i == N_STATES - 1:
            D = np.array(draws, dtype=float)
            rep = {}
            for k in K_REPORT:
                if k > D.shape[1]:
                    continue
                sub = D[:, :k]
                mean_s = sub.mean(axis=1)
                within = sub.var(axis=1, ddof=1).mean()      # E_s[Var(D|s)]
                total = mean_s.var(ddof=1)                   # Var(Dhat_k)
                between = max(total - within / k, 0.0)       # Var_s(Delta*)
                rep[str(k)] = {"var_total": float(total),
                               "var_within_over_k": float(within / k),
                               "var_between": float(between),
                               "sd_between": float(np.sqrt(between)),
                               "mean": float(mean_s.mean())}
            OUT.parent.mkdir(exist_ok=True)
            OUT.write_text(json.dumps(
                {"sigma": SIGMA, "budget": BUDGET, "states_done": len(draws),
                 "k_max": K_MAX, "decomposition": rep,
                 "elapsed_s": round(time.time() - t0, 1),
                 "draws": [list(map(int, r)) for r in draws]}, indent=2))
            el = time.time() - t0
            print(f"  state {i+1}/{N_STATES}  elapsed {el/60:.1f} min  "
                  f"eta {(el/(i+1)*(N_STATES-i-1))/60:.0f} min", flush=True)
            for k, v in rep.items():
                print(f"     k={k:>4}  Var_total {v['var_total']:.5f}  "
                      f"within/k {v['var_within_over_k']:.5f}  "
                      f"SD_between {v['sd_between']:.4f}", flush=True)

    D = np.array(draws, dtype=float)
    print(f"\n  FINAL — {len(draws)} states", flush=True)
    print(f"  {'k':>5} {'Var_total':>11} {'within/k':>11} {'Var_between':>12} "
          f"{'SD_between':>11}", flush=True)
    for k in K_REPORT:
        sub = D[:, :k]
        mean_s = sub.mean(axis=1)
        within = sub.var(axis=1, ddof=1).mean()
        total = mean_s.var(ddof=1)
        between = max(total - within / k, 0.0)
        print(f"  {k:>5} {total:>11.5f} {within/k:>11.5f} {between:>12.5f} "
              f"{np.sqrt(between):>11.4f}", flush=True)
    b256 = max(D.mean(axis=1).var(ddof=1)
               - D.var(axis=1, ddof=1).mean() / K_MAX, 0.0)
    print(f"\n  Var_between should be STABLE across k if Delta* genuinely varies.", flush=True)
    print(f"  A value collapsing toward 0 as k grows means the apparent variation", flush=True)
    print(f"  was Monte Carlo noise the subtraction had not yet removed.", flush=True)
    print(f"\n  SD_between at k={K_MAX}: {np.sqrt(b256):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# RUN D rationale (not launched until reviewed):
#   Run A closed sigma=0.60, B=6 -- Delta* varies (SD 0.0345) but its range is
#   [-0.180, -0.016], entirely negative, so a perfect predictor still says
#   "stay with H" everywhere.
#   The four-configuration sweep found the SIGN of Delta(M2-H) reverses:
#   +0.267 at sigma=0.10/B=3 against -0.156 at sigma=0.60/B=6. This run applies
#   the identical 256-draw decomposition at the POSITIVE end. The question is
#   not whether Delta* varies -- Run A settled that -- but whether its range
#   STRADDLES ZERO. Only a straddling range gives a selector anything to select.
#     range entirely positive -> always escalate; a constant, not a controller
#     range straddling zero   -> the live region, and Env 5 has a real problem
#     range entirely negative -> Env 5 closes
