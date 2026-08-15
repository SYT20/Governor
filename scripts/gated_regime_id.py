#!/usr/bin/env python3
"""How fast can an observable agent tell WHAT KIND of task it is in?

Two questions, and the second is the one that decides whether Phase 2 is worth
building.

1. LEAKAGE AUDIT. At t=0 the agent has observed nothing, so its belief about the
   regime must equal the preregistered prior EXACTLY -- 0.200 mass on the truth,
   for every true regime. Any deviation means configuration information is
   reaching the policy through some path other than observation: array ordering,
   normalisation, a seed, a cached table. The previous version of this family
   failed exactly that audit, so it is now measured rather than asserted.

2. THE ECONOMIC QUESTION. Theoretical identifiability is not enough. Budgets in
   this family are 3-6 acquisitions. If distinguishing sigma_other=0.10 from
   1.50 takes five observations, the agent learns what kind of task it is in
   only after the budget that knowledge would have directed is already spent,
   and the metacognitive decision is worthless in practice even though it is
   well posed in theory.

So this reports the trajectory of P(true regime | o_1:t) against t, alongside
the budgets, and asks whether useful separation arrives while it can still be
acted on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.envs.gated_family import (  # noqa: E402
    REGIME_GRID,
    REGIME_PRIOR,
    GateConfig,
    GatedTask,
    ObservableBayes,
)

N = 200
MAX_T = 6


def main() -> int:
    print("=" * 86)
    print("REGIME IDENTIFICATION — can the agent learn what kind of task it is in,")
    print("                        soon enough for the answer to be actionable?")
    print("=" * 86)
    prior_mass = REGIME_PRIOR[0]
    print(f"\n  {len(REGIME_GRID)} regimes, preregistered prior {REGIME_PRIOR}")
    print(f"  budgets used by the family: 3-6 acquisitions\n")

    mass = np.zeros((len(REGIME_GRID), MAX_T + 1))
    top1 = np.zeros((len(REGIME_GRID), MAX_T + 1))

    for ri, so in enumerate(REGIME_GRID):
        task = GatedTask(cfg=GateConfig(sigma_other=so), n_samples=N, seed=11)
        bayes = ObservableBayes(task)
        for i in range(N):
            x = task.features[i]
            logL = bayes.prior_logL()
            available = list(range(bayes.n_groups))
            for t in range(MAX_T + 1):
                post = bayes.regime_posterior(logL)
                mass[ri, t] += post[ri]
                top1[ri, t] += int(np.argmax(post) == ri)
                if t == MAX_T:
                    break
                # the agent explores under its OWN myopic rule, not a script
                g = bayes.myopic_step(logL, available, np.inf)
                available.remove(g)
                logL = logL + bayes.loglik_cols(x, bayes.group_cols[g])
        mass[ri] /= N
        top1[ri] /= N
        print(f"    sigma_other={so:<5}  done")

    print(f"\n[1] Posterior mass on the TRUE regime, by acquisitions made")
    print(f"    {'true sigma':>11} " + " ".join(f"{'t=' + str(t):>7}"
                                                for t in range(MAX_T + 1)))
    print("    " + "-" * (11 + 8 * (MAX_T + 1)))
    for ri, so in enumerate(REGIME_GRID):
        print(f"    {so:>11.2f} " + " ".join(f"{mass[ri, t]:>7.3f}"
                                             for t in range(MAX_T + 1)))
    print(f"    {'prior':>11} " + " ".join(f"{prior_mass:>7.3f}"
                                           for _ in range(MAX_T + 1)))

    print(f"\n[2] Leakage audit at t=0 (must equal the prior EXACTLY)")
    worst = float(np.max(np.abs(mass[:, 0] - prior_mass)))
    clean = worst < 1e-9
    print(f"    max |P(true regime at t=0) - {prior_mass}| = {worst:.2e}   "
          f"{'CLEAN' if clean else 'LEAK'}")
    if not clean:
        for ri, so in enumerate(REGIME_GRID):
            print(f"      sigma_other={so}: {mass[ri, 0]:.6f}")

    print(f"\n[3] Top-1 regime identification accuracy (chance = "
          f"{1/len(REGIME_GRID):.2f})")
    print(f"    {'true sigma':>11} " + " ".join(f"{'t=' + str(t):>7}"
                                                for t in range(MAX_T + 1)))
    for ri, so in enumerate(REGIME_GRID):
        print(f"    {so:>11.2f} " + " ".join(f"{top1[ri, t]:>7.2f}"
                                             for t in range(MAX_T + 1)))
    mean_acc = top1.mean(axis=0)
    print(f"    {'mean':>11} " + " ".join(f"{mean_acc[t]:>7.2f}"
                                          for t in range(MAX_T + 1)))

    # Pinning the exact grid point is a harder question than the Governor
    # actually faces. It only needs to know which SIDE it is on -- is the gate
    # worth buying or not -- so scoring the binary question is the honest bar,
    # and it is the one that decides whether Phase 2 is buildable.
    print(f"\n[4] Binary 'is the gate worth buying' separation (the real question)")
    lo = [i for i, s in enumerate(REGIME_GRID) if s <= 0.20]
    side = np.zeros((len(REGIME_GRID), MAX_T + 1))
    for ri, so in enumerate(REGIME_GRID):
        task = GatedTask(cfg=GateConfig(sigma_other=so), n_samples=N, seed=11)
        bayes = ObservableBayes(task)
        truth_low = so <= 0.20
        for i in range(N):
            x = task.features[i]
            logL = bayes.prior_logL()
            available = list(range(bayes.n_groups))
            for t in range(MAX_T + 1):
                post = bayes.regime_posterior(logL)
                p_low = float(sum(post[j] for j in lo))
                side[ri, t] += int((p_low > 0.5) == truth_low)
                if t == MAX_T:
                    break
                g = bayes.myopic_step(logL, available, np.inf)
                available.remove(g)
                logL = logL + bayes.loglik_cols(x, bayes.group_cols[g])
        side[ri] /= N
    print(f"    {'true sigma':>11} " + " ".join(f"{'t=' + str(t):>7}"
                                                for t in range(MAX_T + 1)))
    for ri, so in enumerate(REGIME_GRID):
        tag = "gate poor" if so <= 0.20 else "gate good"
        print(f"    {so:>11.2f} " + " ".join(f"{side[ri, t]:>7.2f}"
                                             for t in range(MAX_T + 1))
              + f"   {tag}")
    ms = side.mean(axis=0)
    print(f"    {'mean':>11} " + " ".join(f"{ms[t]:>7.2f}"
                                          for t in range(MAX_T + 1)))
    t_star = next((t for t in range(MAX_T + 1) if ms[t] > 0.70), None)
    print(f"\n    first t with mean side-accuracy > 0.70: "
          f"{t_star if t_star is not None else 'never within ' + str(MAX_T)}")
    actionable = t_star is not None and t_star <= 3
    print(f"    budgets are 3-6, so the answer must arrive by t~2-3 to be "
          f"actionable -> {'ACTIONABLE' if actionable else 'TOO LATE TO ACT ON'}")

    Path("results").mkdir(exist_ok=True)
    Path("results/gated_regime_id.json").write_text(json.dumps(
        {"regimes": list(REGIME_GRID), "prior": list(REGIME_PRIOR),
         "mass_on_truth": mass.tolist(), "top1": top1.tolist(),
         "side_accuracy": side.tolist(), "t_star": t_star,
         "leak_free": bool(clean), "n": N}, indent=2))
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
