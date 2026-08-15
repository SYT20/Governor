#!/usr/bin/env python3
"""Is the regime hard to identify, or does the myopic explorer just not try?

The identification diagnostic measured regime belief while the agent acquired
under its MYOPIC rule -- which maximises information about the LABEL. Nothing
in that objective rewards learning what kind of task it is in. So a slow curve
there is ambiguous between two very different conclusions:

    (a) the regime is not cheaply identifiable  -> the family is unusable,
                                                   Phase 2 is not buildable
    (b) the myopic explorer never tries          -> identifying it is itself a
                                                   metareasoning act, which is
                                                   the thesis, not a defect

Deciding between them requires an explorer that DOES try. The regime is exactly
the question "are non-primary blocks informative?", and the cheapest evidence is
one feature from each of several DIFFERENT blocks: if two distant blocks both
look code-like, sigma_other must be small; if only one ever does, it is large.
Myopic acquisition has no reason to spread that way.

    myopic   the label-greedy rule, as before
    spread   one feature from each distinct block, round-robin
    hybrid   spread for two acquisitions, then myopic

If `spread` separates the regimes markedly faster, conclusion (b) holds and the
value of regime-directed exploration is a measurable quantity -- which is a
better result than the one this script was written to check.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.envs.gated_family import (  # noqa: E402
    REGIME_GRID,
    GateConfig,
    GatedTask,
    ObservableBayes,
)

N = 200
MAX_T = 6
LOW = [i for i, s in enumerate(REGIME_GRID) if s <= 0.20]


def spread_order(bayes) -> list[int]:
    """One feature from each distinct block, round-robin, skipping the gate."""
    out = []
    for j in range(bayes.M):
        for b in range(bayes.K):
            out.append(1 + b * bayes.M + j)
    return out


def run(explorer: str):
    side = np.zeros((len(REGIME_GRID), MAX_T + 1))
    mass = np.zeros((len(REGIME_GRID), MAX_T + 1))
    for ri, so in enumerate(REGIME_GRID):
        task = GatedTask(cfg=GateConfig(sigma_other=so), n_samples=N, seed=11)
        bayes = ObservableBayes(task)
        order = spread_order(bayes)
        truth_low = so <= 0.20
        for i in range(N):
            x = task.features[i]
            logL = bayes.prior_logL()
            available = list(range(bayes.n_groups))
            for t in range(MAX_T + 1):
                post = bayes.regime_posterior(logL)
                side[ri, t] += int((float(sum(post[j] for j in LOW)) > 0.5)
                                   == truth_low)
                mass[ri, t] += post[ri]
                if t == MAX_T:
                    break
                if explorer == "myopic" or (explorer == "hybrid" and t >= 2):
                    g = bayes.myopic_step(logL, available, np.inf)
                else:
                    g = next(q for q in order if q in available)
                available.remove(g)
                logL = logL + bayes.loglik_cols(x, bayes.group_cols[g])
        side[ri] /= N
        mass[ri] /= N
    return side, mass


def main() -> int:
    print("=" * 84)
    print("REGIME-DIRECTED EXPLORATION — is the regime hard to find, or unsought?")
    print("=" * 84)
    print(f"\n  Baseline note: at t=0 the posterior is the uniform prior, so the")
    print(f"  binary rule always answers 'gate good'. {len(REGIME_GRID)-len(LOW)}"
          f" of {len(REGIME_GRID)} regimes are gate-good, so 0.60 is the")
    print(f"  no-information baseline, NOT 0.50. Read every row against 0.60.\n")

    out = {}
    for explorer in ("myopic", "spread", "hybrid"):
        side, mass = run(explorer)
        out[explorer] = {"side": side.tolist(), "mass": mass.tolist()}
        ms, mm = side.mean(axis=0), mass.mean(axis=0)
        print(f"  {explorer}")
        print(f"    {'side acc':>10} " + " ".join(f"{'t=' + str(t):>6}"
                                                  for t in range(MAX_T + 1)))
        print(f"    {'':>10} " + " ".join(f"{ms[t]:>6.2f}"
                                          for t in range(MAX_T + 1)))
        print(f"    {'P(true)':>10} " + " ".join(f"{mm[t]:>6.3f}"
                                                 for t in range(MAX_T + 1)))
        out[explorer]["t_star"] = next(
            (t for t in range(MAX_T + 1) if ms[t] > 0.70), None)
        print(f"    first t above 0.70: {out[explorer]['t_star']}\n")

    print("  Verdict")
    for e in ("myopic", "spread", "hybrid"):
        s = np.array(out[e]["side"]).mean(axis=0)
        print(f"    {e:<8} gain over the 0.60 baseline at t=2: {s[2]-0.60:+.3f}, "
              f"t=3: {s[3]-0.60:+.3f}")
    sm = np.array(out["myopic"]["side"]).mean(axis=0)
    ss = np.array(out["spread"]["side"]).mean(axis=0)
    adv = float(max(ss[t] - sm[t] for t in (1, 2, 3)))
    print(f"\n    best early advantage of directed exploration: {adv:+.3f}")
    if adv > 0.05:
        print("    -> The regime IS cheaply identifiable; the myopic explorer")
        print("       simply has no objective that rewards finding it. Directed")
        print("       exploration has measurable value, which is the thesis.")
    else:
        print("    -> Directed exploration does not help either. The regime is")
        print("       genuinely not identifiable within the budget: Case B, and")
        print("       the family is not usable for Phase 2 as parameterised.")

    Path("results").mkdir(exist_ok=True)
    Path("results/gated_regime_explore.json").write_text(json.dumps(
        {"explorers": out, "baseline": 0.60, "n": N}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
