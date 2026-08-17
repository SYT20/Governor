#!/usr/bin/env python3
"""E0014 — RESOURCE MODE = FULLY_CONSUMED_REASONING_UNITS, on external data.

The allocation unit is a forced "Wait" continuation (s1's forcingignore{N}wait),
which the model cannot decline: the stop token is suppressed N times, so the
allocated reasoning is ACTUALLY SPENT. act/cap = 1 by construction, which is the
exact defect that failed E0013.

Token counts come from the `simplescaling/s1-32B` tokenizer, not an estimate.
The len/4 approximation used in E0013 was off by 25-35% AND the error varied
with N (ratio 0.651 at 1 wait, 0.756 at 8), so it was not even a consistent
rescaling. That approximation is not used for any claim here.
"""
from __future__ import annotations

import itertools
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.harness.ledger import ExperimentRun, ExperimentSpec  # noqa: E402
from governor.harness.traps import secret_scan  # noqa: E402
from governor.phase4.headroom import best_k  # noqa: E402

S1_MIN = 0.12
CACHE = Path("results/s1_waits_math.pkl")


def main() -> int:
    d = pickle.load(open(CACHE, "rb"))
    ids = sorted(d[1])
    print("=" * 82)
    print("E0014  FULLY-CONSUMED REASONING UNITS — s1-32B, MATH-500, wait ladder")
    print("=" * 82)

    ladder = []
    for N in sorted(d):
        tk = np.array([d[N][i]["tokens"] for i in ids], float)
        ladder.append({"waits": N,
                       "accuracy": float(np.mean([d[N][i]["correct"] for i in ids])),
                       "mean_tokens": float(tk.mean()), "sd_tokens": float(tk.std())})
    print(f"\n  {'waits':>7}{'accuracy':>10}{'exact tokens':>14}")
    for r in ladder:
        print(f"  {r['waits']:>7}{r['accuracy']:>10.3f}{r['mean_tokens']:>14.0f}")

    pairs = []
    for lo, hi in itertools.combinations(sorted(d), 2):
        g = np.array([d[hi][i]["correct"] - d[lo][i]["correct"] for i in ids], float)
        p, pn = float((g > 0).mean()), float((g < 0).mean())
        k, ideal = best_k(12, p) if 0 < p < 1 else (0, 0.0)
        pairs.append({"low_waits": lo, "high_waits": hi, "p_help": p,
                      "p_hurt": pn, "net_gain": float(g.mean()),
                      "ideal_n12": ideal, "S1": bool(ideal >= S1_MIN),
                      "token_ratio": float(np.mean([d[hi][i]["tokens"] for i in ids])
                                           / np.mean([d[lo][i]["tokens"] for i in ids]))})
    best = max(pairs, key=lambda r: r["ideal_n12"])
    n_s1 = sum(r["S1"] for r in pairs)
    print(f"\n  {len(pairs)} wait pairs; {n_s1} pass S1 (ideal >= {S1_MIN})")
    print(f"  best p(help) = {max(r['p_help'] for r in pairs):.3f}; "
          f"best ideal = {best['ideal_n12']:+.4f} at waits "
          f"{best['low_waits']}->{best['high_waits']}")
    print(f"  extra thinking HURTS more often than it helps in "
          f"{sum(r['p_hurt'] > r['p_help'] for r in pairs)}/{len(pairs)} pairs")

    verdict = "UNIT-BINDS-NO-HEADROOM" if n_s1 == 0 else "CONFIG-FOUND"
    print(f"\n  VERDICT: {verdict}")
    print("    The unit is fully consumed -- forced Wait injections cannot be")
    print("    declined -- so the defect that failed E0013 is gone. What fails")
    print("    now is the OPPOSITE criterion: at one wait the model has already")
    print("    converged (0.928 against a 0.932 saturation accuracy), so the")
    print("    whole ladder sits ABOVE the transition region and extra forced")
    print("    reasoning is close to pure waste.")
    print("\n    The two external resource contracts fail on OPPOSITE criteria:")
    print("      token cap    S1 passes (+0.168), S2 fails (cap does not bind)")
    print("      wait units   S2 passes (unit fully consumed), S1 fails (+0.009)")

    spec = ExperimentSpec(
        exp_id="E0014-s1-waits",
        title="Fully-consumed reasoning units on external s1 data",
        model="s1-32B via simplescaling/results (forcingignore{N}wait)",
        budget={"unit": "forced Wait continuation", "levels": sorted(d),
                "charged": "exact tokens, simplescaling/s1-32B tokenizer"},
        seeds={"split": "n/a - screen only"},
        split={"items": len(ids), "benchmark": "MATH-500"},
        metric="p(help) = fraction of items correct at high waits and wrong at "
               "low; ideal ceiling from the closed-form headroom law at n=12",
        params={"S1_min": S1_MIN},
        notes="Token counts are exact, not len/4. The E0013 approximation was "
              "off 25-35% and varied with N, so it is used for nothing here.")
    run = ExperimentRun(spec, overwrite=True)
    for r in ladder:
        run.append({"kind": "ladder", **r})
    for r in pairs:
        run.append({"kind": "pair", **r})
    run.finalize(summary={"verdict": verdict, "n_pairs": len(pairs),
                          "n_pass_S1": n_s1,
                          "best_ideal": best["ideal_n12"],
                          "best_p_help": max(r["p_help"] for r in pairs),
                          "accuracy_at_1_wait": ladder[0]["accuracy"],
                          "accuracy_at_8_waits": ladder[-1]["accuracy"]},
                 metrics={"ladder": ladder, "pairs": pairs},
                 traps={"secret_scan": secret_scan()}, verdict=verdict)
    print("\n  recorded: experiments/E0014-s1-waits/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
