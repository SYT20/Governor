#!/usr/bin/env python3
"""E0005 — Phase 4R structural search. NO CONTROLLER, NO API CALLS.

Searches (items per episode, LOW, HIGH, budget) against already-cached responses
and reports, for each configuration, the two preregistered structural quantities
and the ceiling.

    S1 competition    P(X > K) >= 0.60  and  E[X]/K >= 1.8
    S2 decidability   mean(actual cost) / cap(DEEP) >= 0.70

X is the number of items in an episode that benefit from the deep budget; K is
the number of upgrades the budget affords.

THE CONFIGURATION IS SELECTED BY S1 AND S2, NOT BY THE CEILING. Both criteria
were fixed in PREREGISTRATION-phase4R-structure.md before this ran. Picking the
configuration with the biggest ceiling would be selecting on the outcome, which
is the objection the preregistration exists to answer, so the ceiling is
printed for every cell and used only at the gate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.harness.ledger import ExperimentRun, ExperimentSpec  # noqa: E402
from governor.harness.traps import secret_scan  # noqa: E402
from governor.phase4.collect import ResponseCache, outcome  # noqa: E402
from governor.phase4.config import CAL_POOL_SEED, ENGINES, PROMPT_CAP  # noqa: E402
from governor.phase4.env import DEEP, P4Env, make_episodes  # noqa: E402
from governor.phase4.evaluate import constant, execute  # noqa: E402
from governor.phase4.policies import all_cheap, clairvoyant, greedy  # noqa: E402
from governor.phase4.tasks import make_pool  # noqa: E402

# FROZEN in the preregistration, before this script produced a number.
S1_P_MIN, S1_RATIO_MIN, S2_RATIO_MIN, CEILING_GATE = 0.60, 1.8, 0.70, 0.02

N_ITEMS = (4, 6, 8, 10)
MODE_PAIRS = ((300, 700), (300, 1400), (300, 2800), (700, 1400), (700, 2800))


def cached_pool(cache, budgets, n=400):
    return [i for i in make_pool(CAL_POOL_SEED, n)
            if all(cache.get(i, b) for b in budgets)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="qwen", choices=list(ENGINES))
    ap.add_argument("--exp", default="E0005-structure")
    a = ap.parse_args()
    cfg = ENGINES[a.engine]
    cache = ResponseCache(Path(cfg["cache"]), model=cfg["model"],
                          provider=cfg["provider"])

    print("=" * 100)
    print(f"E0005  PHASE 4R STRUCTURAL SEARCH — {cfg['model']}  (no API calls)")
    print("=" * 100)
    print(f"  FROZEN: S1 P(X>K)>={S1_P_MIN} and E[X]/K>={S1_RATIO_MIN} | "
          f"S2 actual/cap>={S2_RATIO_MIN} | ceiling gate >{CEILING_GATE}")

    spec = ExperimentSpec(
        exp_id=a.exp,
        title="Phase 4R structural search: which configuration can pose an "
              "allocation problem at all?",
        model=cfg["model"],
        budget={"swept": "per configuration", "charged": "usage.total_tokens"},
        seeds={"pool": CAL_POOL_SEED, "grouping": 7},
        split={"note": "calibration items only; the gate re-tests on held-out "
                       "episodes"},
        metric="S1 = P(X>K) and E[X]/K where X is useful opportunities and K "
               "affordable upgrades; S2 = mean actual DEEP cost / cap(DEEP); "
               "ceiling = U(oracle) - U(greedy) through the canonical executor",
        params={"n_items": list(N_ITEMS), "mode_pairs": [list(m) for m in MODE_PAIRS],
                "S1_p_min": S1_P_MIN, "S1_ratio_min": S1_RATIO_MIN,
                "S2_ratio_min": S2_RATIO_MIN, "ceiling_gate": CEILING_GATE},
        notes="Configuration selected by S1 and S2, both frozen before this ran. "
              "The ceiling is reported for every cell, never used to select.")
    run = ExperimentRun(spec, overwrite=True)

    rows = []
    print(f"\n  {'n':>3}{'LOW':>6}{'HIGH':>6}{'B':>7}{'items':>7}{'eps':>5}"
          f"{'P(X>K)':>8}{'E[X]/K':>8}{'act/cap':>8}"
          f"{'cheap':>7}{'greedy':>7}{'oracle':>7}{'CEILING':>9}  gate")
    for low, high in MODE_PAIRS:
        pool = cached_pool(cache, (low, high))
        if len(pool) < 40:
            print(f"  LOW={low} HIGH={high}: only {len(pool)} cached items, skipped")
            continue
        gains = np.array([outcome(cache, i, high)["correct"]
                          - outcome(cache, i, low)["correct"] for i in pool], float)
        act_hi = float(np.mean([outcome(cache, i, high)["total_tokens"]
                                for i in pool]))
        cap_lo, cap_hi = PROMPT_CAP + low, PROMPT_CAP + high
        s2 = act_hi / cap_hi
        p_useful = float((gains > 0).mean())

        for n in N_ITEMS:
            n_ep = len(pool) // n
            if n_ep < 6:
                continue
            eps = make_episodes(pool, n_ep, 7, n_items=n)
            E = list(range(n_ep))
            X = np.array([sum(1 for it in eps[e] if
                              outcome(cache, it, high)["correct"]
                              - outcome(cache, it, low)["correct"] > 0)
                          for e in E], float)
            for B in range(n * cap_lo, n * cap_lo + 5 * (cap_hi - cap_lo), 300):
                env = P4Env(cache, eps, low, high, float(B), PROMPT_CAP)
                g = execute(env, "g", constant(greedy(env)), E)
                K = np.array([sum(m == DEEP for m in ms) for ms in g.modes], float)
                if K.mean() < 0.5 or K.mean() > n - 0.5:
                    continue                      # degenerate: all or nothing
                pXK = float((X > K).mean())
                ratio = float(X.mean() / max(K.mean(), 1e-9))
                c = execute(env, "c", constant(all_cheap(env)), E)
                o = execute(env, "o", lambda e: clairvoyant(env, e), E)
                ceil = o.mean - g.mean
                s1 = pXK >= S1_P_MIN and ratio >= S1_RATIO_MIN
                gate = ("S1+S2" if (s1 and s2 >= S2_RATIO_MIN)
                        else "S1" if s1 else "S2" if s2 >= S2_RATIO_MIN else "-")
                row = {"n_items": n, "low": low, "high": high, "budget": B,
                       "n_pool": len(pool), "n_episodes": n_ep,
                       "p_useful_item": p_useful, "P_X_gt_K": pXK,
                       "EX_over_K": ratio, "act_over_cap": s2,
                       "mean_X": float(X.mean()), "mean_K": float(K.mean()),
                       "cheap": c.mean, "greedy": g.mean, "oracle": o.mean,
                       "ceiling": ceil, "S1": s1, "S2": s2 >= S2_RATIO_MIN}
                rows.append(row)
                run.append(row)
                if s1 or s2 >= S2_RATIO_MIN:
                    print(f"  {n:>3}{low:>6}{high:>6}{B:>7}{len(pool):>7}{n_ep:>5}"
                          f"{pXK:>8.2f}{ratio:>8.2f}{s2:>8.2f}"
                          f"{c.mean:>7.3f}{g.mean:>7.3f}{o.mean:>7.3f}"
                          f"{ceil:>+9.4f}  {gate}")

    both = [r for r in rows if r["S1"] and r["S2"]]
    print(f"\n  {len(rows)} configurations evaluated; "
          f"{sum(r['S1'] for r in rows)} satisfy S1, "
          f"{sum(r['S2'] for r in rows)} satisfy S2, "
          f"{len(both)} satisfy BOTH")

    chosen, verdict = None, "NO-VIABLE-CONFIGURATION"
    if both:
        # Selection among S1+S2 survivors is by the STRUCTURAL quantities only:
        # most competition first, then most decidable. The ceiling is not
        # consulted -- it is what the gate then tests.
        chosen = max(both, key=lambda r: (round(r["P_X_gt_K"], 2),
                                          round(r["act_over_cap"], 2),
                                          -r["budget"]))
        verdict = "CONFIG-FOUND"
        print(f"\n  SELECTED BY S1+S2 (ceiling not consulted):")
        print(f"    n_items={chosen['n_items']} LOW={chosen['low']} "
              f"HIGH={chosen['high']} budget={chosen['budget']}")
        print(f"    P(X>K)={chosen['P_X_gt_K']:.2f}  E[X]/K={chosen['EX_over_K']:.2f}"
              f"  actual/cap={chosen['act_over_cap']:.2f}"
              f"  (X={chosen['mean_X']:.2f}, K={chosen['mean_K']:.2f})")
        print(f"    ceiling AT THAT CONFIG: {chosen['ceiling']:+.4f}"
              f"  -> {'clears' if chosen['ceiling'] > CEILING_GATE else 'FAILS'} "
              f"the {CEILING_GATE} gate on calibration")
    else:
        best_s1 = max(rows, key=lambda r: r["P_X_gt_K"]) if rows else None
        best_s2 = max(rows, key=lambda r: r["act_over_cap"]) if rows else None
        print("\n  NO configuration satisfies both criteria.")
        if best_s1:
            print(f"    best P(X>K)={best_s1['P_X_gt_K']:.2f} at n={best_s1['n_items']} "
                  f"LOW={best_s1['low']} HIGH={best_s1['high']}")
        if best_s2:
            print(f"    best actual/cap={best_s2['act_over_cap']:.2f} at "
                  f"LOW={best_s2['low']} HIGH={best_s2['high']}")
        print("    -> this engine and family cannot pose an allocation problem "
              "under a hard reserved budget.")

    run.finalize(summary={"verdict": verdict, "n_configs": len(rows),
                          "n_satisfying_both": len(both), "chosen": chosen},
                 metrics={"configurations": rows},
                 traps={"secret_scan": secret_scan()}, verdict=verdict)
    print(f"\n  recorded: experiments/{a.exp}/")
    return 0 if chosen else 1


if __name__ == "__main__":
    raise SystemExit(main())
