#!/usr/bin/env python3
"""E0004 — the observable ceiling for Phase 4, swept over every budget.

THE CHECK THAT SHOULD COME BEFORE BUILDING A CONTROLLER, AND DID NOT.

This project has a standing rule, learned in Environment 2: an omniscient oracle
proves nothing about solvability; measure what a PERFECT policy could do with
the information and the resources actually available. Phase 4 built the value
predictor, the dynamic program and seven policies before asking what the best
possible allocator would gain. This script asks.

    ceiling(B) = U(clairvoyant optimum at B) - U(budget-limited greedy at B)

If that is near zero at every budget, no allocator can earn its keep, and the
Governor's held-out result would have been a measurement of noise around zero
regardless of how good the controller was.

Costs no API calls.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="qwen", choices=list(ENGINES))
    ap.add_argument("--low", type=int, default=700)
    ap.add_argument("--high", type=int, default=2800)
    ap.add_argument("--exp", default="E0004-ceiling")
    a = ap.parse_args()

    cfg = ENGINES[a.engine]
    cache = ResponseCache(Path(cfg["cache"]), model=cfg["model"],
                          provider=cfg["provider"])
    have = [i for i in make_pool(CAL_POOL_SEED, 400)
            if cache.get(i, a.low) and cache.get(i, a.high)]
    n = (len(have) // 4) * 4
    items = have[:n]
    eps = make_episodes(items, n // 4, 7)
    E = list(range(n // 4))
    cap_lo, cap_hi = PROMPT_CAP + a.low, PROMPT_CAP + a.high

    gains = np.array([outcome(cache, i, a.high)["correct"]
                      - outcome(cache, i, a.low)["correct"] for i in items], float)
    act_lo = float(np.mean([outcome(cache, i, a.low)["total_tokens"] for i in items]))
    act_hi = float(np.mean([outcome(cache, i, a.high)["total_tokens"] for i in items]))

    print("=" * 80)
    print(f"E0004  OBSERVABLE CEILING — {cfg['model']}  LOW={a.low} HIGH={a.high}")
    print("=" * 80)
    print(f"  {n} items, {len(E)} episodes")
    print(f"  reservation vs reality: cap_low={cap_lo} actual={act_lo:.0f} "
          f"({act_lo/cap_lo:.0%}) | cap_high={cap_hi} actual={act_hi:.0f} "
          f"({act_hi/cap_hi:.0%})")
    print(f"  items that benefit from the deep budget: {float((gains > 0).mean()):.1%} "
          f"-> {4*float((gains > 0).mean()):.2f} of 4 per episode")

    print(f"\n  {'B':>6} {'cheap':>7}{'greedy':>8}{'oracle':>8} {'CEILING':>9}"
          f" {'g.deep':>7}{'o.deep':>7}")
    rows = []
    for B in range(4 * cap_lo, 4 * cap_lo + 4 * (cap_hi - cap_lo) + 400, 400):
        env = P4Env(cache, eps, a.low, a.high, float(B), PROMPT_CAP)
        c = execute(env, "c", constant(all_cheap(env)), E)
        g = execute(env, "g", constant(greedy(env)), E)
        o = execute(env, "o", lambda e: clairvoyant(env, e), E)
        row = {"budget": B, "cheap": c.mean, "greedy": g.mean, "oracle": o.mean,
               "ceiling": o.mean - g.mean,
               "greedy_deep": float(np.mean([sum(m == DEEP for m in ms)
                                             for ms in g.modes])),
               "oracle_deep": float(np.mean([sum(m == DEEP for m in ms)
                                             for ms in o.modes]))}
        rows.append(row)
        print(f"  {B:>6} {c.mean:>7.4f}{g.mean:>8.4f}{o.mean:>8.4f}"
              f" {row['ceiling']:>+9.4f} {row['greedy_deep']:>7.2f}"
              f"{row['oracle_deep']:>7.2f}")

    best = max(rows, key=lambda r: r["ceiling"])
    print(f"\n  MAXIMUM ceiling over every budget: {best['ceiling']:+.4f} "
          f"at B={best['budget']}")
    verdict = "PREMISE-FAILS" if best["ceiling"] < 0.05 else "HEADROOM-EXISTS"
    print(f"  VERDICT: {verdict}")
    if verdict == "PREMISE-FAILS":
        print("    No allocator can earn its keep on this task family at these\n"
              "    modes. A held-out Governor result would have been noise\n"
              "    around zero however good the controller was.")

    spec = ExperimentSpec(
        exp_id=a.exp,
        title="Phase 4 observable ceiling: what could ANY allocator gain?",
        model=cfg["model"],
        budget={"swept": [r["budget"] for r in rows], "low": a.low, "high": a.high},
        seeds={"pool": CAL_POOL_SEED, "grouping": 7},
        split={"items": n, "episodes": len(E), "note": "calibration pool only"},
        metric="U(clairvoyant optimum) - U(budget-limited greedy), per episode "
               "budget, both executed through the canonical executor",
        params={"cap_low": cap_lo, "cap_high": cap_hi,
                "actual_low": act_lo, "actual_high": act_hi,
                "frac_items_benefiting": float((gains > 0).mean())},
        notes="The check that should have preceded building the controller.")
    run = ExperimentRun(spec, overwrite=True)
    for r in rows:
        run.append(r)
    run.finalize(summary={"max_ceiling": best["ceiling"],
                          "at_budget": best["budget"],
                          "verdict": verdict,
                          "frac_benefiting": float((gains > 0).mean()),
                          "cap_to_actual_high": act_hi / cap_hi},
                 metrics={"sweep": rows},
                 traps={"secret_scan": secret_scan()}, verdict=verdict)
    print(f"\n  recorded: experiments/{a.exp}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
