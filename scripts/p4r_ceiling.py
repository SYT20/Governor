#!/usr/bin/env python3
"""E0006 — Phase 4R ceiling GATE on the configuration S1+S2 selected.

    PASS requires  U(oracle) - U(greedy) > 0.02  with a 95% CI excluding 0.02,
    and the same result on episodes the configuration was not selected on.

THE INDEPENDENT UNIT IS THE ITEM. Episodes are groupings of a shared item pool,
so a CI over episodes counts the same item several times and is
anticonservative. Every interval here comes from a cluster bootstrap that
resamples ITEMS with replacement and re-forms episodes from the resample.

Two things are reported separately and must not be conflated:

  IN-SELECTION   the items S1/S2 were computed on. Informative, not a gate.
  HELD-OUT       items never used for selection. THIS is the gate.

If there are not enough uncollected items to form a held-out split, the script
says so and returns a non-zero code rather than promoting the in-selection
number. No API calls.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.harness.ledger import ExperimentRun, ExperimentSpec  # noqa: E402
from governor.harness.traps import secret_scan, split_leakage  # noqa: E402
from governor.phase4.collect import ResponseCache, outcome  # noqa: E402
from governor.phase4.config import CAL_POOL_SEED, ENGINES, PROMPT_CAP  # noqa: E402
from governor.phase4.env import DEEP, P4Env, make_episodes  # noqa: E402
from governor.phase4.evaluate import constant, execute  # noqa: E402
from governor.phase4.policies import all_cheap, clairvoyant, fixed_schedule, greedy  # noqa: E402
from governor.phase4.split import (  # noqa: E402
    SPLIT_FILE, filter_evaluation, filter_selection, freeze, verify_disjoint,
)
from governor.phase4.tasks import make_pool  # noqa: E402

CEILING_GATE = 0.02          # frozen project materiality threshold


def measure(cache, items, n_items, low, high, budget, group_seed):
    """One read of the ceiling on a given item set and grouping."""
    n_ep = len(items) // n_items
    eps = make_episodes(items, n_ep, group_seed, n_items=n_items)
    env = P4Env(cache, eps, low, high, float(budget), PROMPT_CAP)
    E = list(range(n_ep))
    c = execute(env, "c", constant(all_cheap(env)), E)
    g = execute(env, "g", constant(greedy(env)), E)
    o = execute(env, "o", lambda e: clairvoyant(env, e), E)
    scheds = [{i} for i in range(n_items)] + [set(range(n_items))]
    fixed = max(execute(env, "s", constant(fixed_schedule(env, s)), E).mean
                for s in scheds)
    K = float(np.mean([sum(m == DEEP for m in ms) for ms in g.modes]))
    return {"cheap": c.mean, "greedy": g.mean, "oracle": o.mean,
            "best_fixed": fixed, "ceiling": o.mean - g.mean,
            "ceiling_vs_fixed": o.mean - fixed, "greedy_deep": K,
            "n_episodes": n_ep}


def bootstrap(cache, items, n_items, low, high, budget, n_boot, seed=0):
    """Cluster bootstrap over ITEMS: resample, re-form episodes, re-measure."""
    rng = np.random.default_rng(seed)
    out = []
    for b in range(n_boot):
        idx = rng.integers(0, len(items), len(items))
        resampled = [items[int(i)] for i in idx]
        out.append(measure(cache, resampled, n_items, low, high, budget,
                           10_000 + b))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="qwen", choices=list(ENGINES))
    ap.add_argument("--n-items", type=int, default=6)
    ap.add_argument("--low", type=int, default=300)
    ap.add_argument("--high", type=int, default=700)
    ap.add_argument("--budget", type=int, default=2868)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--exp", default="E0006-ceiling-gate")
    a = ap.parse_args()

    cfg = ENGINES[a.engine]
    cache = ResponseCache(Path(cfg["cache"]), model=cfg["model"],
                          provider=cfg["provider"])
    split = freeze()
    ok_split, split_detail = verify_disjoint()
    pool = [i for i in make_pool(CAL_POOL_SEED, 400)
            if cache.get(i, a.low) and cache.get(i, a.high)]
    # The split comes from the FROZEN id list, not from a slice of whatever
    # happens to be cached: a slice moves as collection proceeds, so the
    # "held-out" set would silently change identity between runs.
    in_sel = filter_selection(pool)
    held = filter_evaluation(pool)

    print("=" * 84)
    print(f"E0006  PHASE 4R CEILING GATE — {cfg['model']}")
    print("=" * 84)
    print(f"  config: n_items={a.n_items} LOW={a.low} HIGH={a.high} "
          f"budget={a.budget}")
    print(f"  gate:   U(oracle) - U(greedy) > {CEILING_GATE}, 95% CI excluding "
          f"{CEILING_GATE}, on HELD-OUT items")
    print(f"  split:  {SPLIT_FILE} — {split_detail}")
    print(f"  items:  {len(pool)} cached  ->  {len(in_sel)} in-selection, "
          f"{len(held)} held out (by frozen id, not by slice)")
    if not ok_split:
        print("  split is not disjoint; refusing to run the gate.")
        return 2

    gains = np.array([outcome(cache, i, a.high)["correct"]
                      - outcome(cache, i, a.low)["correct"] for i in pool], float)
    act = float(np.mean([outcome(cache, i, a.high)["total_tokens"] for i in pool]))
    print(f"  S2 check: actual {act:.0f} / cap {PROMPT_CAP + a.high} = "
          f"{act / (PROMPT_CAP + a.high):.2f}")
    print(f"  items benefiting: {float((gains > 0).mean()):.1%}")

    results = {}
    for label, items in (("IN-SELECTION", in_sel), ("HELD-OUT", held)):
        if len(items) < a.n_items * 4:
            print(f"\n  {label}: only {len(items)} items — cannot form four "
                  f"episodes of {a.n_items}. NOT REPORTED.")
            results[label] = None
            continue
        point = measure(cache, items, a.n_items, a.low, a.high, a.budget, 7)
        boots = bootstrap(cache, items, a.n_items, a.low, a.high, a.budget, a.boot)
        cl = np.array([b["ceiling"] for b in boots])
        lo, hi = float(np.percentile(cl, 2.5)), float(np.percentile(cl, 97.5))
        passed = lo > CEILING_GATE
        results[label] = {**point, "boot_mean": float(cl.mean()),
                          "ci_lo": lo, "ci_hi": hi, "n_items": len(items),
                          "passes_gate": bool(passed)}
        print(f"\n  {label}  ({len(items)} items, {point['n_episodes']} episodes, "
              f"{a.boot} bootstrap resamples)")
        print(f"    all-cheap {point['cheap']:.4f} | best-fixed "
              f"{point['best_fixed']:.4f} | greedy {point['greedy']:.4f} | "
              f"oracle {point['oracle']:.4f}   (greedy deep {point['greedy_deep']:.2f}"
              f" of {a.n_items})")
        print(f"    CEILING oracle-greedy = {point['ceiling']:+.4f}   "
              f"bootstrap {cl.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
        print(f"    oracle - best fixed   = {point['ceiling_vs_fixed']:+.4f}")
        print(f"    gate (CI lower bound > {CEILING_GATE}): "
              f"{'PASS' if passed else 'FAIL'}")

    ho = results.get("HELD-OUT")
    if ho is None:
        verdict = "GATE-INCONCLUSIVE-NEED-ITEMS"
        need = a.n_items * 4 - len(held)
        print(f"\n  VERDICT: {verdict}")
        print(f"    The in-selection ceiling is informative but is NOT the gate. "
              f"{need} more collected items are required to run it.")
        print(f"    At ({a.low}+{PROMPT_CAP}) + ({a.high}+{PROMPT_CAP}) = "
              f"{(a.low + PROMPT_CAP) + (a.high + PROMPT_CAP)} reserved tokens "
              f"per item, Groq's 200k/day allows "
              f"{200000 // ((a.low + PROMPT_CAP) + (a.high + PROMPT_CAP))} items/day.")
    else:
        verdict = "CEILING-PASS" if ho["passes_gate"] else "CEILING-FAIL"
        print(f"\n  VERDICT: {verdict}")

    spec = ExperimentSpec(
        exp_id=a.exp,
        title="Phase 4R ceiling gate on the S1+S2-selected configuration",
        model=cfg["model"],
        budget={"episode_total_tokens": a.budget, "low": a.low, "high": a.high,
                "n_items": a.n_items, "charged": "usage.total_tokens"},
        seeds={"pool": CAL_POOL_SEED, "grouping": 7, "bootstrap": 0},
        split={"in_selection_items": len(in_sel), "held_out_items": len(held),
               "frozen_split_sha256": split["sha256"],
               "independent_unit": "ITEM (episodes are groupings of a shared "
                                   "pool, so an episode-level CI would be "
                                   "anticonservative)"},
        metric=f"U(clairvoyant optimum) - U(budget-limited greedy) through the "
               f"canonical executor; 95% CI from a cluster bootstrap over items; "
               f"PASS iff the CI lower bound exceeds {CEILING_GATE}",
        params={"ceiling_gate": CEILING_GATE, "n_bootstrap": a.boot},
        notes="No API calls. Configuration chosen by S1+S2 in E0005 before any "
              "ceiling was consulted.")
    run = ExperimentRun(spec, overwrite=True)
    for label, res in results.items():
        run.append({"split": label, **(res or {"skipped": True})})
    run.finalize(summary={"verdict": verdict, "config": {
                     "n_items": a.n_items, "low": a.low, "high": a.high,
                     "budget": a.budget},
                     "in_selection": results.get("IN-SELECTION"),
                     "held_out": ho},
                 metrics={"results": results},
                 traps={"secret_scan": secret_scan(),
                        "split_leakage": split_leakage(
                            [i.item_id for i in in_sel],
                            [i.item_id for i in held])},
                 verdict=verdict)
    print(f"\n  recorded: experiments/{a.exp}/")
    return 0 if verdict == "CEILING-PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
