#!/usr/bin/env python3
"""E0020-* — Step 7 robustness. One experiment directory per cell.

Three sweeps, all on already-cached responses:

  grouping   fresh episode groupings of the same test items
  budget     0.6x .. 1.6x the preregistered total-token budget
  mixture    test pools restricted by generative complexity

BOTH SIDES ADAPT. Each cell re-runs calibration under that cell's conditions, so
the Governor is compared against the best fixed policy FOR THAT BUDGET, not
against a baseline frozen at a different one. Comparing an adapted controller to
a stale baseline is how a robustness sweep manufactures a trend.

`n_ops` is used here to CONSTRUCT the mixture cells. That is an experimental
design choice about which items to test on, not an input to any policy -- the
Governor still sees only text.

Nothing is tuned after inspecting these results. A cell that fails is reported
as a cell that fails.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.harness.ledger import ExperimentRun, ExperimentSpec, file_commit  # noqa: E402
from governor.phase4.collect import ResponseCache  # noqa: E402
from governor.phase4.config import (  # noqa: E402
    CAL_GROUP_SEED, CAL_POOL_SEED, ENGINES, PROMPT_CAP, TEST_GROUP_SEED,
    TEST_POOL_SEED,
)
from governor.phase4.env import P4Env, make_episodes  # noqa: E402
from governor.phase4.pipeline import (  # noqa: E402
    PREREG, calibrate, evaluate_heldout, summarise,
)
from governor.phase4.tasks import make_pool  # noqa: E402

MIXTURES = {
    "all": lambda it: True,
    "simple": lambda it: it.n_ops <= 2,
    "complex": lambda it: it.n_ops >= 3,
    "large_numbers": lambda it: it.scale == 1,
}
BUDGET_MULTIPLIERS = (0.6, 0.8, 1.0, 1.25, 1.6)
GROUPING_SEEDS = (TEST_GROUP_SEED, 101, 202, 303)


def cell(name: str, engine_cfg, cal_pool, test_pool, low, high, budget,
         group_seed, exp_id, allow_dirty) -> dict:
    cache = ResponseCache(Path(engine_cfg["cache"]), model=engine_cfg["model"],
                          provider=engine_cfg["provider"])
    n_cal, n_test = (len(cal_pool) // 4) * 4, (len(test_pool) // 4) * 4
    if n_cal < 40 or n_test < 40:
        return {"cell": name, "skipped": f"too few items ({n_cal},{n_test})"}
    floor = 4 * (PROMPT_CAP + low)
    if budget < floor:
        return {"cell": name, "skipped": f"budget {budget:.0f} below the "
                                         f"all-cheap floor {floor}"}

    cal_env = P4Env(cache, make_episodes(cal_pool[:n_cal], n_cal // 4, CAL_GROUP_SEED),
                    low, high, budget, PROMPT_CAP)
    test_env = P4Env(cache, make_episodes(test_pool[:n_test], n_test // 4, group_seed),
                     low, high, budget, PROMPT_CAP)
    C, T = list(range(n_cal // 4)), list(range(n_test // 4))

    spec = ExperimentSpec(
        exp_id=exp_id, title=f"Phase 4 robustness cell: {name}",
        model=engine_cfg["model"],
        budget={"episode_total_tokens": budget, "low": low, "high": high},
        seeds={"cal_pool": CAL_POOL_SEED, "test_pool": TEST_POOL_SEED,
               "test_group": group_seed},
        split={"cal_episodes": len(C), "test_episodes": len(T)},
        metric="held-out mean utility; paired difference vs the best fixed "
               "policy re-frozen on calibration UNDER THIS CELL's conditions",
        params={"cell": name})
    run = ExperimentRun(spec, overwrite=True)

    cal = calibrate(cal_env, cal_pool[:n_cal], C)
    R, trace = evaluate_heldout(test_env, cal, T)
    s = summarise(R, cal, test_env, trace, commit=run.commit,
                  froze_commit=file_commit(PREREG))
    for i, e in enumerate(T):
        run.append({"episode": int(e),
                    "U": {k: float(R[k].U[i]) for k in R},
                    "tokens": {k: float(R[k].spent[i]) for k in R}})
    p = s["primary"]
    out = {"cell": name, "exp_id": exp_id, "base": cal.base,
           "budget": budget, "group_seed": group_seed,
           "n_test_episodes": len(T),
           "governor_U": s["metrics"]["GOVERNOR"]["U"],
           "baseline_U": s["metrics"][cal.base]["U"],
           "oracle_U": s["metrics"]["oracle"]["U"],
           "deep_calls": s["metrics"]["GOVERNOR"]["deep_calls_per_episode"],
           "utilization": s["metrics"]["GOVERNOR"]["budget_utilization"],
           "delta": p["mean"], "lo": p["lo"], "hi": p["hi"],
           "beats": p["beats"], "loses": p["loses"],
           "headroom": s["headroom"], "red": s["red"]}
    run.finalize(summary=out, metrics={"policies": s["metrics"],
                                       "deltas": s["deltas"]},
                 traps=s["traps"], verdict="PASS" if s["passed"] else "FAIL",
                 allow_dirty=allow_dirty)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=list(ENGINES))
    ap.add_argument("--low", type=int, required=True)
    ap.add_argument("--high", type=int, required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--cal-items", type=int, default=300)
    ap.add_argument("--test-items", type=int, default=400)
    ap.add_argument("--prefix", default="E0020")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()

    cfg = ENGINES[a.engine]
    cal_pool = make_pool(CAL_POOL_SEED, a.cal_items)
    test_pool = make_pool(TEST_POOL_SEED, a.test_items)

    print("=" * 92)
    print(f"E0020  PHASE 4 ROBUSTNESS — {cfg['model']}")
    print("=" * 92)
    rows = []

    def show(r):
        rows.append(r)
        if r.get("skipped"):
            print(f"  {r['cell']:<26} SKIPPED: {r['skipped']}")
            return
        v = ("BEATS" if r["beats"] else "LOSES" if r["loses"] else "  --  ")
        print(f"  {r['cell']:<26} n={r['n_test_episodes']:<4} base={r['base']:<12}"
              f" U={r['governor_U']:.4f} vs {r['baseline_U']:.4f}  "
              f"d={r['delta']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] {v}  "
              f"deep={r['deep_calls']:.2f} util={r['utilization']:.0%}"
              + (f"  RED {r['red']}" if r["red"] else ""))

    print("\n  GROUPING SEEDS (same items, fresh episode structure)")
    for g in GROUPING_SEEDS:
        show(cell(f"group_seed={g}", cfg, cal_pool, test_pool, a.low, a.high,
                  a.budget, g, f"{a.prefix}-grp{g}", a.allow_dirty))

    print("\n  TOTAL-TOKEN BUDGET")
    for m in BUDGET_MULTIPLIERS:
        show(cell(f"budget x{m}", cfg, cal_pool, test_pool, a.low, a.high,
                  a.budget * m, TEST_GROUP_SEED,
                  f"{a.prefix}-bud{str(m).replace('.', '')}", a.allow_dirty))

    print("\n  ITEM MIXTURE")
    for name, pred in MIXTURES.items():
        show(cell(f"mixture={name}", cfg, [i for i in cal_pool if pred(i)],
                  [i for i in test_pool if pred(i)], a.low, a.high, a.budget,
                  TEST_GROUP_SEED, f"{a.prefix}-mix{name}", a.allow_dirty))

    live = [r for r in rows if not r.get("skipped")]
    beats = sum(r["beats"] for r in live)
    loses = sum(r["loses"] for r in live)
    print(f"\n  {beats}/{len(live)} cells BEAT the re-frozen baseline, "
          f"{loses} LOSE, {len(live) - beats - loses} inconclusive")
    print(f"  mean delta across cells: "
          f"{np.mean([r['delta'] for r in live]):+.4f}")
    print(f"  worst cell: "
          f"{min(live, key=lambda r: r['delta'])['cell']} "
          f"{min(r['delta'] for r in live):+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
