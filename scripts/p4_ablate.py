#!/usr/bin/env python3
"""E0010 — Step 9 ablations: does each part of the Governor earn its place?

Three families, each holding everything else fixed:

  RULE       DP  vs  fixed-threshold  vs  q>0            (predictor fixed)
  PREDICTOR  gbt vs  ridge  vs  mean (null model)        (rule fixed)
  STATE      text vs +progress / +budget / +history / +uncertainty

The null cells are the point. If `mean` does as well as `gbt`, nothing was
learned from the text and the effect is the DP's alone. If `q>0` does as well as
the DP, the dynamic program is decoration -- which is exactly what happened in
Env 6's first attempt, where `q>0` fired everywhere and collapsed into greedy.

A component is KEPT only if held-out utility improves. Env 5 filed
`n_blocks_touched` as cognitive state and manufactured +0.035 that vanished when
it was reclassified; predicting "no effect" and confirming it is the stronger
result.

Costs no API calls: every response is already cached.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.harness.ledger import ExperimentRun, ExperimentSpec  # noqa: E402
from governor.harness.traps import (  # noqa: E402
    oracle_leakage, progress_as_cognition, secret_scan,
)
from governor.phase4.collect import ResponseCache  # noqa: E402
from governor.phase4.config import (  # noqa: E402
    CAL_GROUP_SEED, CAL_POOL_SEED, ENGINES, PROMPT_CAP, TEST_GROUP_SEED,
    TEST_POOL_SEED,
)
from governor.phase4.env import P4Env, make_episodes  # noqa: E402
from governor.phase4.evaluate import constant, execute, paired_ci  # noqa: E402
from governor.phase4.statemgr import (  # noqa: E402
    COMPONENTS, EnsemblePredictor, JUSTIFIED_PROGRESS, StatePredictor,
    collect_state_training, component_names,
)
from governor.phase4.pipeline import calibrate  # noqa: E402
from governor.phase4.policies import (  # noqa: E402
    all_cheap, fixed_schedule, governor, governor_state, greedy, myopic,
)
from governor.phase4.predictor import OpportunityCostDP  # noqa: E402
from governor.phase4.tasks import make_pool  # noqa: E402

STATE_CELLS = [("text",), ("text", "progress"), ("text", "budget"),
               ("text", "history"), ("text", "uncertainty"),
               ("text", "progress", "budget", "history", "uncertainty")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=list(ENGINES))
    ap.add_argument("--low", type=int, required=True)
    ap.add_argument("--high", type=int, required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--cal-items", type=int, default=300)
    ap.add_argument("--test-items", type=int, default=400)
    ap.add_argument("--exp", default="E0010")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()

    cfg = ENGINES[a.engine]
    cache = ResponseCache(Path(cfg["cache"]), model=cfg["model"],
                          provider=cfg["provider"])
    cal_pool = make_pool(CAL_POOL_SEED, a.cal_items)
    test_pool = make_pool(TEST_POOL_SEED, a.test_items)
    cal_env = P4Env(cache, make_episodes(cal_pool, a.cal_items // 4, CAL_GROUP_SEED),
                    a.low, a.high, a.budget, PROMPT_CAP)
    test_env = P4Env(cache, make_episodes(test_pool, a.test_items // 4, TEST_GROUP_SEED),
                     a.low, a.high, a.budget, PROMPT_CAP)
    C = list(range(len(cal_env.episodes)))
    T = list(range(len(test_env.episodes)))

    print("=" * 84)
    print(f"E0010  PHASE 4 ABLATIONS — {cfg['model']}")
    print("=" * 84)

    spec = ExperimentSpec(
        exp_id=a.exp, title="Phase 4 ablations: rule, predictor class, "
                            "cognitive-state components",
        model=cfg["model"],
        budget={"episode_total_tokens": a.budget, "low": a.low, "high": a.high},
        seeds={"cal_pool": CAL_POOL_SEED, "test_pool": TEST_POOL_SEED},
        split={"cal_episodes": len(C), "test_episodes": len(T)},
        metric="held-out mean utility per cell and paired difference against "
               "the primary Governor (text features + DP rule)",
        params={"state_cells": [list(c) for c in STATE_CELLS]},
        notes="No API calls: every response was already collected for E0002.")
    run = ExperimentRun(spec, overwrite=True)

    cal = calibrate(cal_env, cal_pool, C)
    ref = execute(test_env, "GOVERNOR", constant(governor(test_env, cal.predictor,
                                                          cal.dp)), T)
    base = execute(test_env, "base", constant(fixed_schedule(test_env,
                                                             set(cal.best_schedule))), T)
    print(f"  reference Governor U={ref.mean:.4f}   "
          f"{cal.base} U={base.mean:.4f}\n")

    rows = []

    def record(family: str, cell: str, res, extra: dict | None = None) -> None:
        d = paired_ci(res.U, ref.U)
        db = paired_ci(res.U, base.U)
        r = {"family": family, "cell": cell, "U": res.mean,
             "vs_governor": d, "vs_baseline": db, **(extra or {})}
        rows.append(r)
        run.append(r)
        print(f"  {family:<10}{cell:<34}U={res.mean:.4f}  "
              f"vs_gov {d['mean']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]  "
              f"vs_base {db['mean']:+.4f}"
              f"{'  WORSE THAN GOVERNOR' if d['loses'] else ''}"
              f"{'  BETTER THAN GOVERNOR' if d['beats'] else ''}")

    # ---- RULE ----------------------------------------------------------------
    print("  RULE (predictor fixed: gbt on text)")
    record("rule", "DP (primary)", ref)
    record("rule", "myopic q>0", execute(
        test_env, "m0", constant(myopic(test_env, cal.predictor, 0.0)), T))
    thr_grid = np.quantile(cal.predictor.q_samples, [0.3, 0.5, 0.6, 0.7, 0.8])
    best_thr = max(thr_grid, key=lambda t: execute(
        cal_env, "t", constant(myopic(cal_env, cal.predictor, float(t))), C).mean)
    record("rule", f"fixed threshold {best_thr:+.3f} (cal-tuned)", execute(
        test_env, "mt", constant(myopic(test_env, cal.predictor, float(best_thr))), T),
        {"threshold": float(best_thr)})
    record("rule", "no predictor: all-cheap", execute(
        test_env, "c", constant(all_cheap(test_env)), T))
    record("rule", "no predictor: greedy", execute(
        test_env, "g", constant(greedy(test_env)), T))

    # ---- PREDICTOR CLASS -----------------------------------------------------
    print("\n  PREDICTOR CLASS (rule fixed: DP)")
    for kind in ("gbt", "ridge", "mean"):
        c2 = calibrate(cal_env, cal_pool, C, predictor_kind=kind)
        res = execute(test_env, kind,
                      constant(governor(test_env, c2.predictor, c2.dp)), T)
        record("predictor", f"{kind} (cv_R2={c2.report.cv_r2:+.3f})", res,
               {"cv_r2": c2.report.cv_r2, "spread": c2.report.spread})

    # ---- COGNITIVE STATE -----------------------------------------------------
    print("\n  COGNITIVE STATE (rule fixed: DP, predictor gbt)")
    roll_policies = {"cheap": all_cheap(cal_env), "greedy": greedy(cal_env),
                     "sched": fixed_schedule(cal_env, set(cal.best_schedule))}
    ens = EnsemblePredictor().fit(cal_pool, cal.gains)
    for comps in STATE_CELLS:
        names = component_names(comps)
        ok_leak, det = oracle_leakage(names)
        ok_prog, det2 = progress_as_cognition(names, justified=JUSTIFIED_PROGRESS)
        if not (ok_leak and ok_prog):
            print(f"    SKIPPED {comps}: {det} {det2}")
            continue
        use_ens = ens if "uncertainty" in comps else None
        X, y = collect_state_training(cal_env, C, comps, roll_policies, use_ens)
        sp = StatePredictor(comps).fit(X, y)
        dp = OpportunityCostDP(sp.q_samples, n_items=4, max_k=4)
        res = execute(test_env, "+".join(comps),
                      constant(governor_state(test_env, sp, dp, comps, use_ens)), T)
        record("state", "+".join(comps) + f" (cv_R2={sp.cv_r2:+.3f})", res,
               {"components": list(comps), "cv_r2": sp.cv_r2,
                "n_features": len(names), "n_rows": int(len(y))})

    kept = [r for r in rows if r["family"] == "state" and r["vs_governor"]["beats"]]
    print(f"\n  COMPONENTS THAT EARN THEIR PLACE: "
          f"{[r['cell'] for r in kept] or 'none — text + DP is the whole model'}")

    traps = {"oracle_leakage": oracle_leakage(component_names(STATE_CELLS[-1])),
             "progress_as_cognition": progress_as_cognition(
                 component_names(STATE_CELLS[-1]), justified=JUSTIFIED_PROGRESS),
             "secret_scan": secret_scan()}
    run.finalize(summary={"reference_governor_U": ref.mean,
                          "baseline_U": base.mean,
                          "components_kept": [r["cell"] for r in kept],
                          "n_cells": len(rows)},
                 metrics={"cells": rows}, traps=traps, verdict="REPORTED",
                 allow_dirty=a.allow_dirty)
    print(f"\n  recorded: experiments/{a.exp}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
