#!/usr/bin/env python3
"""E0003 — PILOT. Underpowered by construction. NOT the preregistered test.

The preregistered E0002 needs 420 items and Groq's per-day token budget allows
about 54, so the real run needs roughly eight days of quota. This uses the items
that ARE collected to answer one question: is there any signal worth waiting
eight days for?

WHAT MAKES THIS A PILOT AND NOT THE TEST
  - the test items come from the CALIBRATION pool, split disjointly by index
    parity, not from the preregistered held-out pool (seed 20260817)
  - about 44 test items, versus 260 preregistered
  - it is recorded with verdict PILOT and its number must never be quoted as
    the Phase 4 result

WHY THE CI IS BOOTSTRAPPED OVER ITEMS. With 44 items only 11 disjoint episodes
exist, and a CI over 11 episodes is worthless. The unit of independent
information here is the ITEM, so the CI comes from a cluster bootstrap:
resample items with replacement, re-form episodes, re-run every policy, and take
percentiles of the resulting distribution of paired differences. Averaging over
many groupings of the SAME items and treating those as independent episodes
would understate the interval, which is the mistake this avoids.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.harness.ledger import ExperimentRun, ExperimentSpec, file_commit  # noqa: E402
from governor.harness.traps import render  # noqa: E402
from governor.phase4.collect import ResponseCache  # noqa: E402
from governor.phase4.config import (  # noqa: E402
    CAL_POOL_SEED, ENGINES, PROMPT_CAP, position_feasibility,
    position_neutral_floor,
)
from governor.phase4.env import DEEP, P4Env, make_episodes  # noqa: E402
from governor.phase4.pipeline import (  # noqa: E402
    BASELINES, PREREG, calibrate, evaluate_heldout, summarise,
)
from governor.phase4.tasks import make_pool  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="qwen", choices=list(ENGINES))
    ap.add_argument("--low", type=int, default=700)
    ap.add_argument("--high", type=int, default=2800)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--exp", default="E0003-pilot")
    a = ap.parse_args()

    cfg = ENGINES[a.engine]
    cache = ResponseCache(Path(cfg["cache"]), model=cfg["model"],
                          provider=cfg["provider"])
    have = [i for i in make_pool(CAL_POOL_SEED, 400)
            if cache.get(i, a.low) and cache.get(i, a.high)]
    budget = float(position_neutral_floor(a.low, a.high))

    # Disjoint split by index parity: deterministic, stated, and no item can
    # appear on both sides.
    cal_items = [it for j, it in enumerate(have) if j % 2 == 0]
    test_items = [it for j, it in enumerate(have) if j % 2 == 1]
    n_cal = (len(cal_items) // 4) * 4
    n_test = (len(test_items) // 4) * 4
    cal_items, test_items = cal_items[:n_cal], test_items[:n_test]

    print("=" * 84)
    print(f"E0003  PHASE 4 PILOT (UNDERPOWERED) — {cfg['model']}")
    print("=" * 84)
    print(f"  cached items {len(have)}  ->  cal {n_cal} / test {n_test} "
          f"(disjoint, index parity, both from pool seed {CAL_POOL_SEED})")
    print(f"  LOW={a.low} HIGH={a.high} BUDGET={budget:.0f} "
          f"(position-neutral floor)")
    if n_test < 8:
        print("  too few cached items to form even two test episodes.")
        return 2

    cal_env = P4Env(cache, make_episodes(cal_items, n_cal // 4, 11),
                    a.low, a.high, budget, PROMPT_CAP)
    C = list(range(n_cal // 4))
    cal = calibrate(cal_env, cal_items, C)
    print(f"\n  CALIBRATION ({n_cal} items, {len(C)} episodes)")
    print(f"    predictor {cal.predictor_kind}: cv_R2={cal.report.cv_r2:+.4f} "
          f"spread={cal.report.spread:.4f} (mean-model MAE "
          f"{cal.report.baseline_mae:.4f} vs {cal.report.cv_mae:.4f})")
    print(f"    gain: mean={cal.gains.mean():+.4f} "
          f"pos={float((cal.gains > 0).mean()):.3f}")
    print(f"    best schedule {cal.best_schedule or '()'}  base={cal.base}")
    print(f"    DP thresholds m4: " + " ".join(
        f"k{k}={cal.dp.threshold(4, k):.3f}" for k in (1, 2, 3, 4)))

    test_env = P4Env(cache, make_episodes(test_items, n_test // 4, 22),
                     a.low, a.high, budget, PROMPT_CAP)
    T = list(range(n_test // 4))
    pf = position_feasibility(lambda: test_env, T)
    print(f"\n  position feasibility on the test half: {pf}"
          f"{'  POSITION-NEUTRAL' if min(pf) >= 0.999 else '  BLOCKED'}")

    R, trace = evaluate_heldout(test_env, cal, T)
    s = summarise(R, cal, test_env, trace, commit="pilot",
                  froze_commit=file_commit(PREREG))
    M = s["metrics"]
    print(f"\n  POINT ESTIMATES ({n_test} items, {len(T)} episodes)")
    print(f"    {'policy':<16}{'U':>8}{'deep':>7}{'tok/ep':>9}{'util':>8}{'starv':>8}")
    for k, m in M.items():
        print(f"    {k:<16}{m['U']:>8.4f}{m['deep_calls_per_episode']:>7.2f}"
              f"{m['total_tokens_per_episode']:>9.0f}"
              f"{m['budget_utilization']:>8.1%}{m['starvation_rate']:>8.1%}")

    # ---- cluster bootstrap over ITEMS ---------------------------------------
    print(f"\n  CLUSTER BOOTSTRAP over {n_test} items, {a.boot} resamples")
    rng = np.random.default_rng(0)
    keys = ["GOVERNOR", *BASELINES, "oracle"]
    boots: dict[str, list[float]] = {k: [] for k in keys}
    diffs: dict[str, list[float]] = {b: [] for b in BASELINES}
    for b in range(a.boot):
        idx = rng.integers(0, n_test, n_test)
        items = [test_items[int(i)] for i in idx]
        env_b = P4Env(cache, make_episodes(items, n_test // 4, 1000 + b),
                      a.low, a.high, budget, PROMPT_CAP)
        Tb = list(range(n_test // 4))
        Rb, _ = evaluate_heldout(env_b, cal, Tb)
        for k in keys:
            boots[k].append(float(Rb[k].U.mean()))
        for base in BASELINES:
            diffs[base].append(float(Rb["GOVERNOR"].U.mean() - Rb[base].U.mean()))

    print(f"    {'policy':<16}{'U':>8}{'  95% bootstrap CI':>22}")
    for k in keys:
        v = np.array(boots[k])
        print(f"    {k:<16}{v.mean():>8.4f}   "
              f"[{np.percentile(v, 2.5):.4f}, {np.percentile(v, 97.5):.4f}]")

    print(f"\n    {'GOVERNOR minus':<20}{'delta':>9}{'  95% bootstrap CI':>22}")
    result = {}
    for base in BASELINES:
        d = np.array(diffs[base])
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        beats, loses = lo > 0, hi < 0
        result[base] = {"mean": float(d.mean()), "lo": lo, "hi": hi,
                        "beats": beats, "loses": loses}
        tag = "  BEATS" if beats else ("  LOSES" if loses else "  not separable")
        print(f"    {base:<20}{d.mean():>+9.4f}   [{lo:+.4f}, {hi:+.4f}]{tag}"
              f"{'  <-- primary comparison' if base == cal.base else ''}")

    d_or = np.array(boots["oracle"]) - np.array(boots[cal.base])
    print(f"\n    ceiling available to ANY allocator (oracle - {cal.base}): "
          f"{d_or.mean():+.4f} [{np.percentile(d_or, 2.5):+.4f}, "
          f"{np.percentile(d_or, 97.5):+.4f}]")
    print(f"    fraction of that ceiling captured: "
          f"{result[cal.base]['mean'] / d_or.mean():.0%}"
          if abs(d_or.mean()) > 1e-9 else "    ceiling is zero")

    print("\n  TRAP CHECKS\n" + render(s["traps"]))

    spec = ExperimentSpec(
        exp_id=a.exp,
        title="Phase 4 PILOT (underpowered): is there signal worth eight days "
              "of quota?",
        model=cfg["model"],
        budget={"episode_total_tokens": budget, "low": a.low, "high": a.high,
                "charged": "usage.total_tokens"},
        seeds={"pool": CAL_POOL_SEED, "split": "index parity", "bootstrap": 0},
        split={"cal_items": n_cal, "test_items": n_test,
               "disjoint": True,
               "WARNING": "test items come from the CALIBRATION pool; this is "
                          "NOT the preregistered held-out split"},
        metric="mean fraction correct per episode; paired difference vs the "
               "calibration-frozen best fixed policy, 95% CI from a cluster "
               "bootstrap over ITEMS (episodes are not independent here)",
        params={"n_bootstrap": a.boot, "position_feasibility": pf},
        notes="PILOT. Underpowered by construction and recorded as such. The "
              "preregistered E0002 needs 420 items; Groq allows about 54/day.")
    run = ExperimentRun(spec, overwrite=True)
    for i, e in enumerate(T):
        run.append({"episode": int(e),
                    "items": [it.item_id for it in test_env.episodes[e]],
                    "U": {k: float(R[k].U[i]) for k in R},
                    "modes": {k: "".join("D" if m == DEEP else "c"
                                         for m in R[k].modes[i]) for k in R},
                    "tokens": {k: float(R[k].spent[i]) for k in R}})
    for b in range(a.boot):
        run.append({"kind": "bootstrap", "draw": b,
                    **{f"U_{k}": boots[k][b] for k in keys}})
    run.finalize(
        summary={"primary_baseline": cal.base, **result[cal.base],
                 "governor_U": M["GOVERNOR"]["U"],
                 "baseline_U": M[cal.base]["U"], "oracle_U": M["oracle"]["U"],
                 "ceiling_available": float(d_or.mean()),
                 "n_test_items": n_test,
                 "WARNING": "PILOT: not the preregistered test, not a Phase 4 "
                            "result"},
        metrics={"policies": M, "bootstrap_deltas": result,
                 "predictor": {"kind": cal.predictor_kind,
                               "cv_r2": cal.report.cv_r2,
                               "spread": cal.report.spread},
                 "dp_thresholds": cal.dp.table()},
        traps=s["traps"], verdict="PILOT")
    print(f"\n  recorded: experiments/{a.exp}/  (verdict PILOT)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
