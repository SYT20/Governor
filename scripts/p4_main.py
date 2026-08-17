#!/usr/bin/env python3
"""E0002 — the primary Phase 4 test.

    U(Governor + LLM)  >  U(best fixed LLM policy)   at equal total-token budget

Thin wrapper: collect the cache, call the pipeline, print, record. The logic
lives in `governor/phase4/pipeline.py` so it can be exercised against a
synthetic cache in the test suite rather than validated by the same run that
produces the result.

Usage:
    python scripts/p4_main.py --engine qwen --low 300 --high 1400 --budget 3912
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.harness.ledger import ExperimentRun, ExperimentSpec, file_commit  # noqa: E402
from governor.harness.traps import render  # noqa: E402
from governor.phase4.collect import (  # noqa: E402
    RateLimited, ResponseCache, api_key, collect,
)
from governor.phase4.config import (  # noqa: E402
    CAL_GROUP_SEED, CAL_POOL_SEED, ENGINES, PROMPT_CAP, TEST_GROUP_SEED,
    TEST_POOL_SEED,
)
from governor.phase4.env import DEEP, P4Env, make_episodes  # noqa: E402
from governor.phase4.pipeline import (  # noqa: E402
    PREREG, calibrate, evaluate_heldout, summarise,
)
from governor.phase4.tasks import FEATURE_NAMES, make_pool, pool_stats  # noqa: E402

COLS = ("U", "ci", "tok/ep", "reas", "ans", "deep", "util", "starv",
        "U/ktok", "lat_s")


def print_table(M: dict) -> None:
    print(f"    {'policy':<16}{'U':>8}{'95% CI':>19}{'tok/ep':>8}{'reas':>7}"
          f"{'ans':>6}{'deep':>6}{'util':>7}{'starv':>7}{'U/ktok':>8}{'lat_s':>7}")
    for k, m in M.items():
        print(f"    {k:<16}{m['U']:>8.4f}   [{m['ci_lo']:.4f},{m['ci_hi']:.4f}]"
              f"{m['total_tokens_per_episode']:>8.0f}"
              f"{m['reasoning_tokens_per_episode']:>7.0f}"
              f"{m['answer_tokens_per_episode']:>6.0f}"
              f"{m['deep_calls_per_episode']:>6.2f}"
              f"{m['budget_utilization']:>7.1%}{m['starvation_rate']:>7.1%}"
              f"{m['utility_per_ktoken']:>8.3f}"
              f"{m['latency_s_per_episode']:>7.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=list(ENGINES))
    ap.add_argument("--low", type=int, required=True)
    ap.add_argument("--high", type=int, required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--cal-items", type=int, default=300)
    ap.add_argument("--test-items", type=int, default=400)
    ap.add_argument("--exp", default="E0002")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()

    cfg = ENGINES[a.engine]
    cache = ResponseCache(Path(cfg["cache"]), model=cfg["model"],
                          provider=cfg["provider"])
    cal_pool = make_pool(CAL_POOL_SEED, a.cal_items)
    test_pool = make_pool(TEST_POOL_SEED, a.test_items)
    budgets = [a.low, a.high]

    print("=" * 84)
    print(f"E0002  PHASE 4 PRIMARY TEST — {cfg['model']}")
    print("=" * 84)
    print(f"  LOW={a.low} HIGH={a.high} EPISODE_BUDGET={a.budget:.0f} "
          f"(cap_low={PROMPT_CAP + a.low} cap_high={PROMPT_CAP + a.high})")
    print(f"  calibration pool {CAL_POOL_SEED}: {pool_stats(cal_pool)}")
    print(f"  test pool        {TEST_POOL_SEED}: {pool_stats(test_pool)}", flush=True)

    need = cache.missing(cal_pool + test_pool, budgets)
    print(f"\n  cache: {cache.count()} rows, {len(need)} of "
          f"{(a.cal_items + a.test_items) * 2} calls missing", flush=True)
    if need:
        try:
            st = collect(cache, cal_pool + test_pool, budgets,
                         api_key(cfg["provider"]), workers=cfg["workers"],
                         tpm=cfg["tpm"], progress_every=50)
            print(f"    {st}", flush=True)
        except RateLimited as e:
            print(f"  THROTTLED — cache kept, rerun to resume:\n    {e}")
            return 2
        still = cache.missing(cal_pool + test_pool, budgets)
        if still:
            print(f"  {len(still)} calls still missing; refusing to evaluate "
                  f"on a partial cache.")
            return 2
    if a.collect_only:
        return 0

    cal_eps = make_episodes(cal_pool, a.cal_items // 4, CAL_GROUP_SEED)
    test_eps = make_episodes(test_pool, a.test_items // 4, TEST_GROUP_SEED)
    cal_env = P4Env(cache, cal_eps, a.low, a.high, a.budget, PROMPT_CAP)
    test_env = P4Env(cache, test_eps, a.low, a.high, a.budget, PROMPT_CAP)
    C, T = list(range(len(cal_eps))), list(range(len(test_eps)))

    spec = ExperimentSpec(
        exp_id=a.exp,
        title=f"Phase 4 primary: Governor + {cfg['model']} vs best fixed policy",
        model=cfg["model"],
        budget={"episode_total_tokens": a.budget, "low": a.low, "high": a.high,
                "charged": "usage.total_tokens", "prompt_cap": PROMPT_CAP},
        seeds={"cal_pool": CAL_POOL_SEED, "test_pool": TEST_POOL_SEED,
               "cal_group": CAL_GROUP_SEED, "test_group": TEST_GROUP_SEED},
        split={"cal_items": a.cal_items, "cal_episodes": len(cal_eps),
               "test_items": a.test_items, "test_episodes": len(test_eps),
               "disjoint_pools": True},
        metric="mean fraction of the 4 items answered correctly per episode; "
               "paired difference against the calibration-frozen best fixed "
               "policy, 95% normal CI over episodes",
        params={"features": list(FEATURE_NAMES),
                "provider": cfg["provider"].name, "cache_rows": cache.count()},
        notes="All (item, budget) responses collected once and shared by every "
              "policy: common random numbers, as env6's roll array was.")
    run = ExperimentRun(spec, overwrite=True)

    print("\n  CALIBRATION (test split untouched)", flush=True)
    cal = calibrate(cal_env, cal_pool, C)
    g = cal.gains
    print(f"    best fixed schedule : {cal.best_schedule or '()'}  "
          f"U={cal.cal_utilities['fixed_best']:.4f}")
    print(f"    best text heuristic : {cal.heuristic_feature} >= "
          f"{cal.heuristic_threshold:.2f}  U={cal.heuristic_utility:.4f}")
    print(f"    realised gain       : mean={g.mean():+.4f} "
          f"pos={float((g > 0).mean()):.3f} zero={float((g == 0).mean()):.3f} "
          f"neg={float((g < 0).mean()):.3f}")
    r = cal.report
    print(f"    value predictor     : n={r.n} cv_R2={r.cv_r2:+.4f} "
          f"MAE={r.cv_mae:.4f} (mean-model {r.baseline_mae:.4f}) "
          f"spread={r.spread:.4f}")
    print(f"    DP thresholds       : {json.dumps(cal.dp.table())}")
    print(f"    BEST FIXED (frozen) : {cal.base} "
          f"{ {k: round(v, 4) for k, v in cal.cal_utilities.items()} }", flush=True)

    print(f"\n  HELD-OUT — pool seed {TEST_POOL_SEED}, {len(test_eps)} episodes")
    R, trace = evaluate_heldout(test_env, cal, T)
    s = summarise(R, cal, test_env, trace, commit=run.commit,
                  froze_commit=file_commit(PREREG))
    M = s["metrics"]
    print_table(M)

    print("\n    PAIRED DIFFERENCES (same episodes, same cached responses)")
    for b, d in s["deltas"].items():
        tag = ("  BEATS" if d["beats"] else
               "  LOSES" if d["loses"] else "  not separable")
        print(f"      GOVERNOR - {b:<15}{d['mean']:+.4f} "
              f"[{d['lo']:+.4f},{d['hi']:+.4f}]{tag}"
              f"{'  <-- PRIMARY' if b == cal.base else ''}")
    print(f"      oracle headroom captured: {s['headroom']:.0%}  "
          f"(oracle {M['oracle']['U']:.4f}, {cal.base} {M[cal.base]['U']:.4f})")
    spread = (max(m["total_tokens_per_episode"] for m in M.values())
              - min(m["total_tokens_per_episode"] for m in M.values()))
    print(f"      all policies capped at {a.budget:.0f} tok/ep; realised mean "
          f"spend spread {spread:.0f} tok")

    for i, e in enumerate(T):
        run.append({
            "episode": int(e),
            "items": [it.item_id for it in test_env.episodes[e]],
            "U": {k: float(R[k].U[i]) for k in R},
            "modes": {k: "".join("D" if m == DEEP else "c" for m in R[k].modes[i])
                      for k in R},
            "tokens": {k: float(R[k].spent[i]) for k in R},
            "governor_calls": R["GOVERNOR"].calls[i]})
    for rec in trace:
        run.append({"kind": "governor_decision", **rec})

    print("\n  TRAP CHECKS\n" + render(s["traps"]))
    print(f"\n  VERDICT: {'PASS' if s['passed'] else 'FAIL / BLOCKED'} — "
          f"GOVERNOR vs {cal.base}: {s['primary']['mean']:+.4f} "
          f"[{s['primary']['lo']:+.4f},{s['primary']['hi']:+.4f}]"
          + (f"; red traps {s['red']}" if s["red"] else ""))

    metrics = {
        "policies": M, "deltas": s["deltas"], "headroom_captured": s["headroom"],
        "predictor": {f: getattr(r, f) for f in
                      ("n", "mean_gain", "cv_r2", "cv_mae", "baseline_mae",
                       "spread")},
        "dp_thresholds": cal.dp.table(),
        "gain_distribution": {"mean": float(g.mean()), "pos": float((g > 0).mean()),
                              "zero": float((g == 0).mean()),
                              "neg": float((g < 0).mean())},
        "calibration": {"best_schedule": list(cal.best_schedule),
                        "schedule_utilities": cal.schedule_utilities,
                        "heuristic_feature": cal.heuristic_feature,
                        "heuristic_threshold": cal.heuristic_threshold,
                        "cal_utilities": cal.cal_utilities,
                        "best_fixed_name": cal.base}}
    run.finalize(summary={"primary_baseline": cal.base, **s["primary"],
                          "governor_U": M["GOVERNOR"]["U"],
                          "baseline_U": M[cal.base]["U"],
                          "oracle_U": M["oracle"]["U"],
                          "headroom_captured": s["headroom"]},
                 metrics=metrics, traps=s["traps"],
                 verdict="PASS" if s["passed"] else "FAIL",
                 allow_dirty=a.allow_dirty)
    print(f"\n  recorded: experiments/{a.exp}/")
    return 0 if s["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
