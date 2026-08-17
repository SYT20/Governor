#!/usr/bin/env python3
"""E0008 — the Phase 4R Governor test. GATED.

    U(Governor) - U(best fixed policy)   at identical actual-token budget

`require_gate_passed()` is the FIRST call. If the held-out ceiling gate has not
recorded CEILING-PASS on untouched evaluation items, this script refuses to run
at all -- it will not train a controller for an environment that has not been
shown to contain anything to allocate.

Calibration uses the frozen SELECTION items; evaluation uses the frozen
EVALUATION items; both come from configs/phase4r_split.json by id.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.harness.ledger import ExperimentRun, ExperimentSpec, file_commit  # noqa: E402
from governor.harness.traps import render  # noqa: E402
from governor.phase4.collect import ResponseCache  # noqa: E402
from governor.phase4.config import CAL_POOL_SEED, ENGINES, PROMPT_CAP  # noqa: E402
from governor.phase4.env import DEEP, P4Env, make_episodes  # noqa: E402
from governor.phase4.gatekeeper import require_gate_passed  # noqa: E402
from governor.phase4.pipeline import (  # noqa: E402
    calibrate, evaluate_heldout, summarise,
)
from governor.phase4.split import filter_evaluation, filter_selection  # noqa: E402
from governor.phase4.tasks import FEATURE_NAMES, make_pool  # noqa: E402

N_ITEMS, LOW, HIGH, BUDGET = 6, 300, 700, 2868.0
PREREG = "PREREGISTRATION-phase4R-structure.md"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="qwen", choices=list(ENGINES))
    ap.add_argument("--exp", default="E0008-governor-phase4r")
    a = ap.parse_args()

    gate = require_gate_passed()          # refuses if the ceiling has not passed
    print("=" * 88)
    print("E0008  PHASE 4R GOVERNOR TEST")
    print("=" * 88)
    print(f"  gate: {gate['verdict']} on {gate['n_items']} held-out items, "
          f"CI lower bound {gate['ci_lo']:+.4f}")

    cfg = ENGINES[a.engine]
    cache = ResponseCache(Path(cfg["cache"]), model=cfg["model"],
                          provider=cfg["provider"])
    pool = [i for i in make_pool(CAL_POOL_SEED, 400)
            if cache.get(i, LOW) and cache.get(i, HIGH)]
    cal_items, test_items = filter_selection(pool), filter_evaluation(pool)
    n_cal, n_test = len(cal_items) // N_ITEMS, len(test_items) // N_ITEMS
    print(f"  config: n_items={N_ITEMS} LOW={LOW} HIGH={HIGH} budget={BUDGET:.0f}")
    print(f"  split:  {len(cal_items)} calibration items ({n_cal} episodes), "
          f"{len(test_items)} evaluation items ({n_test} episodes)")
    if n_test < 4:
        print("  too few evaluation episodes; refusing.")
        return 2

    cal_env = P4Env(cache, make_episodes(cal_items, n_cal, 11, n_items=N_ITEMS),
                    LOW, HIGH, BUDGET, PROMPT_CAP)
    test_env = P4Env(cache, make_episodes(test_items, n_test, 22, n_items=N_ITEMS),
                     LOW, HIGH, BUDGET, PROMPT_CAP)
    C, T = list(range(n_cal)), list(range(n_test))

    spec = ExperimentSpec(
        exp_id=a.exp,
        title="Phase 4R: Governor vs the best fixed policy at equal actual tokens",
        model=cfg["model"],
        budget={"episode_total_tokens": BUDGET, "low": LOW, "high": HIGH,
                "n_items": N_ITEMS, "charged": "usage.total_tokens",
                "prompt_cap": PROMPT_CAP},
        seeds={"pool": CAL_POOL_SEED, "cal_group": 11, "test_group": 22},
        split={"calibration_items": len(cal_items), "cal_episodes": n_cal,
               "evaluation_items": len(test_items), "test_episodes": n_test,
               "frozen_by": "configs/phase4r_split.json (item ids)"},
        metric="mean fraction correct per episode; paired difference vs the "
               "calibration-frozen best fixed policy, 95% normal CI over episodes",
        params={"features": list(FEATURE_NAMES), "gate": gate},
        notes="Runs only because the held-out ceiling gate passed.")
    run = ExperimentRun(spec, overwrite=True)

    print("\n  CALIBRATION (evaluation split untouched)")
    cal = calibrate(cal_env, cal_items, C)
    r = cal.report
    print(f"    predictor {cal.predictor_kind}: cv_R2={r.cv_r2:+.4f} "
          f"MAE={r.cv_mae:.4f} (mean-model {r.baseline_mae:.4f})")
    print(f"    best fixed schedule {cal.best_schedule or '()'}; base={cal.base}")
    print(f"    heuristic {cal.heuristic_feature} >= {cal.heuristic_threshold:.2f}")

    print(f"\n  HELD-OUT — {n_test} episodes from frozen evaluation ids")
    R, trace = evaluate_heldout(test_env, cal, T)
    s = summarise(R, cal, test_env, trace, commit=run.commit,
                  froze_commit=file_commit(PREREG),
                  selection_item_ids=[i.item_id for i in cal_items],
                  evaluation_item_ids=[i.item_id for i in test_items])
    M = s["metrics"]
    print(f"    {'policy':<16}{'U':>8}{'95% CI':>20}{'tok/ep':>8}{'reas':>7}"
          f"{'ans':>6}{'deep':>6}{'util':>7}{'starv':>7}{'U/ktok':>8}{'lat_s':>7}")
    for k, m in M.items():
        print(f"    {k:<16}{m['U']:>8.4f}   [{m['ci_lo']:+.4f},{m['ci_hi']:+.4f}]"
              f"{m['total_tokens_per_episode']:>8.0f}"
              f"{m['reasoning_tokens_per_episode']:>7.0f}"
              f"{m['answer_tokens_per_episode']:>6.0f}"
              f"{m['deep_calls_per_episode']:>6.2f}"
              f"{m['budget_utilization']:>7.1%}{m['starvation_rate']:>7.1%}"
              f"{m['utility_per_ktoken']:>8.3f}{m['latency_s_per_episode']:>7.2f}")

    print("\n    PAIRED DIFFERENCES")
    for b, d in s["deltas"].items():
        tag = ("BEATS" if d["beats"] else "LOSES" if d["loses"]
               else "not separable")
        print(f"      GOVERNOR - {b:<15}{d['mean']:+.4f} "
              f"[{d['lo']:+.4f},{d['hi']:+.4f}]  {tag}"
              f"{'  <-- PRIMARY' if b == cal.base else ''}")
    print(f"      oracle headroom captured: {s['headroom']:.0%}")
    spread = (max(m["total_tokens_per_episode"] for m in M.values())
              - min(m["total_tokens_per_episode"] for m in M.values()))
    print(f"      equal budget {BUDGET:.0f} tok/ep; realised spend spread {spread:.0f}")

    for i, e in enumerate(T):
        run.append({"episode": int(e),
                    "items": [it.item_id for it in test_env.episodes[e]],
                    "U": {k: float(R[k].U[i]) for k in R},
                    "modes": {k: "".join("D" if m == DEEP else "c"
                                         for m in R[k].modes[i]) for k in R},
                    "tokens": {k: float(R[k].spent[i]) for k in R},
                    "governor_calls": R["GOVERNOR"].calls[i]})
    for rec in trace:
        run.append({"kind": "governor_decision", **rec})

    print("\n  TRAP CHECKS\n" + render(s["traps"]))
    print(f"\n  VERDICT: {'PASS' if s['passed'] else 'FAIL'} — GOVERNOR vs "
          f"{cal.base}: {s['primary']['mean']:+.4f} "
          f"[{s['primary']['lo']:+.4f},{s['primary']['hi']:+.4f}]"
          + (f"; red traps {s['red']}" if s["red"] else ""))

    run.finalize(summary={"primary_baseline": cal.base, **s["primary"],
                          "governor_U": M["GOVERNOR"]["U"],
                          "baseline_U": M[cal.base]["U"],
                          "oracle_U": M["oracle"]["U"],
                          "headroom_captured": s["headroom"]},
                 metrics={"policies": M, "deltas": s["deltas"],
                          "predictor": {"kind": cal.predictor_kind,
                                        "cv_r2": r.cv_r2, "spread": r.spread},
                          "dp_thresholds": cal.dp.table()},
                 traps=s["traps"], verdict="PASS" if s["passed"] else "FAIL")
    print(f"\n  recorded: experiments/{a.exp}/")
    return 0 if s["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
