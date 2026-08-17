#!/usr/bin/env python3
"""E0012 — the Phase 5 Governor test. GATED on E0011 (CEILING-PASS).

    U(Governor) - U(best fixed policy)   at identical actual-token budget

STATISTICS. 55 evaluation items form only 4 episodes of 12, so a paired CI over
episodes would rest on four numbers. The independent unit is the ITEM: every
interval here comes from a cluster bootstrap that resamples items, re-forms
episodes, refits nothing, and re-runs every policy. The predictor and thresholds
are frozen from calibration and never refit inside the bootstrap -- refitting
would fold predictor variance into a comparison that is supposed to hold the
controller fixed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.harness.ledger import ExperimentRun, ExperimentSpec, file_commit  # noqa: E402
from governor.harness.traps import render, run_trap_checks, secret_scan  # noqa: E402
from governor.phase4.collect import ResponseCache, outcome  # noqa: E402
from governor.phase4.config import CAL_POOL_SEED, ENGINES, PROMPT_CAP  # noqa: E402
from governor.phase4.env import DEEP, P4Env, make_episodes  # noqa: E402
from governor.phase4.evaluate import constant, execute, token_evidence  # noqa: E402
from governor.phase4.gatekeeper import PHASE5_GATE, require_gate_passed  # noqa: E402
from governor.phase4.headroom import ceiling_binary, realisation_ratio  # noqa: E402
from governor.phase4.pipeline import calibrate  # noqa: E402
from governor.phase4.policies import (  # noqa: E402
    all_cheap, clairvoyant, fixed_schedule, governor, greedy, myopic,
    text_heuristic,
)
from governor.phase4.split import filter_evaluation, filter_selection  # noqa: E402
from governor.phase4.tasks import FEATURE_NAMES, make_pool  # noqa: E402

N, LOW, HIGH, BUDGET = 12, 300, 700, 6486.0
PREREG = "PREREGISTRATION-phase5.md"


def policies(env, cal):
    return {
        "H_all_cheap": constant(all_cheap(env)),
        "fixed_best": constant(fixed_schedule(env, set(cal.best_schedule))),
        "greedy": constant(greedy(env)),
        "heuristic": constant(text_heuristic(env, cal.heuristic_feature,
                                             cal.heuristic_threshold)),
        "myopic_q>0": constant(myopic(env, cal.predictor, 0.0)),
        "GOVERNOR": constant(governor(env, cal.predictor, cal.dp)),
        "oracle": lambda e: clairvoyant(env, e),
    }


def run_all(cache, items, cal, group_seed):
    n_ep = len(items) // N
    env = P4Env(cache, make_episodes(items, n_ep, group_seed, n_items=N),
                LOW, HIGH, BUDGET, PROMPT_CAP)
    E = list(range(n_ep))
    return env, {k: execute(env, k, f, E) for k, f in policies(env, cal).items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--exp", default="E0012-governor-phase5")
    a = ap.parse_args()

    gate = require_gate_passed(PHASE5_GATE)
    print("=" * 92)
    print("E0012  PHASE 5 GOVERNOR TEST")
    print("=" * 92)
    print(f"  gate {PHASE5_GATE}: {gate['verdict']}, held-out ceiling CI lower "
          f"bound {gate['ci_lo']:+.4f} on {gate['n_items']} items")

    cfg = ENGINES["qwen"]
    cache = ResponseCache(Path(cfg["cache"]), model=cfg["model"],
                          provider=cfg["provider"])
    pool = [i for i in make_pool(CAL_POOL_SEED, 400)
            if cache.get(i, LOW) and cache.get(i, HIGH)]
    cal_items, test_items = filter_selection(pool), filter_evaluation(pool)
    print(f"  config n={N} LOW={LOW} HIGH={HIGH} budget={BUDGET:.0f}")
    print(f"  {len(cal_items)} calibration items, {len(test_items)} evaluation items")

    # Calibration uses SELECTION items only. n=12 leaves few calibration
    # episodes, so the schedule/heuristic are chosen on 3 episodes while the
    # predictor is fit on all 40 items -- the predictor is per-item and does not
    # need episodes.
    n_cal_ep = len(cal_items) // N
    cal_env = P4Env(cache, make_episodes(cal_items, n_cal_ep, 11, n_items=N),
                    LOW, HIGH, BUDGET, PROMPT_CAP)
    cal = calibrate(cal_env, cal_items, list(range(n_cal_ep)))
    r = cal.report
    print(f"\n  CALIBRATION: predictor {cal.predictor_kind} cv_R2={r.cv_r2:+.4f} "
          f"(mean-model MAE {r.baseline_mae:.4f} vs {r.cv_mae:.4f})")
    print(f"    best fixed {cal.best_schedule or '()'} | base={cal.base} | "
          f"heuristic {cal.heuristic_feature}>={cal.heuristic_threshold:.2f}")
    print(f"    DP thresholds m12: " + " ".join(
        f"k{k}={cal.dp.threshold(N, k):.3f}" for k in (2, 4, 6, 8)))

    env, R = run_all(cache, test_items, cal, 22)
    M = {k: v.metrics(BUDGET) for k, v in R.items()}
    print(f"\n  HELD-OUT ({len(test_items)} items, {len(R['GOVERNOR'].U)} episodes)")
    print(f"    {'policy':<14}{'U':>8}{'tok/ep':>9}{'reas':>8}{'ans':>7}"
          f"{'deep':>7}{'util':>7}{'starv':>7}{'U/ktok':>8}")
    for k, m in M.items():
        print(f"    {k:<14}{m['U']:>8.4f}{m['total_tokens_per_episode']:>9.0f}"
              f"{m['reasoning_tokens_per_episode']:>8.0f}"
              f"{m['answer_tokens_per_episode']:>7.0f}"
              f"{m['deep_calls_per_episode']:>7.2f}"
              f"{m['budget_utilization']:>7.1%}{m['starvation_rate']:>7.1%}"
              f"{m['utility_per_ktoken']:>8.3f}")

    # ---- cluster bootstrap over ITEMS, controller frozen --------------------
    print(f"\n  CLUSTER BOOTSTRAP over {len(test_items)} items, {a.boot} resamples")
    rng = np.random.default_rng(0)
    keys = list(R)
    boots = {k: [] for k in keys}
    for b in range(a.boot):
        idx = rng.integers(0, len(test_items), len(test_items))
        _, Rb = run_all(cache, [test_items[int(i)] for i in idx], cal, 5000 + b)
        for k in keys:
            boots[k].append(float(Rb[k].U.mean()))
    B = {k: np.array(v) for k, v in boots.items()}

    print(f"    {'policy':<14}{'U':>8}{'  95% bootstrap CI':>22}")
    for k in keys:
        print(f"    {k:<14}{B[k].mean():>8.4f}   "
              f"[{np.percentile(B[k],2.5):.4f}, {np.percentile(B[k],97.5):.4f}]")

    print(f"\n    {'GOVERNOR minus':<18}{'delta':>9}{'  95% bootstrap CI':>22}")
    deltas = {}
    for base in ("H_all_cheap", "fixed_best", "greedy", "heuristic", "myopic_q>0"):
        d = B["GOVERNOR"] - B[base]
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        deltas[base] = {"mean": float(d.mean()), "lo": lo, "hi": hi,
                        "beats": bool(lo > 0), "loses": bool(hi < 0)}
        tag = "BEATS" if lo > 0 else ("LOSES" if hi < 0 else "not separable")
        print(f"    {base:<18}{d.mean():>+9.4f}   [{lo:+.4f}, {hi:+.4f}]  {tag}"
              f"{'  <-- PRIMARY' if base == cal.base else ''}")

    ceil = B["oracle"] - B[cal.base]
    head = float((B["GOVERNOR"] - B[cal.base]).mean() / ceil.mean()) if ceil.mean() else float("nan")
    p = float(np.mean([outcome(cache, i, HIGH)["correct"]
                       - outcome(cache, i, LOW)["correct"] > 0 for i in test_items]))
    k_real = M["greedy"]["deep_calls_per_episode"]
    ideal = ceiling_binary(N, int(round(k_real)), p)
    print(f"\n    ceiling available (oracle - {cal.base}): {ceil.mean():+.4f} "
          f"[{np.percentile(ceil,2.5):+.4f}, {np.percentile(ceil,97.5):+.4f}]")
    print(f"    headroom captured by the Governor: {head:.0%}")
    print(f"    law check: p={p:.3f} k={k_real:.2f} ideal={ideal:+.4f} "
          f"realised={realisation_ratio(float(ceil.mean()), ideal):.0%}")
    spread = (max(m["total_tokens_per_episode"] for m in M.values())
              - min(m["total_tokens_per_episode"] for m in M.values()))
    print(f"    equal budget {BUDGET:.0f} tok/ep; realised spend spread {spread:.0f}")

    ev = {"gov_utils": R["GOVERNOR"].U, "greedy_utils": R["greedy"].U,
          "gov_calls": [sum(m == DEEP for m in ms) for ms in R["GOVERNOR"].modes],
          "greedy_calls": [sum(m == DEEP for m in ms) for ms in R["greedy"].modes],
          "decisions_by_state": R["GOVERNOR"].modes,
          "feature_names": list(FEATURE_NAMES),
          "answered_rate": M["GOVERNOR"]["answered_rate"],
          "utility": M["GOVERNOR"]["U"], "scored_via_executor": True,
          "decisions": [m for ms in R["GOVERNOR"].modes for m in ms],
          "cell_ids": [f"t{t}" for ms in R["GOVERNOR"].modes
                       for t in range(len(ms))],
          "froze_commit": file_commit(PREREG), "heldout_commit": "run",
          "selection_item_ids": [i.item_id for i in cal_items],
          "evaluation_item_ids": [i.item_id for i in test_items],
          **token_evidence(R["GOVERNOR"], env)}
    traps = run_trap_checks(ev); traps["secret_scan"] = secret_scan()
    print("\n  TRAP CHECKS\n" + render(traps))
    red = [n for n, (ok, _) in traps.items() if not ok]
    primary = deltas[cal.base]
    passed = primary["beats"] and not red
    print(f"\n  VERDICT: {'PASS' if passed else 'FAIL'} — GOVERNOR vs {cal.base}: "
          f"{primary['mean']:+.4f} [{primary['lo']:+.4f}, {primary['hi']:+.4f}]"
          + (f"; red {red}" if red else ""))

    spec = ExperimentSpec(
        exp_id=a.exp, title="Phase 5: Governor vs best fixed policy at equal tokens",
        model=cfg["model"],
        budget={"episode_total_tokens": BUDGET, "low": LOW, "high": HIGH,
                "n_items": N, "charged": "usage.total_tokens"},
        seeds={"pool": CAL_POOL_SEED, "cal_group": 11, "test_group": 22,
               "bootstrap": 0},
        split={"calibration_items": len(cal_items),
               "evaluation_items": len(test_items),
               "frozen_by": "configs/phase4r_split.json",
               "independent_unit": "ITEM (4 episodes only; an episode-level CI "
                                   "would rest on four numbers)"},
        metric="mean fraction correct per episode; paired difference vs the "
               "calibration-frozen best fixed policy; 95% CI from a cluster "
               "bootstrap over items with the controller held fixed",
        params={"gate": gate, "n_bootstrap": a.boot,
                "features": list(FEATURE_NAMES)},
        notes="Runs only because E0011 recorded CEILING-PASS on held-out items.")
    run = ExperimentRun(spec, overwrite=True)
    for i in range(len(R["GOVERNOR"].U)):
        run.append({"episode": i,
                    "U": {k: float(R[k].U[i]) for k in R},
                    "modes": {k: "".join("D" if m == DEEP else "c"
                                         for m in R[k].modes[i]) for k in R},
                    "tokens": {k: float(R[k].spent[i]) for k in R}})
    for b in range(a.boot):
        run.append({"kind": "bootstrap", "draw": b,
                    **{f"U_{k}": float(B[k][b]) for k in keys}})
    run.finalize(summary={"primary_baseline": cal.base, **primary,
                          "governor_U": float(B["GOVERNOR"].mean()),
                          "baseline_U": float(B[cal.base].mean()),
                          "oracle_U": float(B["oracle"].mean()),
                          "headroom_captured": head,
                          "ceiling_available": float(ceil.mean())},
                 metrics={"policies": M, "deltas": deltas,
                          "predictor": {"kind": cal.predictor_kind,
                                        "cv_r2": r.cv_r2, "spread": r.spread},
                          "law": {"p": p, "k": k_real, "ideal": ideal}},
                 traps=traps, verdict="PASS" if passed else "FAIL")
    print(f"\n  recorded: experiments/{a.exp}/")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
