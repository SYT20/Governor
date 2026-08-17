#!/usr/bin/env python3
"""E0010 — Phase 5 design search, screened by the headroom LAW. No API calls.

Screens (n_items, LOW, HIGH, budget) against the three frozen criteria in
PREREGISTRATION-phase5.md, all computable from cached data:

    S1 headroom      ceiling(n,k,p) >= 0.12          [derived, not measured]
    S2 decidability  mean actual / cap(DEEP) >= 0.70
    S3 stability     ceiling(n,k-1,p) >= 0.80*ceiling(n,k,p)
                     AND |k_selection - k_evaluation| <= 0.30

S3 is new and is the criterion Phase 4R lacked: its k drifted 1.50 -> 1.25
between splits, halving the ideal ceiling and failing the gate.

The measured ceiling is reported for every cell but is NEVER used to select --
selection is by S1/S2/S3 only, all three of which are properties of the
configuration rather than of the outcome. `k_evaluation` is computed from
evaluation items, which is legitimate: it is a resource-feasibility statistic of
the environment, carries no information about which items have gains, and S3
exists precisely to catch drift that only shows up across splits.
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
from governor.phase4.headroom import ceiling_binary, realisation_ratio  # noqa: E402
from governor.phase4.policies import all_cheap, clairvoyant, greedy  # noqa: E402
from governor.phase4.split import filter_evaluation, filter_selection, freeze  # noqa: E402
from governor.phase4.tasks import make_pool  # noqa: E402

S1_MIN, S2_MIN, S3_RATIO, S3_DRIFT = 0.12, 0.70, 0.80, 0.30
MODE_PAIRS = ((300, 700), (700, 1400), (300, 1400), (700, 2800))
N_ITEMS = (6, 8, 10, 12)


def realised_k(env, eps) -> float:
    r = execute(env, "g", constant(greedy(env)), eps)
    return float(np.mean([sum(m == DEEP for m in ms) for ms in r.modes]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="qwen", choices=list(ENGINES))
    ap.add_argument("--exp", default="E0010-phase5-design")
    a = ap.parse_args()
    cfg = ENGINES[a.engine]
    cache = ResponseCache(Path(cfg["cache"]), model=cfg["model"],
                          provider=cfg["provider"])
    split = freeze()
    pool = make_pool(CAL_POOL_SEED, 400)

    print("=" * 104)
    print(f"E0010  PHASE 5 DESIGN — screened by the headroom law (no API calls)")
    print("=" * 104)
    print(f"  FROZEN: S1 ceiling>={S1_MIN} | S2 act/cap>={S2_MIN} | "
          f"S3 drop<={1-S3_RATIO:.0%} and |dk|<={S3_DRIFT}")

    spec = ExperimentSpec(
        exp_id=a.exp,
        title="Phase 5 design: configurations screened by the derived headroom law",
        model=cfg["model"],
        budget={"swept": "per configuration"},
        seeds={"pool": CAL_POOL_SEED, "grouping": 7},
        split={"frozen_by": "configs/phase4r_split.json",
               "selection_items": len(split["selection_ids"])},
        metric="S1 = ceiling(n,k,p) from the closed form; S2 = actual/cap; "
               "S3 = ceiling(n,k-1)/ceiling(n,k) and cross-split k drift; "
               "measured ceiling = U(clairvoyant) - U(greedy) via the executor",
        params={"S1_min": S1_MIN, "S2_min": S2_MIN, "S3_ratio": S3_RATIO,
                "S3_drift": S3_DRIFT, "n_items": list(N_ITEMS),
                "mode_pairs": [list(m) for m in MODE_PAIRS]},
        notes="Selection by S1/S2/S3 only. The measured ceiling is reported for "
              "every cell and never used to select.")
    run = ExperimentRun(spec, overwrite=True)

    rows = []
    print(f"\n  {'n':>3}{'LOW':>6}{'HIGH':>6}{'B':>7}{'sel':>5}{'ev':>4}"
          f"{'p':>7}{'k_sel':>7}{'k_ev':>6}{'S1 ideal':>10}{'S2':>6}"
          f"{'S3 drop':>9}{'measured':>10}{'realised':>10}  gate")
    for low, high in MODE_PAIRS:
        have = [i for i in pool if cache.get(i, low) and cache.get(i, high)]
        sel, ev = filter_selection(have), filter_evaluation(have)
        if len(sel) < 24 or len(ev) < 24:
            print(f"  ({low},{high}): sel={len(sel)} ev={len(ev)} — too few, skipped")
            continue
        g = np.array([outcome(cache, i, high)["correct"]
                      - outcome(cache, i, low)["correct"] for i in sel], float)
        p = float((g > 0).mean())
        act = float(np.mean([outcome(cache, i, high)["total_tokens"] for i in sel]))
        cap_lo, cap_hi = PROMPT_CAP + low, PROMPT_CAP + high
        s2 = act / cap_hi

        for n in N_ITEMS:
            n_sel, n_ev = len(sel) // n, len(ev) // n
            if n_sel < 3 or n_ev < 3:
                continue
            for B in range(n * cap_lo, n * cap_lo + 6 * (cap_hi - cap_lo), 150):
                e_sel = P4Env(cache, make_episodes(sel, n_sel, 7, n_items=n),
                              low, high, float(B), PROMPT_CAP)
                k_sel = realised_k(e_sel, list(range(n_sel)))
                if k_sel < 0.5 or k_sel > n - 0.5:
                    continue
                e_ev = P4Env(cache, make_episodes(ev, n_ev, 8, n_items=n),
                             low, high, float(B), PROMPT_CAP)
                k_ev = realised_k(e_ev, list(range(n_ev)))
                k = int(round(k_sel))
                ideal = ceiling_binary(n, k, p)
                lower = ceiling_binary(n, max(k - 1, 0), p)
                drop = 1.0 - (lower / ideal if ideal > 1e-12 else 1.0)
                s1 = ideal >= S1_MIN
                s3 = (lower >= S3_RATIO * ideal) and abs(k_sel - k_ev) <= S3_DRIFT
                E = list(range(n_sel))
                gr = execute(e_sel, "g", constant(greedy(e_sel)), E)
                orc = execute(e_sel, "o", lambda e: clairvoyant(e_sel, e), E)
                meas = orc.mean - gr.mean
                row = {"n_items": n, "low": low, "high": high, "budget": B,
                       "n_sel_items": len(sel), "n_ev_items": len(ev),
                       "p": p, "k_sel": k_sel, "k_ev": k_ev, "k": k,
                       "ideal": ideal, "ideal_at_k_minus_1": lower,
                       "s3_drop": drop, "act_over_cap": s2,
                       "measured_selection": meas,
                       "realised": realisation_ratio(meas, ideal),
                       "S1": bool(s1), "S2": bool(s2 >= S2_MIN), "S3": bool(s3)}
                rows.append(row); run.append(row)
                if s1 and s2 >= S2_MIN:
                    tag = "S1+S2+S3" if s3 else "S1+S2"
                    print(f"  {n:>3}{low:>6}{high:>6}{B:>7}{len(sel):>5}{len(ev):>4}"
                          f"{p:>7.3f}{k_sel:>7.2f}{k_ev:>6.2f}{ideal:>+10.4f}"
                          f"{s2:>6.2f}{drop:>9.0%}{meas:>+10.4f}"
                          f"{row['realised']:>10.0%}  {tag}")

    ok = [r for r in rows if r["S1"] and r["S2"] and r["S3"]]
    print(f"\n  {len(rows)} configurations; {sum(r['S1'] for r in rows)} pass S1, "
          f"{sum(r['S2'] for r in rows)} pass S2, {sum(r['S3'] for r in rows)} pass S3, "
          f"{len(ok)} pass ALL THREE")

    chosen, verdict = None, "NO-VIABLE-CONFIGURATION"
    if ok:
        # Selection among survivors by STRUCTURE only: biggest ideal ceiling,
        # then least cross-split drift. The measured ceiling is not consulted.
        chosen = max(ok, key=lambda r: (round(r["ideal"], 3),
                                        -round(abs(r["k_sel"] - r["k_ev"]), 2)))
        verdict = "CONFIG-FOUND"
        print(f"\n  SELECTED BY S1+S2+S3 (measured ceiling not consulted):")
        print(f"    n_items={chosen['n_items']} LOW={chosen['low']} "
              f"HIGH={chosen['high']} budget={chosen['budget']}")
        print(f"    p={chosen['p']:.3f}  k={chosen['k']}  "
              f"ideal={chosen['ideal']:+.4f}  act/cap={chosen['act_over_cap']:.2f}")
        print(f"    stability: drop if k falls by one = {chosen['s3_drop']:.0%}; "
              f"cross-split k drift = {abs(chosen['k_sel']-chosen['k_ev']):.2f}")
        print(f"    measured on SELECTION: {chosen['measured_selection']:+.4f} "
              f"({chosen['realised']:.0%} of ideal)")
        print(f"    evaluation items available: {chosen['n_ev_items']}")
    else:
        print("\n  No configuration satisfies all three. Best by each:")
        for crit in ("ideal", "act_over_cap"):
            if rows:
                b = max(rows, key=lambda r: r[crit])
                print(f"    best {crit}: {b[crit]:.3f} at n={b['n_items']} "
                      f"({b['low']},{b['high']}) B={b['budget']}")

    run.finalize(summary={"verdict": verdict, "n_configs": len(rows),
                          "n_passing_all": len(ok), "chosen": chosen},
                 metrics={"configurations": rows},
                 traps={"secret_scan": secret_scan(),
                        "split_leakage": split_leakage(split["selection_ids"],
                                                       split["evaluation_ids"])},
                 verdict=verdict)
    print(f"\n  recorded: experiments/{a.exp}/")
    return 0 if chosen else 1


if __name__ == "__main__":
    raise SystemExit(main())
