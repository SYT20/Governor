#!/usr/bin/env python3
"""E0001 — Phase 4 reasoning curve, and engine/mode selection by frozen rule.

Measures correctness vs total-token budget for each candidate engine on a
calibration sample, then applies the rule in PREREGISTRATION-phase4-nemotron.md
WITHOUT LOOKING FIRST. The rule is code, not a paragraph, so the selection is
made by the same object that was committed before the data existed.

An engine QUALIFIES only if the curve contains a genuine transition region: a
budget where the model is materially worse than at saturation. Without one there
is no allocation problem to solve, and Phase 4 would be measuring nothing.

Usage:  python scripts/p4_curve.py [--engines nemotron,qwen] [--n 40]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.harness.ledger import ExperimentRun, ExperimentSpec  # noqa: E402
from governor.harness.traps import secret_scan  # noqa: E402
from governor.phase4.collect import (  # noqa: E402
    GROQ, OPENROUTER, RateLimited, ResponseCache, api_key, collect, outcome,
)
from governor.phase4.tasks import make_pool, pool_stats  # noqa: E402

# ---- FROZEN before any data was collected (see the preregistration) ----------
GRID = [300, 700, 1400, 2800]
CAL_POOL_SEED, CURVE_N = 1000, 40
SAT_TOL, MIN_GAP = 0.02, 0.15
PROMPT_CAP = 128

ENGINES = {
    "nemotron": dict(provider=OPENROUTER, model="nvidia/nemotron-nano-9b-v2:free",
                     cache="results/p4_cache_nemotron.sqlite", workers=4, tpm=None),
    "qwen":     dict(provider=GROQ, model="qwen/qwen3.6-27b",
                     cache="results/p4_cache_qwen.sqlite", workers=6, tpm=8000),
    "gptoss":   dict(provider=GROQ, model="openai/gpt-oss-120b",
                     cache="results/p4_cache_gptoss.sqlite", workers=6, tpm=8000),
}
# Tie-break fixed in advance: the directive names nemotron as the Phase 4
# engine, so if more than one engine qualifies, nemotron wins -- not the one
# with the biggest gap, which would be selecting the engine on the outcome.
PREFERENCE = ["nemotron", "qwen", "gptoss"]


def select_modes(acc: dict[int, float]) -> dict:
    """The frozen rule. HIGH = cheapest saturating budget; LOW = the dearest
    budget below it that is materially worse."""
    grid = sorted(acc)
    best = max(acc.values())
    high = next(b for b in grid if acc[b] >= best - SAT_TOL)
    lows = [b for b in grid if b < high and acc[b] <= acc[high] - MIN_GAP]
    low = max(lows) if lows else None
    return {"high": high, "low": low, "acc_high": acc[high],
            "acc_low": acc[low] if low is not None else None,
            "gap": (acc[high] - acc[low]) if low is not None else None,
            "qualifies": low is not None}


def episode_budget(low: int, high: int) -> int:
    """4*cap(LOW) + 2*(cap(HIGH) - cap(LOW)); cap(m) = PROMPT_CAP + m."""
    cap_lo, cap_hi = PROMPT_CAP + low, PROMPT_CAP + high
    return int(4 * cap_lo + 2 * (cap_hi - cap_lo))


def run_engine(name: str, pool, run, n: int) -> dict:
    cfg = ENGINES[name]
    cache = ResponseCache(Path(cfg["cache"]), model=cfg["model"],
                          provider=cfg["provider"])
    key = api_key(cfg["provider"])
    items = pool[:n]
    print(f"\n  {name} ({cfg['model']}) — {len(items)} items x {len(GRID)} budgets "
          f"= {len(items)*len(GRID)} calls, {len(cache.missing(items, GRID))} missing")
    try:
        stats = collect(cache, items, GRID, key, workers=cfg["workers"],
                        tpm=cfg["tpm"], progress_every=20)
    except RateLimited as e:
        print(f"    THROTTLED: {e}")
        return {"engine": name, "throttled": True, "error": str(e)}
    print(f"    fetched={stats['fetched']} errors={stats['errors']} "
          f"cached={stats['cached']}")

    acc, rows, over_cap = {}, [], 0
    for b in GRID:
        cs, tt, ans, starv, lat = [], [], [], [], []
        for it in items:
            try:
                o = outcome(cache, it, b)
            except KeyError:
                continue
            over_cap += int(o["prompt_tokens"] > PROMPT_CAP)
            cs.append(o["correct"]); tt.append(o["total_tokens"])
            ans.append(o["answered"]); starv.append(o["starved"]); lat.append(o["latency_s"])
            run.append({"engine": name, "model": cfg["model"], "budget": b,
                        "item_id": it.item_id, "n_ops": it.n_ops,
                        "framing": it.framing, "scale": it.scale, **o})
        if not cs:
            continue
        acc[b] = float(np.mean(cs))
        rows.append({"budget": b, "n": len(cs), "acc": acc[b],
                     "answered": float(np.mean(ans)),
                     "starved": float(np.mean(starv)),
                     "mean_total_tokens": float(np.mean(tt)),
                     "mean_latency_s": float(np.mean(lat))})
        print(f"      b={b:<5} acc={acc[b]:.3f}  answered={np.mean(ans):.3f}  "
              f"starved={np.mean(starv):.3f}  tokens={np.mean(tt):7.1f}  "
              f"{np.mean(lat):.2f}s")

    sel = select_modes(acc) if acc else {"qualifies": False}
    if sel.get("qualifies"):
        sel["episode_budget"] = episode_budget(sel["low"], sel["high"])
    print(f"    RULE -> {json.dumps(sel)}")
    if over_cap:
        print(f"    WARNING: {over_cap} prompts exceeded PROMPT_CAP={PROMPT_CAP}")
    return {"engine": name, "model": cfg["model"], "curve": rows,
            "selection": sel, "prompts_over_cap": over_cap, "throttled": False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="nemotron,qwen,gptoss")
    ap.add_argument("--n", type=int, default=CURVE_N)
    ap.add_argument("--exp", default="E0001")
    a = ap.parse_args()
    names = [x.strip() for x in a.engines.split(",") if x.strip()]

    pool = make_pool(seed=CAL_POOL_SEED, n=max(a.n, 400))
    print("=" * 78)
    print("E0001  PHASE 4 REASONING CURVE — engine and mode selection")
    print("=" * 78)
    print(f"  calibration pool seed {CAL_POOL_SEED}: {pool_stats(pool)}")
    print(f"  FROZEN: grid={GRID} sat_tol={SAT_TOL} min_gap={MIN_GAP} "
          f"prompt_cap={PROMPT_CAP}")

    spec = ExperimentSpec(
        exp_id=a.exp,
        title="Phase 4 reasoning curve and frozen engine/mode selection",
        model=",".join(ENGINES[n]["model"] for n in names),
        budget={"grid_max_tokens": GRID, "charged": "usage.total_tokens"},
        seeds={"calibration_pool": CAL_POOL_SEED},
        split={"calibration_items": a.n, "test": "not touched"},
        metric="mean exact-integer correctness over calibration items at each "
               "max_tokens budget; engine/mode chosen by the frozen rule in "
               "PREREGISTRATION-phase4-nemotron.md",
        params={"grid": GRID, "sat_tol": SAT_TOL, "min_gap": MIN_GAP,
                "prompt_cap": PROMPT_CAP, "preference": PREFERENCE,
                "engines": {n: ENGINES[n]["model"] for n in names}},
        notes="Every (item, budget) pair is called once and cached; all later "
              "policies read the same responses (common random numbers).")
    run = ExperimentRun(spec, overwrite=True)

    results = [run_engine(n, pool, run, a.n) for n in names]

    qualified = [r for r in results
                 if not r.get("throttled") and r.get("selection", {}).get("qualifies")]
    chosen = None
    for pref in PREFERENCE:
        m = [r for r in qualified if r["engine"] == pref]
        if m:
            chosen = m[0]
            break

    print("\n" + "-" * 78)
    for r in results:
        s = r.get("selection", {})
        print(f"  {r['engine']:<10} qualifies={s.get('qualifies')} "
              f"low={s.get('low')} high={s.get('high')} gap={s.get('gap')}")
    print(f"  CHOSEN ENGINE: {chosen['engine'] if chosen else 'NONE — Phase 4 premise fails'}")
    if chosen:
        s = chosen["selection"]
        print(f"    LOW={s['low']} (acc {s['acc_low']:.3f})  "
              f"HIGH={s['high']} (acc {s['acc_high']:.3f})  "
              f"EPISODE_BUDGET={s['episode_budget']} tokens")

    summary = {"chosen_engine": chosen["engine"] if chosen else None,
               "chosen_model": chosen["model"] if chosen else None,
               "selection": chosen["selection"] if chosen else None,
               "qualified": [r["engine"] for r in qualified]}
    traps = {"secret_scan": secret_scan()}
    run.finalize(summary=summary, metrics={"engines": results},
                 traps=traps,
                 verdict="PASS" if chosen else "PREMISE-FAILS")
    print(f"\n  recorded: experiments/{a.exp}/")
    return 0 if chosen else 1


if __name__ == "__main__":
    raise SystemExit(main())
