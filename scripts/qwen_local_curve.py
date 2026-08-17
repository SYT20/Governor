#!/usr/bin/env python3
"""E0009 — local Qwen reasoning curve. Backend experiment, not a Governor one.

Measures the same thing E0001 measured for the hosted engines, on the same
calibration items, so the two are directly comparable. No quota, no network.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.gate.qwen_local import DEFAULT_MODEL, QwenLocalM2  # noqa: E402
from governor.harness.ledger import ExperimentRun, ExperimentSpec  # noqa: E402
from governor.harness.traps import secret_scan  # noqa: E402
from governor.phase4.config import CAL_POOL_SEED, PROMPT_CAP, select_modes  # noqa: E402
from governor.phase4.split import filter_selection  # noqa: E402
from governor.phase4.tasks import SYSTEM_PROMPT, is_correct, make_pool  # noqa: E402

GRID = [300, 700, 1400, 2800]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--exp", default="E0009-qwen-local")
    a = ap.parse_args()

    items = filter_selection(make_pool(CAL_POOL_SEED, 400))[:a.n]
    m2 = QwenLocalM2(a.model, system_prompt=SYSTEM_PROMPT)
    print("=" * 78)
    print(f"E0009  LOCAL QWEN CURVE — {a.model}")
    print("=" * 78)
    t0 = time.time()
    m2.load()
    print(f"  model loaded in {time.time() - t0:.1f}s; {len(items)} items x "
          f"{len(GRID)} budgets", flush=True)

    spec = ExperimentSpec(
        exp_id=a.exp, title="Local Qwen backend reasoning curve (MLX)",
        model=a.model,
        budget={"grid_max_tokens": GRID, "charged": "completion tokens"},
        seeds={"pool": CAL_POOL_SEED},
        split={"selection_items": len(items)},
        metric="mean exact-integer correctness at each max_tokens budget, on the "
               "same calibration items as E0001",
        params={"runtime": "mlx", "grid": GRID},
        notes="Backend experiment. The Governor is unchanged.")
    run = ExperimentRun(spec, overwrite=True)

    acc, rows = {}, []
    for b in GRID:
        cs, tt, lat, starv = [], [], [], []
        for it in items:
            r = m2({"prompt": it.prompt}, b)
            ok = int(is_correct(it, r.result if isinstance(r.result, str) else ""))
            cs.append(ok); tt.append(r.total_tokens); lat.append(r.latency_s)
            starv.append(int("starved" in r.error))
            run.append({"budget": b, "item_id": it.item_id, "correct": ok,
                        "total_tokens": r.total_tokens,
                        "completion_tokens": r.cost_units,
                        "latency_s": round(r.latency_s, 3), "ok": r.ok,
                        "error": r.error, "n_ops": it.n_ops})
        acc[b] = float(np.mean(cs))
        rows.append({"budget": b, "acc": acc[b],
                     "starved": float(np.mean(starv)),
                     "mean_total_tokens": float(np.mean(tt)),
                     "mean_latency_s": float(np.mean(lat))})
        print(f"    b={b:<5} acc={acc[b]:.3f} starved={np.mean(starv):.3f} "
              f"tok={np.mean(tt):7.1f} {np.mean(lat):.2f}s/call", flush=True)

    sel = select_modes(acc)
    print(f"\n  frozen mode rule -> {sel}")
    verdict = "CURVE-VALID" if sel.get("qualifies") else "NO-TRANSITION"
    print(f"  VERDICT: {verdict}")
    run.finalize(summary={"selection": sel, "verdict": verdict,
                          "max_acc": max(acc.values())},
                 metrics={"curve": rows}, traps={"secret_scan": secret_scan()},
                 verdict=verdict)
    print(f"\n  recorded: experiments/{a.exp}/")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
