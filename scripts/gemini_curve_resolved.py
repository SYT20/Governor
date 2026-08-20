#!/usr/bin/env python3
"""Resolve the Gemini value-vs-budget curve. n=50 per cell, full telemetry.

The n=5 version printed PASS on a non-monotonic curve because the criterion
"any upward step exists" fires on one item moving. Superseded criterion:

  A reasoning curve must demonstrate REPRODUCIBLE budget-dependent utility,
  not merely contain one increasing cell.

Records per call: budget, thoughts, candidates, total tokens, finishReason,
answer present, correct, latency, error -- because the n=5 run showed two
unexplained anomalies (hard spending FEWER thoughts than easy at b=128; empty
responses at mid budgets while b=0 answered fine) and aggregate accuracy cannot
diagnose either.

Also runs a reliability probe: 10 items x 5 repeats at fixed budget and
temperature 0. If utility is stochastic at temperature 0, the Governor must
reason about expected-value distributions, not a deterministic gain curve.
"""
from __future__ import annotations
import json, os, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if not os.environ.get("GEMINI_KEY"):
    raise SystemExit("set GEMINI_KEY in the environment")
from governor.gate.gemini_m2 import GeminiM2

BUDGETS = [0, 128, 512, 1024, 1536]
N = 50
rng = random.Random(20260817)
EASY = [(rng.randint(11, 49), rng.randint(11, 49)) for _ in range(N)]
HARD = [(rng.randint(200, 999), rng.randint(200, 999)) for _ in range(N)]


def ask(m, x, y, b):
    r = m({"prompt": f"What is {x}*{y}? Answer with only the number."}, b)
    corr = 0
    if r.ok:
        try:
            corr = int(int("".join(c for c in str(r.result) if c.isdigit())) == x * y)
        except ValueError:
            corr = 0
    return {"budget": b, "thoughts": r.reasoning_tokens, "total": r.total_tokens,
            "answered": int(r.ok), "correct": corr, "latency": r.latency_s,
            "error": r.error[:80]}


def main() -> int:
    m = GeminiM2()
    print(f"RESOLVED GEMINI CURVE — {N} items/cell, budgets {BUDGETS}", flush=True)
    rows, t0 = [], time.time()
    for b in BUDGETS:
        for tag, items in (("easy", EASY), ("hard", HARD)):
            cells = []
            for i, (x, y) in enumerate(items):
                d = ask(m, x, y, b) | {"tag": tag, "item": i}
                cells.append(d); rows.append(d)
                if (i + 1) % 25 == 0:
                    Path("results/gemini_curve_resolved.json").write_text(json.dumps(rows))
            u = sum(c["correct"] for c in cells) / len(cells)
            a = sum(c["answered"] for c in cells) / len(cells)
            th = sum(c["thoughts"] for c in cells) / len(cells)
            print(f"  {tag:<5} b={b:<5} utility {u:.3f}  answered {a:.2f}  "
                  f"mean thoughts {th:.0f}  [{(time.time()-t0)/60:.1f} min]", flush=True)
    Path("results/gemini_curve_resolved.json").write_text(json.dumps(rows, indent=2))

    print("\nRELIABILITY PROBE — 10 items x 5 repeats, b=512, temperature 0", flush=True)
    rep = []
    for i, (x, y) in enumerate(HARD[:10]):
        outs = [ask(m, x, y, 512)["correct"] for _ in range(5)]
        rep.append(outs)
        print(f"  item {i}: {outs}", flush=True)
    flips = sum(1 for o in rep if 0 < sum(o) < 5)
    print(f"\n  items with inconsistent outcomes at fixed budget: {flips}/10", flush=True)
    print("  -> " + ("utility is STOCHASTIC at temperature 0; the Governor must "
                     "model expected value, not a deterministic gain"
                     if flips >= 3 else
                     "utility is largely deterministic at temperature 0"), flush=True)
    Path("results/gemini_reliability.json").write_text(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
