#!/usr/bin/env python3
"""E0023 — LiveCodeBench sample-allocation ceiling. New axis, zero quota.

MATH-500 was closed by E0022: settling the remaining effect needs ~26,000 items
and the benchmark has 500. The problem was structural -- most items allocate
identically, so mean/sd is hostile to detection.

This changes the ALLOCATION AXIS rather than the benchmark size. LiveCodeBench's
published submissions carry `graded_list`: pass/fail for TEN independent samples
per problem, 400 problems, with the raw generations for exact token counts. So
the unit is "how many samples does this problem get", the cost is genuinely
additive, and none of it needs an API call.

That axis is also the one the literature says has headroom where token budgets
do not (Damani 2410.04707 allocate best-of-k samples), and code is the case
where their online variant fell BELOW the uniform baseline -- because roughly
half of all problems are unsolvable at any budget, so a predictor that cannot
identify them burns the budget. That failure mode is measurable here.

Ceiling before controller, as always.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.harness.ledger import ExperimentRun, ExperimentSpec  # noqa: E402
from governor.harness.traps import exact_token_counts, secret_scan  # noqa: E402

EPS = 0.02
SRC = "livecodebench/submissions Gemini-Pro-1.5 (May) codegeneration_10_0.2"
TOKENS = "exact tokenizer count over the published generations"


def env_fixed(U, C, B):
    pts = sorted((float(C[:, j].mean()), float(U[:, j].mean()))
                 for j in range(C.shape[1]))
    best = None
    for (c0, u0), (c1, u1) in zip(pts, pts[1:]):
        if c0 <= B <= c1 and c1 > c0:
            w = (B - c0) / (c1 - c0)
            best = max(best if best is not None else -1.0, u0 + w * (u1 - u0))
    for c, u in pts:
        if c <= B + 1e-9:
            best = max(best if best is not None else -1.0, u)
    return float(best if best is not None else pts[0][1])


def oracle(U, C, B):
    b = -1.0
    for lam in np.concatenate([[0.0], np.geomspace(1e-7, 1e-1, 500)]):
        i = np.argmax(U - lam * C, axis=1)
        r = np.arange(len(i))
        if float(C[r, i].mean()) <= B + 1e-9:
            b = max(b, float(U[r, i].mean()))
    return b


def main() -> int:
    d = pickle.load(open("results/lcb_samples.pkl", "rb"))
    U, C, G = d["U"], d["C"], d["G"]
    n, K = U.shape
    npass = G.sum(axis=1)
    struct = {"never_pass": float((npass == 0).mean()),
              "always_pass": float((npass == K).mean()),
              "mixed": float(((npass > 0) & (npass < K)).mean())}
    print("=" * 78)
    print("E0023  LIVECODEBENCH SAMPLE-ALLOCATION CEILING")
    print("=" * 78)
    print(f"  {n} problems x {K} samples, {SRC}")
    print(f"  never pass {struct['never_pass']:.1%} | always pass "
          f"{struct['always_pass']:.1%} | MIXED {struct['mixed']:.1%}")
    rows = []
    for B in np.linspace(C[:, 0].mean() * 1.05, C[:, -1].mean() * 0.85, 12):
        f, o = env_fixed(U, C, B), oracle(U, C, B)
        rows.append({"budget_tokens": float(B), "best_fixed": f, "oracle": o,
                     "ceiling": o - f})
    best = max(rows, key=lambda r: r["ceiling"])
    print(f"\n  {'E[tokens]':>10}{'best fixed':>12}{'oracle':>9}{'CEILING':>10}")
    for r in rows[::2]:
        print(f"  {r['budget_tokens']:>10.0f}{r['best_fixed']:>12.4f}"
              f"{r['oracle']:>9.4f}{r['ceiling']:>+10.4f}")
    verdict = "CEILING-PASS" if best["ceiling"] >= EPS else "CEILING-FAIL"
    print(f"\n  MAX CEILING {best['ceiling']:+.4f} at "
          f"E[tokens]={best['budget_tokens']:.0f}  -> {verdict}")
    print("\n  The 54.5% that never pass are the interesting part: they are pure")
    print("  waste that a good allocator must identify and refuse. That is")
    print("  exactly where Damani et al. report an online allocator falling")
    print("  BELOW the uniform baseline on code, so this benchmark can produce")
    print("  a negative result rather than only a small positive one.")

    spec = ExperimentSpec(
        exp_id="E0023-lcb-ceiling",
        title="LiveCodeBench sample-allocation ceiling (no API calls)",
        model="Gemini-Pro-1.5 (May) generations, published by LiveCodeBench",
        budget={"axis": "number of samples k in 1..10",
                "contract": "SOFT_EXPECTED_BUDGET", "charged": TOKENS},
        seeds={"split": "not yet split - ceiling screen only"},
        split={"problems": int(n), "samples_per_problem": int(K)},
        metric="U(multiple-choice-knapsack oracle) - U(best fixed k, randomised "
               "between adjacent k to match the expected budget), at identical "
               "expected tokens",
        params={"epsilon": EPS, "structure": struct},
        notes="Allocation axis changed, benchmark not merely enlarged. MATH-500 "
              "was closed by E0022 as statistically unsettleable.")
    run = ExperimentRun(spec, overwrite=True)
    for r in rows:
        run.append(r)
    run.finalize(summary={"verdict": verdict, "max_ceiling": best["ceiling"],
                          "at_tokens": best["budget_tokens"],
                          "n_problems": int(n), **struct},
                 metrics={"sweep": rows, "structure": struct},
                 traps={"exact_token_counts": exact_token_counts(TOKENS),
                        "secret_scan": secret_scan()},
                 verdict=verdict)
    print(f"\n  recorded: experiments/E0023-lcb-ceiling/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
