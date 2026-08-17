#!/usr/bin/env python3
"""E0016 — the SOFT_EXPECTED_BUDGET ceiling. Cheap, and it runs before any
controller is built.

CONTRACT (new, and deliberately different from the two already eliminated):

    constraint      E[ sum of ACTUAL generation tokens ] <= B
    NOT             sum of reserved worst-case caps <= B

Under hard reservation a policy must set aside what a call COULD cost; here it
is charged what calls DO cost, on average. That removes the exact defect that
eliminated the token-cap contract (E0013): the model self-terminates on 98%+ of
items, so reservation overstated consumption 2-11x.

THE BASELINE IS THE STRONG ONE. A fixed policy is not restricted to the seven
discrete budget levels: it may RANDOMISE between two adjacent levels to hit the
expected budget exactly. That is the upper concave envelope of the fixed points,
and it is strictly stronger than "pick the best single level". Comparing an
adaptive policy against a handicapped baseline is how this project has
manufactured effects before.

THE ORACLE is the exact solution to the multiple-choice knapsack: choose a
budget level per item to maximise total correctness subject to the token
constraint. Solved by a Lagrangian sweep, which traces the concave envelope of
achievable (cost, utility) pairs -- the true upper bound for ANY allocator,
clairvoyant or not.

    ceiling(B) = U_oracle(B) - U_bestfixed(B)

If that is below the frozen materiality threshold at every B, there is no
Governor problem under this contract and the project stops redesigning resource
semantics.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.harness.ledger import ExperimentRun, ExperimentSpec  # noqa: E402
from governor.harness.traps import exact_token_counts, secret_scan  # noqa: E402

EPSILON = 0.02          # the project's frozen materiality threshold
CACHE = Path("results/s1_caps_exact.pkl")
TOKEN_SOURCE = "simplescaling/s1-32B tokenizer (exact)"


def envelope(points):
    """Upper concave envelope of (cost, utility) points, as a lookup function."""
    pts = sorted(points)
    hull = []
    for c, u in pts:
        while hull and u >= hull[-1][1]:
            hull.pop()
        while len(hull) >= 2:
            (c0, u0), (c1, u1) = hull[-2], hull[-1]
            if (u1 - u0) * (c - c0) <= (u - u0) * (c1 - c0):
                hull.pop()
            else:
                break
        hull.append((c, u))

    def f(B):
        if B <= hull[0][0]:
            return hull[0][1] if B >= hull[0][0] else float("nan")
        for (c0, u0), (c1, u1) in zip(hull, hull[1:]):
            if c0 <= B <= c1:
                w = (B - c0) / (c1 - c0) if c1 > c0 else 0.0
                return u0 + w * (u1 - u0)
        return hull[-1][1]
    return f, hull


def oracle_envelope(data, ids, levels):
    """Multiple-choice knapsack by Lagrangian sweep.

    For each lambda every item independently picks the level maximising
    (correct - lambda*tokens). Sweeping lambda traces the concave envelope of
    achievable (mean tokens, mean correct) pairs, which is the exact LP optimum
    and an upper bound on any integral allocation.
    """
    C = np.array([[data[b][i]["correct"] for b in levels] for i in ids], float)
    T = np.array([[data[b][i]["tokens"] for b in levels] for i in ids], float)
    pts = set()
    for lam in np.concatenate([[0.0], np.geomspace(1e-7, 1e-1, 400)]):
        pick = np.argmax(C - lam * T, axis=1)
        pts.add((float(T[np.arange(len(ids)), pick].mean()),
                 float(C[np.arange(len(ids)), pick].mean())))
    return envelope(pts)


def main() -> int:
    all_data = pickle.load(open(CACHE, "rb"))
    print("=" * 86)
    print("E0016  SOFT EXPECTED-BUDGET CEILING — E[sum actual tokens] <= B")
    print("=" * 86)
    print(f"  token source: {TOKEN_SOURCE}; materiality epsilon = {EPSILON}")

    results = {}
    for bench, data in all_data.items():
        levels = sorted(data)
        ids = sorted(data[levels[0]])
        fixed_pts = [(float(np.mean([data[b][i]["tokens"] for i in ids])),
                      float(np.mean([data[b][i]["correct"] for i in ids])))
                     for b in levels]
        f_fixed, hull_fixed = envelope(fixed_pts)
        f_orac, hull_orac = oracle_envelope(data, ids, levels)

        lo = min(c for c, _ in fixed_pts)
        hi = max(c for c, _ in fixed_pts)
        rows = []
        for B in np.linspace(lo, hi, 25):
            uf, uo = f_fixed(B), f_orac(B)
            rows.append({"budget_tokens": float(B), "best_fixed": float(uf),
                         "oracle": float(uo), "ceiling": float(uo - uf)})
        best = max(rows, key=lambda r: r["ceiling"])
        results[bench] = {"rows": rows, "best": best, "n_items": len(ids),
                          "fixed_hull": hull_fixed, "oracle_hull": hull_orac}

        print(f"\n  {bench.upper()} ({len(ids)} items)")
        print(f"    {'E[tokens]':>10}{'best fixed':>12}{'oracle':>9}{'CEILING':>10}")
        for r in rows[::3]:
            print(f"    {r['budget_tokens']:>10.0f}{r['best_fixed']:>12.4f}"
                  f"{r['oracle']:>9.4f}{r['ceiling']:>+10.4f}")
        print(f"    MAX ceiling {best['ceiling']:+.4f} at E[tokens]="
              f"{best['budget_tokens']:.0f}  -> "
              f"{'PASS' if best['ceiling'] >= EPSILON else 'FAIL'} (eps {EPSILON})")

    passing = [b for b, r in results.items() if r["best"]["ceiling"] >= EPSILON]
    verdict = "SOFT-BUDGET-HEADROOM" if passing else "SOFT-BUDGET-NO-HEADROOM"
    print(f"\n  VERDICT: {verdict}")
    if passing:
        print(f"    Material headroom under the soft contract on: {passing}.")
        print(f"    A learned Governor is now justified to build and test.")
    else:
        print("    No material headroom under ANY of the three resource")
        print("    contracts tested. Stop redesigning resource semantics.")

    spec = ExperimentSpec(
        exp_id="E0016-soft-budget-ceiling",
        title="Soft expected-budget ceiling on external s1 data",
        model="s1-32B via simplescaling/results",
        budget={"contract": "SOFT_EXPECTED_BUDGET",
                "definition": "E[sum actual generation tokens] <= B",
                "levels": sorted(all_data["math"]),
                "charged": TOKEN_SOURCE},
        seeds={"split": "n/a - ceiling screen over all items"},
        split={b: r["n_items"] for b, r in results.items()},
        metric="U(multiple-choice-knapsack oracle) - U(best fixed policy, "
               "randomised between adjacent levels to match the expected budget "
               "exactly), both at identical E[tokens]",
        params={"epsilon": EPSILON,
                "baseline": "upper concave envelope of fixed levels",
                "oracle": "Lagrangian sweep over the multiple-choice knapsack"},
        notes="No API calls. Runs BEFORE any controller, per the project rule "
              "that the ceiling is measured first.")
    run = ExperimentRun(spec, overwrite=True)
    for bench, r in results.items():
        for row in r["rows"]:
            run.append({"benchmark": bench, **row})
    run.finalize(
        summary={"verdict": verdict,
                 "benchmarks_with_headroom": passing,
                 **{f"max_ceiling_{b}": r["best"]["ceiling"]
                    for b, r in results.items()},
                 **{f"at_tokens_{b}": r["best"]["budget_tokens"]
                    for b, r in results.items()}},
        metrics={b: {"rows": r["rows"], "best": r["best"]} for b, r in results.items()},
        traps={"secret_scan": secret_scan(),
               "exact_token_counts": exact_token_counts(TOKEN_SOURCE)},
        verdict=verdict)
    print(f"\n  recorded: experiments/E0016-soft-budget-ceiling/")
    return 0 if passing else 1


if __name__ == "__main__":
    raise SystemExit(main())
