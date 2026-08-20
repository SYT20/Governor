#!/usr/bin/env python3
"""Does the gated family actually contain harmful, neutral AND helpful deliberation?

This is the precondition for everything downstream. If Delta_meta > 0 everywhere,
a controller scores well by always deliberating and we cannot tell that apart
from competence. If it never varies at fixed budget, we have rebuilt the lookup
table with extra steps.

Two things are checked, and the second is the one that matters:

1. SIGN COVERAGE. Across the factorial family, does Delta_meta take clearly
   negative, near-zero and clearly positive values?

2. REGIME COLLISION. Are there PAIRS of configurations that are identical in
   every coarse regime variable a controller could read off without observing
   anything -- budget, group count, feature count, cost structure, prior entropy
   -- yet demand opposite decisions? Without such collisions the benchmark is
   still solvable by lookup, just a wider one.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.envs.gated_family import GateConfig, delta_meta  # noqa: E402

SIGMA_OTHER = [0.10, 0.20, 0.35, 0.60, 1.50]
GATE_COST = [1.0, 2.0]
GATE_STD = [0.10]
BUDGETS = [3.0, 4.0, 6.0]
N = 250


def main() -> int:
    print("=" * 92)
    print("GATED FAMILY PROBE — does deliberation help, do nothing, AND hurt?")
    print("=" * 92)
    print(f"\n  sigma_sig fixed at 0.10. sigma_other=0.10 means every block is as")
    print(f"  informative as the primary one, so the gate is a wasted slot.")
    print(f"  sigma_other=1.50 means only the primary block informs.\n")

    # OBSERVABLE is the policy-relevant column. ORACLE is reported alongside as
    # the teacher/upper bound, and because the gap between them is itself the
    # cost of not knowing the regime -- which is the quantity the whole Governor
    # idea is trying to recover. The first version of this probe reported only
    # the oracle and described it as a policy; that claim is retracted.
    rows = []
    print(f"  {'sig_other':>9} {'gcost':>6} {'B':>4} | "
          f"{'myopic':>7} {'strat':>7} {'D_meta':>8} {'buys gate':>10} |"
          f" {'orc myo':>8} {'orc D':>7}")
    print("  " + "-" * 84)
    for so, gc, gs, B in itertools.product(SIGMA_OTHER, GATE_COST, GATE_STD, BUDGETS):
        cfg = GateConfig(sigma_other=so, gate_cost=gc, sigma_gate=gs)
        r = delta_meta(cfg, B, n=N, seed=1, observable=True)
        o = delta_meta(cfg, B, n=N, seed=1, observable=False)
        rows.append({"sigma_other": so, "gate_cost": gc, "sigma_gate": gs,
                     "budget": B, **r,
                     "oracle_myopic": o["myopic"],
                     "oracle_delta_meta": o["delta_meta"]})
        print(f"  {so:>9.2f} {gc:>6.1f} {B:>4.0f} | "
              f"{r['myopic']:>7.3f} {r['strategic']:>7.3f} "
              f"{r['delta_meta']:>+8.3f} {r['myopic_buys_gate']:>10.1%} |"
              f" {o['myopic']:>8.3f} {o['delta_meta']:>+7.3f}")

    d = [r["delta_meta"] for r in rows]
    neg = [r for r in rows if r["delta_meta"] <= -0.03]
    zero = [r for r in rows if abs(r["delta_meta"]) < 0.03]
    pos = [r for r in rows if r["delta_meta"] >= 0.03]
    print(f"\n[1] Sign coverage over {len(rows)} configurations")
    print(f"    harmful  (D <= -0.03): {len(neg):>3}   min {min(d):+.3f}")
    print(f"    neutral  (|D| < 0.03): {len(zero):>3}")
    print(f"    helpful  (D >= +0.03): {len(pos):>3}   max {max(d):+.3f}")
    cover = bool(neg and zero and pos)
    print(f"    coverage: {'PASS' if cover else 'FAIL -- family is one-sided'}")

    print(f"\n[2] Regime collisions: same budget AND same cost structure,")
    print(f"    opposite correct decision (these are what defeat a lookup table)")
    hits = 0
    for a, b in itertools.combinations(rows, 2):
        if (a["budget"] == b["budget"] and a["gate_cost"] == b["gate_cost"]
                and a["delta_meta"] <= -0.03 and b["delta_meta"] >= 0.03):
            hits += 1
            if hits <= 6:
                print(f"    B={a['budget']:.0f} cost={a['gate_cost']:.0f}: "
                      f"sigma_other {a['sigma_other']:.2f} -> D {a['delta_meta']:+.3f}  "
                      f"vs {b['sigma_other']:.2f} -> D {b['delta_meta']:+.3f}")
    print(f"    {hits} colliding pairs found")
    print(f"\n[3] Price of not knowing the regime (oracle myopic - observable myopic)")
    gaps = [r["oracle_myopic"] - r["myopic"] for r in rows]
    print(f"    mean {sum(gaps)/len(gaps):+.3f}, max {max(gaps):+.3f}, "
          f"min {min(gaps):+.3f}")
    flips = sum(1 for r in rows
                if (r["delta_meta"] > 0.03) != (r["oracle_delta_meta"] > 0.03))
    print(f"    configurations where the CORRECT DECISION differs between "
          f"oracle and observable: {flips}/{len(rows)}")
    print("    (nonzero means the oracle cannot be used to label the observable")
    print("     agent's decisions -- teacher labels must come from observable")
    print("     counterfactuals, not from the regime-aware scorer)")

    ok = cover and hits > 0
    print(f"\n  FAMILY GATE: {'PASS' if ok else 'FAIL'}")
    if ok:
        print("  Deliberation is helpful, neutral and harmful within one family,")
        print("  and the coarse regime variables cannot tell which -- only the")
        print("  observations can. That is the state-dependence the fixed")
        print("  benchmark lacked.")

    Path("results").mkdir(exist_ok=True)
    Path("results/gated_family_probe.json").write_text(json.dumps(
        {"rows": rows, "n_neg": len(neg), "n_zero": len(zero),
         "n_pos": len(pos), "collisions": hits, "n_per_cell": N}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
