#!/usr/bin/env python3
"""What is the BEST-OBSERVABLE-POLICY ceiling, as opposed to the omniscient oracle?

The oracle writes truth_solution and scores 100%. That verifies the scoring path
and nothing else: it reads ground truth the agent cannot see.

The real question is what a perfect policy could achieve from the ALLOWED queries.
Measured structure of the task:
  * slot constraints narrow the candidate pool to k
  * every survivor satisfies every slot rule, so further local queries cannot
    disambiguate
  * only the global check separates them, it returns a single bit (is_valid),
    and the budget is g checks
So a perfect policy is choosing among k indistinguishable candidates with g binary
probes. Its success ceiling is bounded by (g+1)/k for H=1 -- it can test g of them
and commit to another, hence at most g+1 of k outcomes are winnable.

This bound is what "headroom" must be measured against, not 100%.
"""
from __future__ import annotations
import sys, json
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from governor.accounting.meter import Envelope
from governor.envs.agentce import AgentCEEpisode, load_instances, resolve_field

BIG = Envelope(tokens=1e9, cost=1e9, wall_s=1e9, tool_calls=1e9)

def ids(res):
    d = (res or {}).get("data") or {}
    return [x.get("id") if isinstance(x, dict) else x for x in (d.get("matches") or [])]

def main() -> int:
    print("=" * 82)
    print("OBSERVABLE CEILING vs OMNISCIENT ORACLE — AgentCE-Bench")
    print("=" * 82)
    rows = []
    for dom in ("course", "meal", "travel"):
        for inst in load_instances(dom):
            ep = AgentCEEpisode(instance=inst, envelope=BIG, seed=0)
            g = ep.global_check_budget() or 0
            pools, truth_ok = [], True
            for (r, c) in ep.hidden_slots:
                sl = [s for s in inst.slots if s.get("row") == r and s.get("col") == c][0]
                sc = sl["slot_constraints"]; pool = None
                for fld in sc.get("active_rule_names", []):
                    v = sc.get(fld)
                    if not isinstance(v, (int, float)): continue
                    op = "<=" if fld.startswith("max") else ">="
                    key = resolve_field(fld, getattr(inst, "item_pool", {}) or {})
                    if key is None: continue
                    s = set(ids(ep.call(f"query_{dom}_candidate_from_attribute",
                                        row=r, col=c, field=key, operator=op, value=v)))
                    pool = s if pool is None else (pool & s)
                pool = pool or set()
                pools.append(len(pool))
                truth_ok &= sl["truth_id"] in pool
            k = int(np.prod([max(p, 1) for p in pools])) if pools else 1
            rows.append({"domain": dom, "H": ep.horizon, "B": ep.decoy_budget,
                         "pool_per_slot": pools, "joint_space": k,
                         "global_checks": g, "truth_reachable": truth_ok,
                         "ceiling": min(1.0, (g + 1) / max(k, 1))})
    print(f"\n  {'B':>4} {'n':>4} {'mean pool/slot':>15} {'mean joint space':>17} "
          f"{'checks':>7} {'CEILING':>9} {'truth reachable':>16}")
    print("  " + "-" * 80)
    byB = defaultdict(list)
    for r in rows: byB[r["B"]].append(r)
    for b in sorted(byB, key=lambda x: (x is None, x)):
        g = byB[b]
        mp = np.mean([np.mean(x["pool_per_slot"]) if x["pool_per_slot"] else 0 for x in g])
        js = np.mean([x["joint_space"] for x in g])
        ck = np.mean([x["global_checks"] for x in g])
        cl = np.mean([x["ceiling"] for x in g])
        tr = np.mean([x["truth_reachable"] for x in g])
        print(f"  {b!s:>4} {len(g):>4} {mp:>15.1f} {js:>17.3g} {ck:>7.1f} "
              f"{cl:>9.1%} {tr:>16.0%}")
    print(f"\n  ceiling = (global_checks + 1) / joint_candidate_space")
    print(f"  This is an UPPER bound on any policy that cannot see ground truth.")
    Path("results/agentce_ceiling.json").write_text(json.dumps(rows, indent=1))

    print("\n  === JOINT H x B SURFACE (ceiling), not marginals ===")
    grid = defaultdict(list)
    for r in rows: grid[(r["H"], r["B"])].append(r["ceiling"])
    Hs = sorted({r["H"] for r in rows}); Bs = sorted({r["B"] for r in rows}, key=lambda x:(x is None,x))
    print("        " + "".join(f"B={b!s:<6}" for b in Bs))
    for h in Hs:
        cells = "".join(f"{np.mean(grid[(h,b)]):>7.1%}" if grid[(h,b)] else f"{'--':>7}" for b in Bs)
        print(f"  H={h:<4}{cells}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
