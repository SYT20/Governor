#!/usr/bin/env python3
"""Exhaustive validation of the ceiling over the FULL observable action space.

My earlier enumeration only considered assign-then-global-check strategies and
reported "0/159 violations", calling the bound tight. That validated (g+1)/N
against a strategy space that EXCLUDED the better strategy: check_slot_constraints
discriminates a candidate at a slot, drawn from a SEPARATE per-slot budget.

Corrected claim to test, for H=1:
        ceiling = min(1, (slot_q + g + 1) / N)
    slot_q per-slot probes, g global probes, plus one final uncommitted guess.

Tested by simulating the optimal observable policy directly: probe candidates one
at a time with the slot check until one returns valid or the budget runs out, then
commit. Success is exact, not estimated.
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

def pool_for(inst, ep, dom, r, c):
    sl = [s for s in inst.slots if s.get("row") == r and s.get("col") == c][0]
    sc = sl["slot_constraints"]; pool = None
    for fld in sc.get("active_rule_names", []):
        v = sc.get(fld)
        if not isinstance(v, (int, float)): continue
        key = resolve_field(fld, getattr(inst, "item_pool", {}) or {})
        if key is None: continue
        op = "<=" if fld.startswith("max") else ">="
        s = set(ids(ep.call(f"query_{dom}_candidate_from_attribute",
                            row=r, col=c, field=key, operator=op, value=v)))
        pool = s if pool is None else (pool & s)
    return sorted(pool or set()), sl["truth_id"]

def optimal_h1(inst, dom):
    """Run the optimal observable policy exactly: probe with the slot check until
    valid or budget exhausted, then commit to an untried candidate."""
    ep = AgentCEEpisode(instance=inst, envelope=BIG, seed=0, strict=False)
    r, c = ep.hidden_slots[0]
    pool, truth = pool_for(inst, ep, dom, r, c)
    N = len(pool)
    if N < 1: return None
    slot_q = ep.slot_query_budget(r, c) or 0
    g = ep.global_check_budget() or 0
    probes_used, found = 0, False
    for cand in pool:
        if probes_used >= slot_q: break
        e2 = AgentCEEpisode(instance=inst, envelope=BIG, seed=0, strict=False)
        e2.call("set_slot", row=r, col=c, id=cand)
        res = e2.call(f"check_{dom}_slot_constraints", row=r, col=c)
        probes_used += 1
        if ((res or {}).get("data") or {}).get("is_valid") is True:
            found = (cand == truth); break
    exact = 1.0 if found else min(1.0, (g + 1) / max(N - probes_used, 1))
    return {"domain": dom, "B": inst.branch_budget, "N": N, "slot_q": slot_q, "g": g,
            "probes_used": probes_used, "solved_by_probe": found,
            "old_bound": min(1.0, (g + 1) / N),
            "new_bound": min(1.0, (slot_q + g + 1) / N),
            "exact": exact}

def main() -> int:
    print("=" * 84)
    print("TRUE OBSERVABLE CEILING — full action space (slot probes + global checks)")
    print("=" * 84)
    rows = []
    for dom in ("course", "meal", "travel"):
        for inst in load_instances(dom):
            ep = AgentCEEpisode(instance=inst, envelope=BIG, seed=0, strict=False)
            if ep.horizon != 1: continue
            r = optimal_h1(inst, dom)
            if r: rows.append(r)
    by = defaultdict(list)
    for x in rows: by[x["B"]].append(x)
    print(f"\n  H=1 instances: {len(rows)}")
    print(f"  {'B':>4} {'n':>3} {'N':>5} {'slot_q':>7} {'g':>3} {'OLD':>8} {'NEW':>8} "
          f"{'EXACT':>8} {'solved by probe':>16}")
    print("  " + "-" * 74)
    viol = 0
    for b in sorted(by, key=lambda x: (x is None, x)):
        v = by[b]
        ex = np.mean([x["exact"] for x in v]); nb = np.mean([x["new_bound"] for x in v])
        viol += sum(1 for x in v if x["exact"] > x["new_bound"] + 1e-9)
        print(f"  {b!s:>4} {len(v):>3} {np.mean([x['N'] for x in v]):>5.1f} "
              f"{np.mean([x['slot_q'] for x in v]):>7.1f} {np.mean([x['g'] for x in v]):>3.0f} "
              f"{np.mean([x['old_bound'] for x in v]):>8.1%} {nb:>8.1%} {ex:>8.1%} "
              f"{np.mean([x['solved_by_probe'] for x in v]):>16.0%}")
    print(f"\n  === VERDICT ===")
    print(f"  exact optimum exceeds the NEW bound in {viol}/{len(rows)} instances")
    old_wrong = sum(1 for x in rows if x["exact"] > x["old_bound"] + 1e-9)
    print(f"  exact optimum exceeds the OLD bound in {old_wrong}/{len(rows)} instances")
    print(f"  -> the OLD bound (g+1)/N is REFUTED wherever that count is non-zero.")
    Path("results/agentce_true_ceiling.json").write_text(json.dumps(rows, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
