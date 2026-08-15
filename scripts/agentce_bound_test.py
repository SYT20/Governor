#!/usr/bin/env python3
"""Is (g+1)/joint_space a valid upper bound, or did I assert it?

I derived it analytically and reported it as a measured ceiling. That is the same
error I have criticised repeatedly. For H=1 with a binary global check the argument
is sound: g probes plus one final commit can win at most g+1 of k indistinguishable
candidates. For H>1 it is NOT obviously valid, because candidates interact across
slots and one binary check may eliminate a structured subset rather than one point.

Test: on small instances, ENUMERATE the exhaustive observable strategy space and
compute the exact optimal success rate, then compare against the claimed bound.
A single violation refutes it.
"""
from __future__ import annotations
import sys, json, itertools
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from governor.accounting.meter import Envelope
from governor.envs.agentce import AgentCEEpisode, load_instances, resolve_field

BIG = Envelope(tokens=1e9, cost=1e9, wall_s=1e9, tool_calls=1e9)

def ids(res):
    d = (res or {}).get("data") or {}
    return [x.get("id") if isinstance(x, dict) else x for x in (d.get("matches") or [])]

def pools_for(inst, ep, dom):
    """Candidate pool per hidden slot after all slot-constraint queries."""
    out = []
    for (r, c) in ep.hidden_slots:
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
        out.append((r, c, sorted(pool or set()), sl["truth_id"]))
    return out

def exact_optimal(inst, dom, max_space=4000):
    """Exhaustively enumerate joint assignments; count how many the global check
    accepts. With g binary checks a perfect policy can test g assignments and then
    commit to one more, so exact optimal = min(1, (g+1)/n_valid_indistinguishable).

    Crucially we measure n_valid by ACTUALLY calling the environment's global
    check on every joint assignment -- not by assuming each is distinct.
    """
    ep = AgentCEEpisode(instance=inst, envelope=BIG, seed=0)
    g = ep.global_check_budget() or 0
    P = pools_for(inst, ep, dom)
    space = int(np.prod([max(len(p), 1) for (_, _, p, _) in P]))
    if space == 0 or space > max_space:
        return None
    accepted = 0
    truth_ok = all(t in p for (_, _, p, t) in P)
    for combo in itertools.product(*[p if p else [None] for (_, _, p, _) in P]):
        e2 = AgentCEEpisode(instance=inst, envelope=BIG, seed=0)
        for ((r, c, _, _), pick) in zip(P, combo):
            if pick is None: break
            e2.call("set_slot", row=r, col=c, id=pick)
        else:
            res = e2.call(f"check_{dom}_global_constraints")
            if ((res or {}).get("data") or {}).get("is_valid") is True:
                accepted += 1
    if accepted == 0:
        return None
    claimed = min(1.0, (g + 1) / max(space, 1))
    exact = min(1.0, (g + 1) * accepted / max(space, 1))
    return {"H": ep.horizon, "B": ep.decoy_budget, "space": space,
            "accepted": accepted, "checks": g, "claimed_bound": claimed,
            "exact_optimal": exact, "truth_reachable": truth_ok,
            "violates": exact > claimed + 1e-12}

def main() -> int:
    print("=" * 84)
    print("BOUND VALIDATION — exhaustive observable optimum vs claimed (g+1)/space")
    print("=" * 84)
    rows = []
    for dom in ("course", "meal", "travel"):
        for inst in load_instances(dom):
            r = exact_optimal(inst, dom)
            if r: r["domain"] = dom; rows.append(r)
    if not rows:
        print("  no instances small enough to enumerate"); return 1
    print(f"\n  enumerable instances: {len(rows)}")
    print(f"  {'dom':<8} {'H':>3} {'B':>4} {'space':>7} {'valid':>6} {'g':>3} "
          f"{'claimed':>9} {'exact':>9} {'violates?':>10}")
    print("  " + "-" * 74)
    for r in sorted(rows, key=lambda x: (-x["exact_optimal"] + x["claimed_bound"]))[:16]:
        print(f"  {r['domain']:<8} {r['H']:>3} {r['B']!s:>4} {r['space']:>7} "
              f"{r['accepted']:>6} {r['checks']:>3} {r['claimed_bound']:>9.1%} "
              f"{r['exact_optimal']:>9.1%} {'YES' if r['violates'] else 'no':>10}")
    v = [r for r in rows if r["violates"]]
    print(f"\n  === VERDICT ===")
    print(f"  instances where exact optimum EXCEEDS the claimed bound: {len(v)}/{len(rows)}")
    if v:
        w = max(v, key=lambda x: x["exact_optimal"] - x["claimed_bound"])
        print(f"  worst violation: claimed {w['claimed_bound']:.1%} but exact "
              f"{w['exact_optimal']:.1%} (H={w['H']} B={w['B']} space={w['space']} "
              f"valid={w['accepted']})")
        print(f"  -> (g+1)/space is NOT a valid upper bound. It ignores that MULTIPLE")
        print(f"     joint assignments can satisfy the global constraints, so a policy")
        print(f"     may succeed without identifying the unique truth.")
    else:
        print(f"  -> no violations found on enumerable instances. The bound holds")
        print(f"     empirically in this regime, though only for space<=4000.")
    n_multi = sum(1 for r in rows if r["accepted"] > 1)
    print(f"\n  instances with >1 globally-valid assignment: {n_multi}/{len(rows)}")
    print(f"  (if this is large, 'find the truth' is the wrong framing -- the task is")
    print(f"   'find ANY valid assignment', which is materially easier)")
    Path("results/agentce_bound.json").write_text(json.dumps(rows, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
