#!/usr/bin/env python3
"""THE decisive test: does AgentCE contain sequential information acquisition at H>1?

I concluded from a single H=1 instance that the benchmark lacks an
information-acquisition problem. That generalised beyond the measurement. At H=1
with 34/35 slots pre-filled the global constraint is nearly determined, so of
course it does not bind.

The quantity that matters is NOT candidates-per-slot. It is |F(s)| -- the number of
globally FEASIBLE JOINT assignments consistent with current knowledge. A query is
valuable if it shrinks F, even when it does not resolve any single slot.

Decision rule, fixed before running:
  * if residual slack is tight AND |F| << product of per-slot pools
      -> constraints couple the slots; real acquisition problem; KEEP AgentCE
  * if |F| ~= product of pools (constraints vacuous at every H)
      -> no coupling; DROP AgentCE, move to AFABench/AFAContext
"""
from __future__ import annotations
import sys, json, itertools
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from governor.accounting.meter import Envelope
from governor.envs.agentce import AgentCEEpisode, load_instances

BIG = Envelope(tokens=1e9, cost=1e9, wall_s=1e9, tool_calls=1e9)

def analyse(inst, dom, cap=200_000):
    ep = AgentCEEpisode(instance=inst, envelope=BIG, seed=0, strict=False)
    H = ep.horizon
    pool = inst.item_pool
    gc = inst.global_constraints
    # locally-valid set per hidden slot, via the FREE check
    per_slot = []
    for (r, c) in ep.hidden_slots:
        sl = [s for s in inst.slots if s.get("row") == r and s.get("col") == c][0]
        lv = []
        for cid in sl["candidate_ids"]:
            ep.call("set_slot", row=r, col=c, id=cid)
            o = ep.call(f"check_{dom}_slot_constraints", row=r, col=c)
            if ((o or {}).get("data") or {}).get("is_valid") is True:
                lv.append(cid)
        per_slot.append((r, c, lv, sl["truth_id"]))
    prod = int(np.prod([max(len(p), 1) for (_, _, p, _) in per_slot]))
    # residual budget the hidden slots must jointly satisfy
    filled = [inst.partial_solution[i][j]
              for i in range(len(inst.partial_solution))
              for j in range(len(inst.partial_solution[0]))
              if inst.partial_solution[i][j] is not None]
    kp = sum(pool[v]["price"] for v in filled if v in pool)
    kc = sum(pool[v]["credits"] for v in filled if v in pool)
    kw = sum(pool[v].get("workload", 0) for v in filled if v in pool)
    resid_p = gc.get("total_price_max", 10**9) - kp
    resid_c = gc.get("total_credits_min", 0) - kc
    resid_w = gc.get("total_workload_max", 10**9) - kw
    # |F| : joint assignments satisfying the aggregate constraints
    feas = None
    if prod <= cap and prod > 0:
        feas = 0
        for combo in itertools.product(*[p for (_, _, p, _) in per_slot]):
            if (sum(pool[x]["price"] for x in combo) <= resid_p
                    and sum(pool[x]["credits"] for x in combo) >= resid_c
                    and sum(pool[x].get("workload", 0) for x in combo) <= resid_w):
                feas += 1
    # slack: how much room the AVERAGE assignment leaves. <=0 means binding.
    mean_p = np.mean([np.mean([pool[x]["price"] for x in p]) for (_, _, p, _) in per_slot])
    slack_p = (resid_p - mean_p * H) / max(abs(resid_p), 1)
    return {"H": H, "B": inst.branch_budget, "pools": [len(p) for (_,_,p,_) in per_slot],
            "prod": prod, "feasible": feas,
            "coupling": (feas / prod) if (feas is not None and prod) else None,
            "resid_price": resid_p, "resid_credits": resid_c,
            "price_slack_frac": round(float(slack_p), 3)}

def main() -> int:
    print("=" * 86)
    print("BINDINGNESS TEST — do global constraints COUPLE hidden slots at H>1?")
    print("=" * 86)
    rows = []
    for dom in ("course",):
        for inst in load_instances(dom):
            ep = AgentCEEpisode(instance=inst, envelope=BIG, seed=0, strict=False)
            if ep.horizon not in (1, 5, 7): continue
            try:
                r = analyse(inst, dom)
            except Exception as e:
                print(f"  skip: {type(e).__name__}"); continue
            r["domain"] = dom; rows.append(r)
    print(f"\n  {'H':>3} {'B':>4} {'pools':>18} {'prod |A|':>10} {'feasible |F|':>13} "
          f"{'|F|/|A|':>9} {'price slack':>12}")
    print("  " + "-" * 78)
    for r in sorted(rows, key=lambda x: (x["H"], x["B"] if x["B"] is not None else -1)):
        pools = str(r["pools"][:4]) + ("..." if len(r["pools"]) > 4 else "")
        f = r["feasible"]; cpl = r["coupling"]
        print(f"  {r['H']:>3} {r['B']!s:>4} {pools:>18} {r['prod']:>10} "
              f"{(str(f) if f is not None else 'too big'):>13} "
              f"{(f'{cpl:.3f}' if cpl is not None else '--'):>9} "
              f"{r['price_slack_frac']:>12}")
    print("\n  === VERDICT ===")
    scored = [r for r in rows if r["coupling"] is not None]
    if not scored:
        print("  no instance small enough to enumerate |F|"); return 1
    for H in sorted({r["H"] for r in scored}):
        g = [r for r in scored if r["H"] == H]
        cp = np.mean([r["coupling"] for r in g])
        print(f"    H={H:<3} mean |F|/|A| = {cp:.3f}   "
              f"({'COUPLED - constraints bind' if cp < 0.9 else 'VACUOUS - no coupling'})")
    overall = np.mean([r["coupling"] for r in scored])
    print(f"\n  overall |F|/|A| = {overall:.3f}")
    if overall < 0.9:
        print("  -> global constraints DO couple hidden slots. A query about one slot")
        print("     changes what remains feasible elsewhere. AgentCE contains the")
        print("     sequential acquisition problem. KEEP IT.")
    else:
        print("  -> constraints are vacuous at every tested H: |F| ~= |A|. Slots are")
        print("     independent, so no query has downstream value. DROP AgentCE and")
        print("     move to AFABench/AFAContext.")
    Path("results/agentce_binding.json").write_text(json.dumps(rows, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
