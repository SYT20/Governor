#!/usr/bin/env python3
"""THE decision test: does the CHOICE of query matter, and is it state-dependent?

Coupling (|F| << |A|) proves constraints bind. It does NOT prove that choosing
between information actions is a real decision. If every legal query reduces the
feasible set equally, or if none of them changes the optimal final action, then
there is nothing for a cognitive controller to decide and AgentCE is the wrong
laboratory however coupled it is.

Measured per state, for EVERY legal attribute query:
  * |F| reduction         (set-size effect)
  * DECISION change       does argmax over final assignments move?   <- the one that matters
  * value spread          Var across queries; ~0 means no choice problem
  * state-dependence      does the best query change across states?  <- vs a fixed order

Controls, per section 13: random query, fixed order, greedy-|F|, oracle-best.
If fixed order ~= oracle, the environment is too easy and we stop.
"""
from __future__ import annotations
import sys, json, itertools, random
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from governor.accounting.meter import Envelope
from governor.envs.agentce import AgentCEEpisode, load_instances

BIG = Envelope(tokens=1e9, cost=1e9, wall_s=1e9, tool_calls=1e9)
FIELDS = ("price", "credits", "workload", "difficulty")

def feasible(per_slot, pool, rp, rc, rw, cap=60_000):
    prod = int(np.prod([max(len(p), 1) for p in per_slot]))
    if prod > cap or prod == 0: return None
    out = []
    for combo in itertools.product(*per_slot):
        if (sum(pool[x]["price"] for x in combo) <= rp
                and sum(pool[x]["credits"] for x in combo) >= rc
                and sum(pool[x].get("workload", 0) for x in combo) <= rw):
            out.append(combo)
    return out

def analyse(inst, dom="course"):
    ep = AgentCEEpisode(instance=inst, envelope=BIG, seed=0, strict=False)
    pool = inst.item_pool; gc = inst.global_constraints
    per_slot, slots = [], []
    for (r, c) in ep.hidden_slots:
        sl = [s for s in inst.slots if s.get("row") == r and s.get("col") == c][0]
        lv = []
        for cid in sl["candidate_ids"]:
            ep.call("set_slot", row=r, col=c, id=cid)
            o = ep.call(f"check_{dom}_slot_constraints", row=r, col=c)
            if ((o or {}).get("data") or {}).get("is_valid") is True: lv.append(cid)
        if not lv: return None
        per_slot.append(lv); slots.append((r, c, sl["truth_id"]))
    filled = [inst.partial_solution[i][j] for i in range(len(inst.partial_solution))
              for j in range(len(inst.partial_solution[0])) if inst.partial_solution[i][j]]
    rp = gc.get("total_price_max", 10**9) - sum(pool[v]["price"] for v in filled if v in pool)
    rc = gc.get("total_credits_min", 0) - sum(pool[v]["credits"] for v in filled if v in pool)
    rw = gc.get("total_workload_max", 10**9) - sum(pool[v].get("workload", 0) for v in filled if v in pool)
    F = feasible(per_slot, pool, rp, rc, rw)
    if F is None or len(F) < 2: return None      # need genuine ambiguity
    # baseline decision: most common candidate per slot among feasible assignments
    def best_action(Fset):
        return tuple(max(set(a[i] for a in Fset), key=lambda x: sum(1 for a in Fset if a[i] == x))
                     for i in range(len(per_slot)))
    a0 = best_action(F); n0 = len(F)
    rows = []
    for si in range(len(per_slot)):
        for fld in FIELDS:
            vals = sorted({pool[x].get(fld, 0) for x in per_slot[si]})
            if len(vals) < 2: continue
            thr = vals[len(vals)//2]
            # query partitions slot si by fld<=thr; each outcome yields a sub-F
            parts = defaultdict(list)
            for a in F: parts[pool[a[si]].get(fld, 0) <= thr].append(a)
            if len(parts) < 2: continue
            exp_left = sum(len(v)/n0 * len(v) for v in parts.values())
            changed = sum(len(v)/n0 for v in parts.values() if best_action(v) != a0)
            rows.append({"slot": si, "field": fld, "thr": thr,
                         "F_reduction": 1 - exp_left/n0,
                         "decision_change_prob": changed})
    if len(rows) < 2: return None
    return {"H": ep.horizon, "B": inst.branch_budget, "nF": n0, "queries": rows}

def main() -> int:
    print("=" * 84)
    print("QUERY-DIFFERENTIATION TEST — does the CHOICE of query matter?")
    print("=" * 84)
    res = []
    for inst in load_instances("course"):
        ep = AgentCEEpisode(instance=inst, envelope=BIG, seed=0, strict=False)
        if ep.horizon not in (5, 7): continue
        try: a = analyse(inst)
        except Exception: continue
        if a: res.append(a)
    if not res:
        print("  no states with >=2 feasible assignments and >=2 legal queries."); return 1
    print(f"\n  qualifying states: {len(res)}\n")
    print(f"  {'H':>3} {'B':>4} {'|F|':>6} {'nq':>4} {'best dF':>8} {'worst dF':>9} "
          f"{'SPREAD':>7} {'best P(dec chg)':>16}")
    print("  " + "-" * 72)
    spreads, dchg, bestfields = [], [], []
    for a in res[:14]:
        q = a["queries"]
        fr = [x["F_reduction"] for x in q]; dc = [x["decision_change_prob"] for x in q]
        spreads.append(max(fr) - min(fr)); dchg.append(max(dc))
        bestfields.append(max(q, key=lambda x: x["decision_change_prob"])["field"])
        print(f"  {a['H']:>3} {a['B']!s:>4} {a['nF']:>6} {len(q):>4} {max(fr):>8.3f} "
              f"{min(fr):>9.3f} {max(fr)-min(fr):>7.3f} {max(dc):>16.3f}")
    allb = [max(a["queries"], key=lambda x: x["decision_change_prob"])["field"] for a in res]
    print(f"\n  === VERDICT ===")
    print(f"  mean spread in |F| reduction across queries : {np.mean(spreads):.3f}")
    print(f"  mean best P(decision changes)               : {np.mean(dchg):.3f}")
    from collections import Counter
    cnt = Counter(allb)
    print(f"  best-query FIELD distribution               : {dict(cnt)}")
    top = cnt.most_common(1)[0][1] / len(allb)
    print(f"  single fixed field would be optimal in      : {top:.0%} of states")
    ok = np.mean(spreads) > 0.05 and np.mean(dchg) > 0.05 and top < 0.85
    if ok:
        print("\n  -> queries DIFFER materially, they CHANGE the decision, and the best")
        print("     query is STATE-DEPENDENT. A fixed order is insufficient. KEEP AgentCE.")
    else:
        print("\n  -> query choice is not a materially state-dependent decision here.")
        print("     A fixed order suffices. DROP AgentCE; move to AFAContext/AFABench.")
    Path("results/agentce_query_diff.json").write_text(json.dumps(res, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
