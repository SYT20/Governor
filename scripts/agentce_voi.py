#!/usr/bin/env python3
"""Corrected query-value experiment: exact VOI, all legal queries, real policies.

Three defects in my previous version:
  1. claimed "every legal query" but tested 4 fields at ONE median threshold
  2. used a marginal-mode heuristic for the optimal decision (an exact quantity
     exists, so no heuristic is warranted -- see below)
  3. named random/fixed/greedy/oracle controls in the docstring and never ran them

The exact formulation removes the need for any heuristic. The agent cannot see
which feasible assignment is the truth, and the generator produces a unique valid
solution, so under a uniform prior over the feasible set F:

        U(F) = 1 / |F|                       probability of committing correctly
        VOI(q) = E_o[ 1/|F_o| ] - 1/|F|      exact, no estimation

A query partitions F by an observed attribute predicate. Everything below is
enumerated, not sampled.

The measurement that decides the project question is GREEDY vs LOOKAHEAD:
myopic VOI maximisation versus exhaustive search over query SEQUENCES. If they
tie, the problem is myopic and no cognitive controller is needed. If lookahead
wins materially, non-myopic reasoning has measurable value.
"""
from __future__ import annotations
import sys, json, itertools, random
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from governor.accounting.meter import Envelope
from governor.envs.agentce import AgentCEEpisode, load_instances

BIG = Envelope(tokens=1e9, cost=1e9, wall_s=1e9, tool_calls=1e9)
FIELDS = ("price", "credits", "workload", "difficulty")
BUDGET = 2                      # paid queries per episode in this probe

def build_state(inst, dom="course", cap=20000):
    ep = AgentCEEpisode(instance=inst, envelope=BIG, seed=0, strict=False)
    pool, gc = inst.item_pool, inst.global_constraints
    per_slot = []
    for (r, c) in ep.hidden_slots:
        sl = [s for s in inst.slots if s.get("row") == r and s.get("col") == c][0]
        lv = [cid for cid in sl["candidate_ids"]
              if (ep.call("set_slot", row=r, col=c, id=cid),
                  ((ep.call(f"check_{dom}_slot_constraints", row=r, col=c) or {})
                   .get("data") or {}).get("is_valid"))[1] is True]
        if not lv: return None
        per_slot.append(lv)
    if int(np.prod([len(p) for p in per_slot])) > cap: return None
    filled = [inst.partial_solution[i][j] for i in range(len(inst.partial_solution))
              for j in range(len(inst.partial_solution[0])) if inst.partial_solution[i][j]]
    rp = gc.get("total_price_max", 10**9) - sum(pool[v]["price"] for v in filled if v in pool)
    rc = gc.get("total_credits_min", 0) - sum(pool[v]["credits"] for v in filled if v in pool)
    rw = gc.get("total_workload_max", 10**9) - sum(pool[v].get("workload", 0) for v in filled if v in pool)
    F = [t for t in itertools.product(*per_slot)
         if sum(pool[x]["price"] for x in t) <= rp
         and sum(pool[x]["credits"] for x in t) >= rc
         and sum(pool[x].get("workload", 0) for x in t) <= rw]
    if len(F) < 2: return None
    # EVERY legal query: (slot, field, every distinct threshold)
    queries = []
    for si in range(len(per_slot)):
        for fld in FIELDS:
            vals = sorted({pool[x].get(fld, 0) for x in per_slot[si]})
            for thr in vals[:-1]:
                queries.append((si, fld, thr))
    return {"F": F, "queries": queries, "pool": pool,
            "H": ep.horizon, "B": inst.branch_budget}

def split(F, pool, q):
    si, fld, thr = q
    a = [t for t in F if pool[t[si]].get(fld, 0) <= thr]
    b = [t for t in F if pool[t[si]].get(fld, 0) > thr]
    return [x for x in (a, b) if x]

def U(F): return 1.0 / max(len(F), 1)

def voi(F, pool, q):
    parts = split(F, pool, q)
    if len(parts) < 2: return 0.0
    return sum((len(p)/len(F)) * U(p) for p in parts) - U(F)

def run_policy(st, kind, seed=0, budget=BUDGET):
    """Return expected success after spending `budget` queries under a policy."""
    pool, F0, Q = st["pool"], st["F"], st["queries"]
    rng = random.Random(seed)
    def rec(F, remaining, order_idx):
        if remaining == 0 or len(F) <= 1: return U(F)
        cands = [q for q in Q if len(split(F, pool, q)) > 1]
        if not cands: return U(F)
        if kind == "random":  pick = [rng.choice(cands)]
        elif kind == "fixed": pick = [cands[order_idx % len(cands)]]
        elif kind == "greedy": pick = [max(cands, key=lambda q: voi(F, pool, q))]
        elif kind == "lookahead": pick = cands
        else: raise ValueError(kind)
        best = -1.0
        for q in pick:
            parts = split(F, pool, q)
            v = sum((len(p)/len(F)) * rec(p, remaining-1, order_idx+1) for p in parts)
            best = max(best, v)
        return best
    return rec(F0, budget, 0)

def main() -> int:
    print("=" * 80)
    print("EXACT VOI — all legal queries, exact utility, real policy controls")
    print("=" * 80)
    states = []
    for inst in load_instances("course"):
        ep = AgentCEEpisode(instance=inst, envelope=BIG, seed=0, strict=False)
        if ep.horizon not in (5, 7): continue
        try: st = build_state(inst)
        except Exception: continue
        if st: states.append(st)
    if not states:
        print("  no qualifying states"); return 1
    print(f"\n  qualifying states: {len(states)}   budget={BUDGET} paid queries\n")
    print(f"  {'H':>3} {'B':>4} {'|F|':>6} {'#queries':>9} {'U0':>7} {'random':>8} "
          f"{'fixed':>8} {'greedy':>8} {'lookahead':>10}")
    print("  " + "-" * 76)
    agg = {k: [] for k in ("random", "fixed", "greedy", "lookahead")}
    u0s = []
    for st in states:
        u0 = U(st["F"]); u0s.append(u0)
        vals = {}
        for k in agg:
            v = np.mean([run_policy(st, k, seed=s) for s in range(3)]) if k == "random" \
                else run_policy(st, k)
            vals[k] = v; agg[k].append(v)
        print(f"  {st['H']:>3} {st['B']!s:>4} {len(st['F']):>6} {len(st['queries']):>9} "
              f"{u0:>7.3f} {vals['random']:>8.3f} {vals['fixed']:>8.3f} "
              f"{vals['greedy']:>8.3f} {vals['lookahead']:>10.3f}")
    print(f"\n  === MEANS ===")
    print(f"    no query (U0)      {np.mean(u0s):.4f}")
    for k in ("random", "fixed", "greedy", "lookahead"):
        print(f"    {k:<18} {np.mean(agg[k]):.4f}")
    g, l = np.mean(agg["greedy"]), np.mean(agg["lookahead"])
    f_, r = np.mean(agg["fixed"]), np.mean(agg["random"])
    print(f"\n  === VERDICT ===")
    print(f"    greedy - fixed      {g-f_:+.4f}   (is query CHOICE worth anything?)")
    print(f"    lookahead - greedy  {l-g:+.4f}   (is NON-MYOPIC reasoning worth anything?)")
    if l - g > 0.01 and g - f_ > 0.01:
        print("    -> both matter. Non-myopic query selection has measurable value. KEEP.")
    elif g - f_ > 0.01:
        print("    -> query choice matters but MYOPIC greedy is enough. A cognitive")
        print("       controller adds nothing beyond greedy VOI here.")
    else:
        print("    -> query choice is worthless; fixed order suffices. DROP AgentCE.")
    Path("results/agentce_voi.json").write_text(json.dumps(
        {k: [float(x) for x in v] for k, v in agg.items()}, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
