#!/usr/bin/env python3
"""Environment 2 baseline: does a deterministic policy solve AgentCE tasks under
Governor's resource envelope, and how does success move with H, B and R?

No LLM anywhere. The policy reads slot constraints, queries candidates by those
constraints, intersects, sets the slot, then finishes. Every tool call is metered
and enforced by the same Accountant validated in SynthBug.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from governor.accounting.meter import Envelope
from governor.envs.agentce import AgentCEEpisode, DOMAINS, load_instances

def constraint_policy(ep: AgentCEEpisode) -> None:
    """Read constraints -> query candidates satisfying each -> intersect -> set.

    Deliberately simple and deterministic. It is the analogue of A_fixed/C_heuristic
    in SynthBug: something honest to beat, not something to be proud of.
    """
    dom = ep.domain
    for (r, c) in ep.hidden_slots:
        if ep.exhausted:
            return
        # check_slot_constraints is a VERIFICATION tool -- it errors until the slot
        # is filled. The constraints themselves come from the task description, so
        # the policy reads them from the slot spec and queries against them.
        rules = {}
        for sspec in getattr(ep.instance, "slots", []):
            if sspec.get("row") == r and sspec.get("col") == c:
                rules = sspec.get("slot_constraints", {}) or {}
                break
        pool = None
        for fld in (rules.get("active_rule_names") or [])[:3]:
            if ep.exhausted: break
            val = rules.get(fld)
            if not isinstance(val, (int, float)): continue
            op = "<=" if fld.startswith("max") else ">="
            key = fld.replace("max_", "").replace("min_", "")
            res = ep.call(f"query_{dom}_candidate_from_attribute",
                          row=r, col=c, field=key, operator=op, value=val)
            ids = set(_ids(res))
            pool = ids if pool is None else (pool & ids)
            if pool is not None and len(pool) == 1:
                break          # unique answer: stop paying for more queries
        pick = None
        if pool:
            pick = sorted(pool)[0]
        else:
            cands = getattr(ep.instance, "slots", [])
            for s in cands:
                if s.get("row") == r and s.get("col") == c:
                    cl = s.get("candidate_ids") or []
                    pick = cl[0] if cl else None
                    break
        if pick is not None and not ep.exhausted:
            ep.call("set_slot", row=r, col=c, id=pick)
    if not ep.exhausted:
        ep.call("done")

def _ids(res) -> list:
    """Parse tool output. Verified format: {"status":..,"data":{"matches":[{"id":..}]}}.

    My first version guessed at keys like "ids"/"candidates" and never looked at
    data.matches, so every query returned an empty pool and the baseline scored
    ~0. The lesson: read one real response before writing the parser.
    """
    d = (res or {}).get("data") or {}
    m = d.get("matches") or []
    return [x.get("id") if isinstance(x, dict) else x for x in m]

def oracle_policy(ep: AgentCEEpisode) -> None:
    """Upper bound: writes the truth. Establishes the achievable ceiling."""
    for (r, c) in ep.hidden_slots:
        if ep.exhausted: return
        ep.call("set_slot", row=r, col=c, id=ep.instance.truth_solution[r][c])
    if not ep.exhausted:
        ep.call("done")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="course,meal,travel")
    ap.add_argument("--per-domain", type=int, default=54)
    args = ap.parse_args()

    # Envelope CALIBRATED FROM MEASUREMENT, not guessed. Measured per hidden slot:
    # oracle 0.0009 cost / 180 tok / 1 call; constraint policy ~0.0215 / 4300 / 5.
    # At the maximum horizon H=21 the constraint policy needs roughly
    # 21 x (3 queries + 1 set) -> ~0.29 cost, ~60k tokens, ~85 tool calls.
    # FULL is set just above that so 50% and 25% genuinely bind. My first attempt
    # used 0.60 cost, ~28x too loose, and every budget level scored identically.
    FULL = Envelope(tokens=66_000, cost=0.32, wall_s=90.0, tool_calls=95)
    print("=" * 78)
    print("ENVIRONMENT 2 — AgentCE-Bench baseline under Governor's envelope (no LLM)")
    print("=" * 78)
    print(f"\n  envelope(100%): {FULL.as_dict()}")

    rows = []
    for dom in args.domains.split(","):
        insts = load_instances(dom)[: args.per_domain]
        for scale in (1.0, 0.5, 0.25):
            env = FULL.scaled(scale)
            for pol, name in ((oracle_policy, "oracle"), (constraint_policy, "baseline")):
                wins = []; viol = 0; exh = 0; steps = []; costs = []
                for i, inst in enumerate(insts):
                    ep = AgentCEEpisode(instance=inst, envelope=env, seed=i)
                    try:
                        pol(ep)
                    except Exception:
                        pass
                    ep.acc.reconcile()
                    wins.append(int(ep.score())); viol += int(ep.acc.violated())
                    exh += int(ep.exhausted); steps.append(len(ep.steps))
                    costs.append(ep.acc.consumed()["cost"])
                    rows.append({"domain": dom, "scale": scale, "policy": name,
                                 "H": ep.horizon, "B": ep.decoy_budget,
                                 "win": wins[-1], "steps": len(ep.steps),
                                 "cost": costs[-1], "exhausted": ep.exhausted})
                print(f"  {dom:<9} {scale:>5.0%} {name:<9} "
                      f"TSR {np.mean(wins):>6.1%}  steps {np.mean(steps):>5.1f}  "
                      f"cost {np.mean(costs):>7.4f}  exhausted {exh:>3}  BVR {viol}")
    Path("results").mkdir(exist_ok=True)
    Path("results/agentce_baseline.json").write_text(json.dumps(rows, indent=1))

    print("\n  === SUCCESS vs HORIZON H (baseline, 100% budget) ===")
    byH = defaultdict(list)
    for r in rows:
        if r["policy"] == "baseline" and r["scale"] == 1.0:
            byH[r["H"]].append(r["win"])
    for h in sorted(byH):
        print(f"      H={h:<3} n={len(byH[h]):<4} TSR {np.mean(byH[h]):>6.1%}")
    print("\n  === SUCCESS vs DECOY BUDGET B (baseline, 100% budget) ===")
    byB = defaultdict(list)
    for r in rows:
        if r["policy"] == "baseline" and r["scale"] == 1.0:
            byB[r["B"]].append(r["win"])
    for b in sorted(byB, key=lambda x: (x is None, x)):
        print(f"      B={b!s:<4} n={len(byB[b]):<4} TSR {np.mean(byB[b]):>6.1%}")
    bvr = sum(1 for r in rows if r.get("exhausted") is None)
    print(f"\n  total episodes {len(rows)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
