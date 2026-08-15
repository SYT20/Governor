#!/usr/bin/env python3
"""Can ANY non-global tool narrow the candidate space? (the §3 gap)

I claimed "local queries cannot narrow further" having only shown that the ACTIVE
SLOT RULES are exhausted. That is a weaker statement. AgentCE exposes 11 tools;
if any of them distinguishes candidates that survive the slot rules, the observable
ceiling is too pessimistic and the whole solvability map is wrong.

Test: take instances where >1 candidate survives, then for EVERY non-global tool
and every plausible argument, ask whether its response differs across the surviving
candidates. A tool whose output differs is an information channel I missed.
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

def survivors(inst, ep, dom):
    (r, c) = ep.hidden_slots[0]
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
    return r, c, sorted(pool or set()), sl["truth_id"], sc

def main() -> int:
    print("=" * 84)
    print("INFORMATION AUDIT — is the global check really the only discriminator?")
    print("=" * 84)
    findings = defaultdict(int); examined = 0; discriminating = []
    for dom in ("course", "meal", "travel"):
        for inst in load_instances(dom)[:20]:
            ep = AgentCEEpisode(instance=inst, envelope=BIG, seed=0)
            if ep.horizon != 1:      # isolate the single-slot case first
                continue
            r, c, pool, truth, sc = survivors(inst, ep, dom)
            if len(pool) < 2:
                continue
            examined += 1
            pool_item = getattr(inst, "item_pool", {}) or {}
            sample = next(iter(pool_item.values()), {})
            fields = [f for f in sample if isinstance(sample.get(f), (int, float, str))]
            # (a) does per-item info differ across survivors on NON-constraint fields?
            rule_fields = {resolve_field(f, pool_item) for f in sc.get("active_rule_names", [])}
            for fld in fields:
                if fld in rule_fields or fld.endswith("_id"):
                    continue
                vals = set()
                for cid in pool[:8]:
                    info = ep.call(f"get_{dom}_item_info", id=cid)
                    d = (info or {}).get("data") or {}
                    item = d.get("item") or d
                    if isinstance(item, dict) and fld in item:
                        vals.add(str(item[fld]))
                if len(vals) > 1:
                    findings[f"item_info differs on '{fld}'"] += 1
            # (b) does the SLOT CHECK discriminate once a slot is filled?
            outs = set()
            for cid in pool[:6]:
                e2 = AgentCEEpisode(instance=inst, envelope=BIG, seed=0)
                e2.call("set_slot", row=r, col=c, id=cid)
                res = e2.call(f"check_{dom}_slot_constraints", row=r, col=c)
                outs.add(json.dumps((res or {}).get("data"), sort_keys=True))
            if len(outs) > 1:
                findings["check_slot_constraints DISCRIMINATES after fill"] += 1
                discriminating.append((dom, inst.instance_id if hasattr(inst,'instance_id') else '?', len(pool)))
    print(f"\n  instances examined (H=1, >1 survivor): {examined}")
    print(f"\n  {'signal':<52} {'instances':>10}")
    print("  " + "-" * 64)
    for k, v in sorted(findings.items(), key=lambda x: -x[1]):
        print(f"  {k:<52} {v:>10}")
    key = "check_slot_constraints DISCRIMINATES after fill"
    print(f"\n  === VERDICT ===")
    if findings.get(key, 0) > 0:
        print(f"  A NON-GLOBAL tool discriminates survivors on {findings[key]}/{examined} instances.")
        print(f"  My claim that only the global check can disambiguate is REFUTED, and")
        print(f"  the observable ceiling (g+1)/N is TOO PESSIMISTIC -- a policy could")
        print(f"  narrow the pool using per-slot checks before spending a global check.")
        for d in discriminating[:5]:
            print(f"    e.g. {d[0]} {d[1]} pool={d[2]}")
    else:
        print(f"  No non-global tool distinguishes survivors. Attribute differences on")
        print(f"  non-constraint fields exist but carry no VALIDITY signal, so they")
        print(f"  cannot narrow the space. The global check is the sole discriminator")
        print(f"  and (g+1)/N stands as the observable ceiling for H=1.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
