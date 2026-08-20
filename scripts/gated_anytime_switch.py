#!/usr/bin/env python3
"""Last test on this family: an ANY-TIME switch driven by free telemetry.

Every cell of the Phase 2 ladder committed to a single decision time chosen in
advance. That is not what a cognitive layer does. The proposed architecture is

    ordinary execution -> free cognitive telemetry -> Governor -> continue cheap
                                                              -> or escalate

with the Governor consulted at EVERY step, escalating the moment the free
signals justify it. That is strictly more powerful than any fixed-t policy,
because a fixed-t policy is a special case of it.

Crucially this requires NO change to the environment. The signals used --
posterior entropy over the label, entropy over the regime, confidence margin,
remaining budget -- are byproducts already computed during ordinary myopic
acquisition. Nothing new is purchased, so no design choice here can manufacture
a favourable result: that was the standing risk with inventing a cheap
diagnostic action, and this avoids it entirely.

    A          always myopic
    C_fixed    best fixed-t switch from the sweep (regime_voi/t1, +0.025)
    C_any      escalate at the first step where E[Delta | o_t, rem] > 0
    C'         clairvoyant, unattainable

If C_any also fails to beat A, the negative result is considerably stronger:
not merely "the switch is mistimed", but "no switching rule over free signals
recovers the clairvoyant headroom in this family". That closes the family
honestly rather than by exhaustion.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.envs.gated_family import (  # noqa: E402
    REGIME_GRID,
    GateConfig,
    GatedTask,
    ObservableBayes,
)

TRAIN_SIGMA = [0.10, 0.35, 0.60, 1.50]
TEST_SIGMA = [0.20, 0.22, 0.48, 0.90]
GATE_COSTS = [1.0, 2.0]
BUDGETS = [3.0, 4.0, 5.0]
N_TABLE = 70
N_EVAL = 120
MAX_T = 4


def myopic_to_end(bayes, x, logL, available, rem, forced_first=None):
    logL = logL.copy()
    available = list(available)
    spent = 0.0
    if forced_first is not None and bayes.cost[forced_first] <= rem + 1e-9:
        available.remove(forced_first)
        spent += float(bayes.cost[forced_first])
        logL = logL + bayes.loglik_cols(x, bayes.group_cols[forced_first])
    while True:
        g = bayes.myopic_step(logL, available, rem - spent)
        if g is None:
            break
        available.remove(g)
        spent += float(bayes.cost[g])
        logL = logL + bayes.loglik_cols(x, bayes.group_cols[g])
    return int(np.argmax(bayes.label_posterior(logL)))


def build_table():
    """Dbar[(sigma, gate_cost, t, remaining)] -- keyed by REMAINING budget.

    The ladder's table was keyed by total budget, which only coincides with
    remaining budget when every pre-decision acquisition cost 1. An any-time
    switch reaches states with arbitrary remaining budget, so it has to be
    keyed correctly or the switch reads the wrong row.
    """
    tab = {}
    for so in TRAIN_SIGMA:
        for gc in GATE_COSTS:
            task = GatedTask(cfg=GateConfig(sigma_other=so, gate_cost=gc),
                             n_samples=N_TABLE, seed=101)
            bayes = ObservableBayes(task)
            for t in range(MAX_T + 1):
                for rem in (1.0, 2.0, 3.0, 4.0, 5.0):
                    d = n = 0
                    for i in range(N_TABLE):
                        x, y = task.features[i], int(task.labels[i])
                        logL = bayes.prior_logL()
                        avail = list(range(bayes.n_groups))
                        for _ in range(t):
                            g = bayes.myopic_step(logL, avail, np.inf)
                            avail.remove(g)
                            logL = logL + bayes.loglik_cols(x, bayes.group_cols[g])
                        if 0 not in avail or gc > rem:
                            continue
                        n += 1
                        a = myopic_to_end(bayes, x, logL, avail, rem)
                        b = myopic_to_end(bayes, x, logL, avail, rem, forced_first=0)
                        d += int(b == y) - int(a == y)
                    if n:
                        tab[(so, gc, t, rem)] = d / n
        print(f"    table: sigma_other={so} done")
    return tab


def expected_delta(bayes, logL, gc, t, rem, tab):
    post = bayes.regime_posterior(logL)
    return sum(post[r] * tab.get((s, gc, t, rem), 0.0)
               for r, s in enumerate(REGIME_GRID) if s in TRAIN_SIGMA)


def run_anytime(bayes, x, gc, B, tab):
    """Consult the switch at EVERY step; escalate the moment it says so."""
    logL = bayes.prior_logL()
    avail = list(range(bayes.n_groups))
    spent = 0.0
    t = 0
    escalated_at = None
    while True:
        rem = B - spent
        if 0 in avail and t <= MAX_T and gc <= rem + 1e-9:
            if expected_delta(bayes, logL, gc, t, float(round(rem)), tab) > 0.0:
                escalated_at = t
                avail.remove(0)
                spent += gc
                logL = logL + bayes.loglik_cols(x, bayes.group_cols[0])
                return myopic_to_end(bayes, x, logL, avail,
                                     B - spent), escalated_at
        g = bayes.myopic_step(logL, avail, rem)
        if g is None:
            break
        avail.remove(g)
        spent += float(bayes.cost[g])
        logL = logL + bayes.loglik_cols(x, bayes.group_cols[g])
        t += 1
    return int(np.argmax(bayes.label_posterior(logL))), escalated_at


def main() -> int:
    print("=" * 88)
    print("ANY-TIME SWITCH ON FREE TELEMETRY — the last test on this family")
    print("=" * 88)
    print("\n  No environment change. Every signal used is a byproduct of ordinary")
    print("  acquisition, so nothing here can manufacture a favourable result.\n")
    tab = build_table()

    rows = []
    print(f"\n  {'sigma':>6} {'gcost':>6} {'B':>4} | {'A myo':>7} {'B str':>7} "
          f"{'C_any':>7} {'C-prime':>8} | {'escalated':>10}")
    print("  " + "-" * 74)
    for so in TEST_SIGMA:
        for gc in GATE_COSTS:
            for B in BUDGETS:
                task = GatedTask(cfg=GateConfig(sigma_other=so, gate_cost=gc),
                                 n_samples=N_EVAL, seed=777)
                bayes = ObservableBayes(task)
                hits = {"A": 0, "B": 0, "any": 0, "Cp": 0}
                esc = 0
                for i in range(N_EVAL):
                    x, y = task.features[i], int(task.labels[i])
                    logL0 = bayes.prior_logL()
                    av0 = list(range(bayes.n_groups))
                    u_m = int(myopic_to_end(bayes, x, logL0, av0, B) == y)
                    u_s = int(myopic_to_end(bayes, x, logL0, av0, B,
                                            forced_first=0) == y) \
                        if gc <= B else u_m
                    p, at = run_anytime(bayes, x, gc, B, tab)
                    hits["A"] += u_m
                    hits["B"] += u_s
                    hits["any"] += int(p == y)
                    hits["Cp"] += max(u_m, u_s)
                    esc += int(at is not None)
                r = {k: v / N_EVAL for k, v in hits.items()}
                r |= {"sigma_other": so, "gate_cost": gc, "budget": B,
                      "escalated": esc / N_EVAL}
                rows.append(r)
                print(f"  {so:>6.2f} {gc:>6.1f} {B:>4.0f} | {r['A']:>7.3f} "
                      f"{r['B']:>7.3f} {r['any']:>7.3f} {r['Cp']:>8.3f} | "
                      f"{r['escalated']:>10.0%}")

    A = np.array([r["A"] for r in rows]); Bv = np.array([r["B"] for r in rows])
    An = np.array([r["any"] for r in rows]); Cp = np.array([r["Cp"] for r in rows])
    print(f"\n  means   A {A.mean():.3f}   B {Bv.mean():.3f}   "
          f"C_any {An.mean():.3f}   C' {Cp.mean():.3f}")

    def paired(u, v, label):
        d = u - v
        h = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        ok = d.mean() - h > 0
        print(f"    {label:<24} {d.mean():+.3f} [{d.mean()-h:+.3f}, "
              f"{d.mean()+h:+.3f}]{'*' if ok else ' '}")
        return ok

    print("\n  Paired across configurations")
    beats_a = paired(An, A, "C_any - A")
    beats_b = paired(An, Bv, "C_any - B")
    paired(Cp, A, "C' - A (headroom)")
    head = float((Cp - A).mean())
    print(f"\n    headroom captured by the any-time switch: "
          f"{(An - A).mean() / max(head, 1e-9):.0%}")

    # DEGENERACY AUDIT. Beating A and B is necessary but nowhere near sufficient.
    # A policy that merely picks the better of the two FIXED policies per
    # (gate_cost, budget) cell also beats both globally, while using no state at
    # all. That is the lookup table this family exists to rule out, and it is
    # what an any-time switch collapses to if it always fires at t=0 -- because
    # at t=0 the regime posterior is the prior, identical for every instance.
    esc = np.array([r["escalated"] for r in rows])
    bimodal = int(np.sum((esc == 0.0) | (esc == 1.0)))
    cells: dict[tuple[float, float], set[float]] = {}
    for r in rows:
        cells.setdefault((r["gate_cost"], r["budget"]), set()).add(
            round(r["escalated"], 3))
    constant_in_cell = all(len(v) == 1 for v in cells.values())
    same_as_max = int(sum(1 for r in rows
                          if abs(r["any"] - max(r["A"], r["B"])) < 1e-9))
    lookup = np.maximum(A, Bv)
    print("\n  Degeneracy audit")
    print(f"    configs with escalation exactly 0% or 100% : {bimodal}/{len(rows)}")
    print(f"    escalation constant within each (cost,budget) cell : "
          f"{constant_in_cell}")
    print(f"    configs where C_any == max(A,B) exactly    : "
          f"{same_as_max}/{len(rows)}")
    print(f"    C_any {An.mean():.3f} vs per-config max(A,B) {lookup.mean():.3f}")
    degenerate = constant_in_cell and same_as_max > 0.7 * len(rows)

    print()
    if degenerate:
        print("  DEGENERATE. The switch never conditions on instance state: it")
        print("  fires at t=0 or not at all, so its decision is a function of")
        print("  (gate_cost, budget) alone. Beating A and B here means only that")
        print("  choosing the better fixed policy per budget cell beats either")
        print("  fixed policy globally. That is the lookup table, not a cognitive")
        print("  layer. The Phase 2 STOP stands.")
    elif beats_a and beats_b:
        print("  The any-time switch DOES beat both fixed policies, and its")
        print("  decisions vary within (cost,budget) cells, so it is using state.")
        print("  Phase 3 is justified at this configuration.")
    else:
        print("  The any-time switch does NOT beat always-myopic either. Combined")
        print("  with the six fixed-t cells, no switching rule over free signals")
        print("  recovers the clairvoyant headroom in this family. The family is")
        print("  closed on evidence rather than on exhaustion.")

    Path("results").mkdir(exist_ok=True)
    Path("results/gated_anytime_switch.json").write_text(json.dumps(
        {"rows": rows, "beats_a": bool(beats_a), "beats_b": bool(beats_b),
         "headroom": head}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
