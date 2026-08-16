#!/usr/bin/env python3
"""Phase 2: the switching ladder. Is the switch worth making at all?

    A   always myopic          never deliberate
    B   always strategic       always buy the gate first
    C0  regime-belief switch   strategic iff P(gate-good regime | o) > 0.5
    C   observable Bayes       strategic iff E[Delta | o, t, b] > C_meta
    C'  clairvoyant            strategic iff the REALISED Delta > C_meta

WHY C0 EXISTS. The regime-VOI explorer was shown to identify sigma_other faster
than label-greedy acquisition, and I over-read that as "identifying the task is
a metareasoning act with measurable value". It is not the same claim. That
explorer maximises dH(sigma)/cost, which is information-seeking; a state can
have huge regime-information gain and zero DECISION value, and the reverse.

C0 separates them. It is handed the regime belief and switches on it directly --
regime identification plus a lookup. C additionally weighs how much the regime
is worth GIVEN the remaining budget and the gate's price.

    C > C0   knowing the regime is not sufficient; the controller must know
             whether knowing it is worth paying for. That is the cognitive-layer
             claim.
    C ~ C0   the problem is Bayesian regime identification plus a threshold.
             Still useful, considerably less novel. Report it as such.

HOW C IS COMPUTED, and its approximation stated up front. C needs
E[Delta | o_t, b]. Evaluating that by nested simulation at every decision costs
seconds per decision. Instead a table Dbar(sigma_r, t, b) is precomputed ONCE by
observable simulation, and

    E[Delta | o_t, b] = sum_r P(sigma_r | o_t) * Dbar(sigma_r, t, b)

This is exact in the regime marginal and approximate in everything else: it
conditions on (t, b) but not on the finer within-regime state. It is therefore a
LOWER bound on what a perfect model-based switch could do, which is the safe
direction -- it cannot manufacture an advantage for C.

C IS NOT A PRACTICAL GOVERNOR and must not be described as one. It is handed the
regime grid, the likelihoods and the generative model. It is the teacher and the
target for Phase 3, whose question is whether a learned controller can approach
C from observable state WITHOUT the generative model.

C' uses the realised outcome and is unattainable by any policy. U(C') - U(C) is
the VALUE OF PERFECT INFORMATION about the regime, not the "cost of uncertainty"
-- it also absorbs decision noise and the tabular approximation above.
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

TRAIN_SIGMA = [0.10, 0.35, 0.60, 1.50]      # Dbar is built from these only
TEST_ON_GRID = [0.20]                        # held-out grid point
TEST_OFF_GRID = [0.22, 0.48, 0.90]           # BETWEEN grid points: misspecified
GATE_COSTS = [1.0, 2.0]
BUDGETS = [3.0, 4.0, 5.0]
FORK_T = [0, 1, 2]
N_TABLE = 80
N_EVAL = 150
LOW = [i for i, s in enumerate(REGIME_GRID) if s <= 0.20]


def roll(bayes, x, logL, available, budget_left, forced_first=None):
    logL = logL.copy()
    available = list(available)
    spent = 0.0
    if forced_first is not None and bayes.cost[forced_first] <= budget_left + 1e-9:
        available.remove(forced_first)
        spent += float(bayes.cost[forced_first])
        logL = logL + bayes.loglik_cols(x, bayes.group_cols[forced_first])
    while True:
        g = bayes.myopic_step(logL, available, budget_left - spent)
        if g is None:
            break
        available.remove(g)
        spent += float(bayes.cost[g])
        logL = logL + bayes.loglik_cols(x, bayes.group_cols[g])
    return int(np.argmax(bayes.label_posterior(logL)))


def walk_to(bayes, x, t):
    """Advance the observable myopic policy t steps; return (logL, available)."""
    logL = bayes.prior_logL()
    available = list(range(bayes.n_groups))
    for _ in range(t):
        g = bayes.myopic_step(logL, available, np.inf)
        available.remove(g)
        logL = logL + bayes.loglik_cols(x, bayes.group_cols[g])
    return logL, available


def build_table():
    """Dbar[(sigma, gate_cost, t, budget)] from TRAIN sigmas only."""
    tab = {}
    for so in TRAIN_SIGMA:
        for gc in GATE_COSTS:
            task = GatedTask(cfg=GateConfig(sigma_other=so, gate_cost=gc),
                             n_samples=N_TABLE, seed=101)
            bayes = ObservableBayes(task)
            for t in FORK_T:
                for B in BUDGETS:
                    rem = B - t
                    if rem < 1:
                        continue
                    d = 0
                    for i in range(N_TABLE):
                        x, y = task.features[i], int(task.labels[i])
                        logL, avail = walk_to(bayes, x, t)
                        if 0 not in avail:
                            continue
                        a = roll(bayes, x, logL, avail, rem)
                        b = roll(bayes, x, logL, avail, rem, forced_first=0)
                        d += int(b == y) - int(a == y)
                    tab[(so, gc, t, B)] = d / N_TABLE
        print(f"    table: sigma_other={so} done")
    return tab


def evaluate(so, gc, B, tab):
    """Run every rung of the ladder on one held-out configuration."""
    task = GatedTask(cfg=GateConfig(sigma_other=so, gate_cost=gc),
                     n_samples=N_EVAL, seed=777)
    bayes = ObservableBayes(task)
    hits = {k: 0 for k in ("A", "B", "C0", "C", "Cp")}
    switched = {k: 0 for k in ("C0", "C")}
    n = 0
    for i in range(N_EVAL):
        x, y = task.features[i], int(task.labels[i])
        # decide at t=1: t=0 has no state to condition on (measured, 3.01e-27)
        t = 1
        logL, avail = walk_to(bayes, x, t)
        if 0 not in avail:
            continue
        n += 1
        rem = B - t
        u_m = int(roll(bayes, x, logL, avail, rem) == y)
        u_s = int(roll(bayes, x, logL, avail, rem, forced_first=0) == y)
        hits["A"] += u_m
        hits["B"] += u_s
        hits["Cp"] += max(u_m, u_s)          # unattainable: uses the outcome

        post = bayes.regime_posterior(logL)
        p_low = float(sum(post[j] for j in LOW))
        go_c0 = p_low <= 0.5                  # "gate-good" side
        hits["C0"] += u_s if go_c0 else u_m
        switched["C0"] += int(go_c0)

        # C: expected Delta under the precomputed table, C_meta already inside
        # Dbar because the gate's price is charged against the same budget.
        exp_d = sum(post[r] * tab.get((s_r, gc, t, B), 0.0)
                    for r, s_r in enumerate(REGIME_GRID) if s_r in TRAIN_SIGMA)
        go_c = exp_d > 0.0
        hits["C"] += u_s if go_c else u_m
        switched["C"] += int(go_c)
    return {k: v / max(n, 1) for k, v in hits.items()} | {
        "n": n, "switch_C0": switched["C0"] / max(n, 1),
        "switch_C": switched["C"] / max(n, 1)}


def main() -> int:
    print("=" * 92)
    print("PHASE 2 LADDER — A / B / C0 / C / C'   (decide at t=1)")
    print("=" * 92)
    print(f"\n  Dbar built from sigma_other in {TRAIN_SIGMA} ONLY.")
    print(f"  Held-out on-grid : {TEST_ON_GRID}")
    print(f"  Held-out OFF-grid: {TEST_OFF_GRID}  <- between grid points, so the")
    print(f"  observable agent's regime model is MISSPECIFIED there by design.\n")
    tab = build_table()

    rows = []
    print(f"\n  {'sigma':>6} {'gcost':>6} {'B':>4} {'n':>4} | "
          f"{'A myo':>7} {'B str':>7} {'C0':>7} {'C':>7} {'C-prime':>8} | "
          f"{'sw C0':>6} {'sw C':>6}")
    print("  " + "-" * 90)
    for so in TEST_ON_GRID + TEST_OFF_GRID:
        for gc in GATE_COSTS:
            for B in BUDGETS:
                r = evaluate(so, gc, B, tab)
                rows.append({"sigma_other": so, "gate_cost": gc, "budget": B, **r})
                print(f"  {so:>6.2f} {gc:>6.1f} {B:>4.0f} {r['n']:>4} | "
                      f"{r['A']:>7.3f} {r['B']:>7.3f} {r['C0']:>7.3f} "
                      f"{r['C']:>7.3f} {r['Cp']:>8.3f} | "
                      f"{r['switch_C0']:>6.0%} {r['switch_C']:>6.0%}")

    def agg(sel, key):
        v = np.array([r[key] for r in rows if r["sigma_other"] in sel])
        return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))

    print("\n  Aggregates (mean over configurations, +- 95% CI across configs)")
    for tag, sel in (("on-grid", TEST_ON_GRID), ("OFF-grid", TEST_OFF_GRID),
                     ("all held-out", TEST_ON_GRID + TEST_OFF_GRID)):
        print(f"\n    {tag}")
        for k, name in (("A", "always myopic"), ("B", "always strategic"),
                        ("C0", "regime-belief switch"),
                        ("C", "observable Bayes switch"),
                        ("Cp", "clairvoyant (unattainable)")):
            m, h = agg(sel, k)
            print(f"      {name:<28} {m:.3f} +- {h:.3f}")

    print("\n  Verdict")
    all_sel = TEST_ON_GRID + TEST_OFF_GRID
    A = np.array([r["A"] for r in rows]); Bv = np.array([r["B"] for r in rows])
    C0 = np.array([r["C0"] for r in rows]); C = np.array([r["C"] for r in rows])
    Cp = np.array([r["Cp"] for r in rows])

    def paired(u, v, label):
        d = u - v
        h = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        star = "*" if abs(d.mean()) - h > 0 else " "
        print(f"    {label:<26} {d.mean():+.3f} [{d.mean()-h:+.3f}, "
              f"{d.mean()+h:+.3f}]{star}")
        return d.mean() - h > 0

    beats_a = paired(C, A, "C - A (vs never)")
    beats_b = paired(C, Bv, "C - B (vs always)")
    beats_c0 = paired(C, C0, "C - C0 (decision vs regime)")
    paired(Cp, C, "C' - C (VPI on regime)")

    print()
    if beats_a and beats_b:
        print("    Switching is economically worthwhile: C beats BOTH fixed")
        print("    policies. Phase 3 (learned Governor) is justified.")
    else:
        print("    C does NOT beat both fixed policies. The environment lacks")
        print("    economically useful uncertainty at these settings; no learner")
        print("    fixes that. Stop rather than proceeding to Phase 3.")
    print("    " + ("C > C0: knowing the regime is NOT sufficient -- the "
                    "controller must know whether it is worth paying for."
                    if beats_c0 else
                    "C ~ C0: the problem reduces to regime identification plus "
                    "a threshold. Report it that way, it is less novel."))

    Path("results").mkdir(exist_ok=True)
    Path("results/gated_phase2_ladder.json").write_text(json.dumps(
        {"rows": rows, "table": {str(k): v for k, v in tab.items()},
         "train_sigma": TRAIN_SIGMA, "test_on_grid": TEST_ON_GRID,
         "test_off_grid": TEST_OFF_GRID, "n_eval": N_EVAL}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
