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
N_EVAL = 120
DECIDE_T = [1, 2, 3]
EXPLORERS = ["myopic", "regime_voi"]
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


def walk_to(bayes, x, t, explorer="myopic"):
    """Advance t steps under the chosen pre-decision explorer.

    `regime_voi` spends its early acquisitions on learning WHAT KIND of task
    this is (max dH(sigma)/cost) rather than on the label. Those acquisitions
    are charged against the same budget, so the identification is paid for --
    which is the whole point: it must earn its cost back.
    """
    logL = bayes.prior_logL()
    available = list(range(bayes.n_groups))
    spent = 0.0
    for _ in range(t):
        if explorer == "regime_voi":
            gr = bayes.gains(logL, available, target="regime")
            g = max(gr, key=lambda a: gr[a] / bayes.cost[a])
        else:
            g = bayes.myopic_step(logL, available, np.inf)
        available.remove(g)
        spent += float(bayes.cost[g])
        logL = logL + bayes.loglik_cols(x, bayes.group_cols[g])
    return logL, available, spent


def build_table(explorer):
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
                        logL, avail, sp = walk_to(bayes, x, t, explorer)
                        if 0 not in avail:
                            continue
                        rem = B - sp
                        if rem < 1:
                            continue
                        a = roll(bayes, x, logL, avail, rem)
                        b = roll(bayes, x, logL, avail, rem, forced_first=0)
                        d += int(b == y) - int(a == y)
                    tab[(so, gc, t, B)] = d / N_TABLE
        print(f"    table: sigma_other={so} done")
    return tab


def evaluate(so, gc, B, tab, t, explorer):
    """Run every rung of the ladder on one held-out configuration."""
    task = GatedTask(cfg=GateConfig(sigma_other=so, gate_cost=gc),
                     n_samples=N_EVAL, seed=777)
    bayes = ObservableBayes(task)
    hits = {k: 0 for k in ("A", "B", "C0", "C", "Cp")}
    switched = {k: 0 for k in ("C0", "C")}
    n = 0
    for i in range(N_EVAL):
        x, y = task.features[i], int(task.labels[i])
        logL, avail, sp = walk_to(bayes, x, t, explorer)
        if 0 not in avail or B - sp < 1:
            continue
        n += 1
        rem = B - sp
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
    print("=" * 96)
    print("PHASE 2 LADDER — sweeping DECISION TIME and PRE-DECISION EXPLORER")
    print("=" * 96)
    print("""
  The t=1 / myopic cell is the PREREGISTERED experiment and its null stands:
  C - A = +0.003 [-0.015, +0.020]. It is committed (see git history) and is not
  replaced by anything below.

  The sweep is licensed by a PRIOR measurement, not by that null. gated_regime_id
  established that binary regime side-accuracy is 0.62 at t=1 against a 0.60
  no-information baseline, and only becomes actionable at t=3. I built the ladder
  deciding at t=1 anyway -- chosen because t=0 has zero state variance, never
  checked against the identifiability curve I had already produced. Deciding
  before you can possibly know the regime is a design error, and correcting it
  is not the same as searching for a friendlier configuration.

  regime_voi exploration CHARGES its acquisitions to the same budget, so
  identification has to earn its cost back. Later decision times also leave less
  budget to act with. Both effects push against C, which is the honest setup.
""")
    grid = {}
    for explorer in EXPLORERS:
        print(f"\n  Building Dbar under explorer={explorer}")
        tab = build_table(explorer)
        for t in DECIDE_T:
            rows = []
            for so in TEST_ON_GRID + TEST_OFF_GRID:
                for gc in GATE_COSTS:
                    for B in BUDGETS:
                        r = evaluate(so, gc, B, tab, t, explorer)
                        if r["n"] >= 20:
                            rows.append({"sigma_other": so, "gate_cost": gc,
                                         "budget": B, **r})
            if not rows:
                continue
            grid[(explorer, t)] = rows
            A = np.array([r["A"] for r in rows]); Bv = np.array([r["B"] for r in rows])
            C0 = np.array([r["C0"] for r in rows]); C = np.array([r["C"] for r in rows])
            Cp = np.array([r["Cp"] for r in rows])

            def ci(d):
                h = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
                return d.mean(), h

            dA, hA = ci(C - A); dB, hB = ci(C - Bv)
            d0, h0 = ci(C - C0); dP, hP = ci(Cp - C)
            beats = (dA - hA > 0) and (dB - hB > 0)
            print(f"\n  explorer={explorer:<10} decide at t={t}   "
                  f"({len(rows)} configs)")
            print(f"    A {A.mean():.3f}   B {Bv.mean():.3f}   C0 {C0.mean():.3f}"
                  f"   C {C.mean():.3f}   C' {Cp.mean():.3f}")
            print(f"    C-A  {dA:+.3f} [{dA-hA:+.3f},{dA+hA:+.3f}]"
                  f"{'*' if dA-hA>0 else ' '}"
                  f"   C-B  {dB:+.3f} [{dB-hB:+.3f},{dB+hB:+.3f}]"
                  f"{'*' if dB-hB>0 else ' '}")
            print(f"    C-C0 {d0:+.3f} [{d0-h0:+.3f},{d0+h0:+.3f}]"
                  f"{'*' if d0-h0>0 else ' '}"
                  f"   C'-C {dP:+.3f} [{dP-hP:+.3f},{dP+hP:+.3f}]  "
                  f"-> {'C BEATS BOTH' if beats else 'no'}")

    print("\n" + "=" * 96)
    print("  Summary: does C beat BOTH fixed policies anywhere?")
    print(f"  {'explorer':<12} {'t':>3} {'C-A':>18} {'C-B':>18}  verdict")
    any_pass = False
    out = {}
    for (explorer, t), rows in grid.items():
        A = np.array([r["A"] for r in rows]); Bv = np.array([r["B"] for r in rows])
        C = np.array([r["C"] for r in rows])
        dA = C - A; dB = C - Bv
        hA = 1.96 * dA.std(ddof=1) / np.sqrt(len(dA))
        hB = 1.96 * dB.std(ddof=1) / np.sqrt(len(dB))
        ok = (dA.mean() - hA > 0) and (dB.mean() - hB > 0)
        any_pass |= ok
        out[f"{explorer}|t{t}"] = {
            "C_minus_A": [float(dA.mean()), float(dA.mean()-hA), float(dA.mean()+hA)],
            "C_minus_B": [float(dB.mean()), float(dB.mean()-hB), float(dB.mean()+hB)],
            "rows": rows, "beats_both": bool(ok)}
        print(f"  {explorer:<12} {t:>3} "
              f"{f'{dA.mean():+.3f} [{dA.mean()-hA:+.3f},{dA.mean()+hA:+.3f}]':>18} "
              f"{f'{dB.mean():+.3f} [{dB.mean()-hB:+.3f},{dB.mean()+hB:+.3f}]':>18}"
              f"  {'PASS' if ok else 'no'}")
    print()
    if any_pass:
        print("  Switching is economically worthwhile in at least one cell.")
        print("  Phase 3 is justified, and must be run at that cell only, with")
        print("  the cell fixed BEFORE the learned Governor is built.")
    else:
        print("  C never beats both fixed policies. The environment does not")
        print("  contain economically useful uncertainty for this switch at any")
        print("  decision time or explorer tested. No learner fixes that: STOP.")

    Path("results").mkdir(exist_ok=True)
    Path("results/gated_phase2_sweep.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
