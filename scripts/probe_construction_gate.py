#!/usr/bin/env python3
"""Preregistered gates G1, G2a, G2b, G2c for Environment 4a — construction only.

These test the ENVIRONMENT, not any policy. They must pass before a single
switching result is computed, because every downstream claim rests on the probe
being what the preregistration says it is: cheap, informative about the regime,
and carrying no label information through any path.

G2b is the one that matters and the one that was added under review. Marginal
independence is not enough -- a probe can be marginally independent of y and
still carry label information once conditioned on what else has been observed.
So the test is whether a classifier given (probe + all block observations) beats
one given the block observations alone.

Also reports, as preregistered environment characterisation and NOT as a tuning
input, the break-even probe price

    C*(s) = V_probe(s) - V_no_probe(s)

on TRAINING configurations. probe_cost stays at its preregistered 0.25 whatever
this shows; a change would be a recorded protocol deviation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import mutual_info_score  # noqa: E402

from governor.envs.gated_family import REGIME_GRID, REGIME_PRIOR  # noqa: E402
from governor.envs.probe_family import (  # noqa: E402
    ObservableProbeBayes,
    ProbeTask,
    make_config,
)

TRAIN_SIGMA = [0.10, 0.35, 0.60, 1.50]
SIGMA_PROBE = [0.05, 0.15]
PROBE_COST = 0.25
N = 2500


def g1_observability():
    """Regime posterior before any observation must equal the prior exactly."""
    worst = 0.0
    for so in TRAIN_SIGMA:
        for sp in SIGMA_PROBE:
            t = ProbeTask(cfg=make_config(so, 1.0, sp), n_samples=20, seed=3)
            b = ObservableProbeBayes(t)
            worst = max(worst, float(np.max(np.abs(
                b.regime_posterior(b.prior_logL()) - np.array(REGIME_PRIOR)))))
    return worst < 1e-9, worst


def g2(so, sp):
    """G2a/b/c on one configuration."""
    t = ProbeTask(cfg=make_config(so, 1.0, sp), n_samples=N, seed=5)
    pc = t.cfg.probe_col
    blocks = list(range(t.cfg.n_contexts, pc))
    tr, te = train_test_split(np.arange(N), test_size=0.3, random_state=0)

    def acc(cols):
        m = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        m.fit(t.features[np.ix_(tr, cols)], t.labels[tr])
        return float(m.score(t.features[np.ix_(te, cols)], t.labels[te]))

    a_probe = acc([pc])
    a_blocks = acc(blocks)
    a_both = acc(blocks + [pc])
    # I(probe ; c): discretise the probe, compare against a permutation floor
    bins = np.digitize(t.features[:, pc], np.quantile(t.features[:, pc],
                                                      np.linspace(0.05, 0.95, 12)))
    mi = float(mutual_info_score(bins, t.context))
    rng = np.random.default_rng(0)
    null = float(np.quantile([mutual_info_score(rng.permutation(bins), t.context)
                              for _ in range(60)], 0.95))
    return {"a_probe": a_probe, "a_blocks": a_blocks, "a_both": a_both,
            "mi_probe_ctx": mi, "mi_null_p95": null}


def cstar(so, sp, budget, n=120):
    """Break-even probe price at this configuration: V_probe - V_no_probe.

    The 'with probe' arm buys the probe FREE here, deliberately: C*(s) is the
    price at which buying becomes worthwhile, so the probe's own cost must not
    be inside the value being compared against it.
    """
    t = ProbeTask(cfg=make_config(so, 1.0, sp), n_samples=n, seed=9)
    b = ObservableProbeBayes(t)
    pg = t.cfg.probe_group

    def play(x, y, free_probe):
        logL = b.prior_logL()
        avail = list(range(b.n_groups))
        spent = 0.0
        if free_probe:
            avail.remove(pg)
            logL = logL + b.loglik_cols(x, b.group_cols[pg])
        else:
            avail.remove(pg)                      # not offered at all
        while True:
            g = b.myopic_step(logL, avail, budget - spent)
            if g is None:
                break
            avail.remove(g)
            spent += float(b.cost[g])
            logL = logL + b.loglik_cols(x, b.group_cols[g])
        return int(np.argmax(b.label_posterior(logL)) == y)

    with_p = sum(play(t.features[i], int(t.labels[i]), True) for i in range(n))
    no_p = sum(play(t.features[i], int(t.labels[i]), False) for i in range(n))
    return (with_p - no_p) / n


def main() -> int:
    print("=" * 84)
    print("ENVIRONMENT 4a — PREREGISTERED CONSTRUCTION GATES (G1, G2a/b/c)")
    print("=" * 84)

    ok1, w = g1_observability()
    print(f"\n  G1 observability: max |posterior - prior| at t=0 = {w:.2e}   "
          f"{'PASS' if ok1 else 'FAIL'}")

    print(f"\n  G2 — the probe must carry NO label information")
    print(f"  {'sigma_o':>8} {'sig_p':>6} | {'probe only':>11} {'blocks':>8} "
          f"{'blocks+probe':>13} | {'I(probe;c)':>11} {'null p95':>9}")
    print("  " + "-" * 78)
    rows, ok2 = [], True
    for so in TRAIN_SIGMA:
        for sp in SIGMA_PROBE:
            r = g2(so, sp)
            rows.append({"sigma_other": so, "sigma_probe": sp, **r})
            a = r["a_probe"] < 0.125 + 0.03
            bb = r["a_both"] <= r["a_blocks"] + 0.02
            cc = r["mi_probe_ctx"] <= r["mi_null_p95"]
            ok2 &= a and bb and cc
            flag = "" if (a and bb and cc) else "   <-- FAIL"
            print(f"  {so:>8.2f} {sp:>6.2f} | {r['a_probe']:>11.3f} "
                  f"{r['a_blocks']:>8.3f} {r['a_both']:>13.3f} | "
                  f"{r['mi_probe_ctx']:>11.4f} {r['mi_null_p95']:>9.4f}{flag}")
    print(f"\n  G2a probe alone <= chance+0.03      "
          f"{'PASS' if all(r['a_probe'] < 0.155 for r in rows) else 'FAIL'}")
    print(f"  G2b blocks+probe <= blocks+0.02     "
          f"{'PASS' if all(r['a_both'] <= r['a_blocks'] + 0.02 for r in rows) else 'FAIL'}")
    print(f"  G2c I(probe;c) within permutation   "
          f"{'PASS' if all(r['mi_probe_ctx'] <= r['mi_null_p95'] for r in rows) else 'FAIL'}")

    print(f"\n  Break-even price C*(s) on TRAINING configs "
          f"(characterisation only; probe_cost stays {PROBE_COST})")
    print(f"  {'sigma_o':>8} {'sig_p':>6} " +
          " ".join(f"{'B=' + str(b):>8}" for b in (3, 4, 5)))
    cs = []
    for so in TRAIN_SIGMA:
        for sp in SIGMA_PROBE:
            vals = [cstar(so, sp, float(b)) for b in (3, 4, 5)]
            cs += vals
            print(f"  {so:>8.2f} {sp:>6.2f} " +
                  " ".join(f"{v:>+8.3f}" for v in vals))
    cs = np.array(cs)
    frac = float(np.mean(cs > PROBE_COST))
    print(f"\n    C* range [{cs.min():+.3f}, {cs.max():+.3f}], "
          f"median {np.median(cs):+.3f}")
    print(f"    fraction of configurations with C* > {PROBE_COST}: {frac:.0%}")
    if frac == 0:
        print("    -> the probe is never worth its preregistered price. That is a")
        print("       FINDING, not a reason to lower probe_cost.")
    elif frac == 1:
        print("    -> the probe is always worth buying: the decision would be")
        print("       degenerate and G7 will fail.")
    else:
        print("    -> C* straddles the operating point, so the buy/skip decision")
        print("       is genuinely contested across configurations.")

    ok = ok1 and ok2
    print(f"\n  CONSTRUCTION GATE: {'PASS' if ok else 'FAIL'}")
    Path("results").mkdir(exist_ok=True)
    Path("results/probe_construction_gate.json").write_text(json.dumps(
        {"g1_worst": w, "g2": rows, "cstar": cs.tolist(),
         "frac_cstar_above_cost": frac, "probe_cost": PROBE_COST}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
