#!/usr/bin/env python3
"""Myopic vs non-myopic, with the myopic side given the TRUE posterior.

THE OBJECTION THIS ANSWERS. The previous gate's `greedy_predictive_order`
computes one feature ordering per dataset and replays it on every test row. It
is a static schedule, not a policy: it never conditions on what the current
instance has already revealed. "Non-myopic beats a static schedule" is a much
weaker claim than "non-myopic beats myopic", and only the second is worth having.

THE DESIGN. Every arm predicts with the SAME exact Bayes classifier, so the
predictor is held constant and any difference is attributable to acquisition
alone. The myopic arm plays Bayes-optimal one-step lookahead under the true
generative posterior -- the strongest myopic policy that can exist here. It
cannot be dismissed as a weak estimator.

The decisive pair is:

    myopic_exact       Bayes-optimal one-step lookahead, all k steps
    ctx_then_myopic    group 0 forced at step 1, then IDENTICAL myopic play

They share machinery, predictor, random draws and every subsequent decision
rule, and differ in exactly one action. Whatever separates them is the value of
a single non-myopic choice, isolated.

Bracketing arms:

    ctx_free_myopic    context granted at NO budget cost, then myopic. Upper
                       bound on ctx_then_myopic; the gap between them is the
                       price of the slot rather than the value of the context.
    latent_prefix      TRUE context + a fixed prefix of its block. Named
                       `oracle` in the previous gate, which was wrong -- it is
                       beaten by myopic_exact at budget 8, so it bounds nothing.
    full_obs           every feature observed. The actual information ceiling.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import train_test_split  # noqa: E402

from governor.envs.cube_nm_bayes import (  # noqa: E402
    CubeNMBayes,
    check_exact_vs_sampled,
    validate_likelihood,
)
from governor.envs.cube_nm_repro import CubeNMRepro  # noqa: E402

SEEDS = [1, 2, 3, 4, 5]
KMAX = 8
BUDGETS = list(range(1, KMAX + 1))
N_TEST = 400
ARMS = ("myopic_exact", "ctx_then_myopic", "ctx_free_myopic",
        "context_first", "latent_prefix")


def fixed_arm_preds(bayes, ds, i, groups):
    """Predictions at every prefix of a fixed acquisition list."""
    logL = np.zeros(bayes.H)
    out = []
    for g in groups:
        logL = logL + bayes.loglik_cols(ds.features[i], bayes.group_cols[g])
        out.append(int(np.argmax(bayes.label_posterior(logL))))
    return out


def mc_regression(bayes, ds, test, levels=(8, 32, 128)):
    """Retained evidence for WHY the myopic arm is scored exactly.

    A previous version of this experiment ran the myopic arm by Monte Carlo at
    n_mc=32 and reported a +0.330 advantage for the non-myopic arm at budget 4.
    Accuracy at budget 4 turned out to move 0.428 -> 0.471 -> 0.541 as n_mc went
    8 -> 128: the arm was losing partly to its own sampling noise, and part of
    that +0.330 was an estimator artefact rather than a horizon effect.

    This function reproduces that curve so the reason for the exact scorer stays
    visible in the record instead of being quietly dropped after the fix.
    """
    sub = test[:150]
    out = {}
    for m in levels:
        hits = {4: 0, 8: 0}
        for i in sub:
            _, pred = bayes.run_myopic(ds.features[i], KMAX,
                                       np.random.default_rng(7919 + int(i)), n_mc=m)
            for k in (4, 8):
                hits[k] += int(pred[k - 1] == int(ds.labels[i]))
        out[m] = {k: hits[k] / len(sub) for k in (4, 8)}
    hits = {4: 0, 8: 0}
    for i in sub:
        _, pred = bayes.run_myopic(ds.features[i], KMAX)      # exact
        for k in (4, 8):
            hits[k] += int(pred[k - 1] == int(ds.labels[i]))
    out["exact"] = {k: hits[k] / len(sub) for k in (4, 8)}
    return out


def main() -> int:
    print("=" * 88)
    print("MYOPIC vs NON-MYOPIC — myopic armed with the exact generative posterior")
    print("=" * 88)

    gate_ds = CubeNMRepro(n_samples=4000, seed=1)
    print("\n[1] Likelihood gate (a mis-derived posterior would rig this experiment)")
    for k, (ok, v) in validate_likelihood(gate_ds).items():
        print(f"    {'PASS' if ok else 'FAIL':<5} {k:<32} {v}")

    print("\n[1b] Exactness gate: quadrature/collapse vs a 20k-sample estimate")
    for k, (ok, v) in check_exact_vs_sampled(gate_ds).items():
        print(f"    {'PASS' if ok else 'FAIL':<5} {k:<32} {v}")

    acc = {k: {a: [] for a in ARMS} for k in BUDGETS}
    ceiling: list[float] = []
    ctx_steps: Counter = Counter()
    mc_rows = []

    print(f"\n[2] {len(SEEDS)} seeds x {N_TEST} test rows, budgets 1-{KMAX}, "
          f"exact scoring (no sampling)")
    for seed in SEEDS:
        ds = CubeNMRepro(n_samples=6000, seed=seed)
        bayes = CubeNMBayes(ds)
        allc = list(range(ds.n_features))
        _, te = train_test_split(np.arange(ds.n_samples), test_size=0.3, random_state=seed)
        test = te[:N_TEST]

        hits = {k: {a: 0 for a in ARMS} for k in BUDGETS}
        full = 0
        for i in test:
            y = int(ds.labels[i])
            # Deterministic scoring, so the three Bayes arms are matched by
            # construction -- no CRN bookkeeping and no residual sampling
            # asymmetry between them. The only difference is the forced/free
            # action.
            got_m, p_m = bayes.run_myopic(ds.features[i], KMAX)
            _, p_c = bayes.run_myopic(ds.features[i], KMAX, forced_first=0)
            _, p_g = bayes.run_myopic(ds.features[i], KMAX, free_groups=(0,))
            ctx_steps[got_m.index(0) + 1 if 0 in got_m else 0] += 1

            dec = int(np.argmax(ds.features[i, : ds.n_contexts]))   # implementable
            p_f = fixed_arm_preds(bayes, ds, i,
                                  [0] + ds.block_group_ids(dec)[: KMAX - 1])
            p_l = fixed_arm_preds(bayes, ds, i,
                                  ds.block_group_ids(int(ds.context[i]))[:KMAX])
            full += int(bayes.predict(ds.features[i], allc) == y)

            for k in BUDGETS:
                hits[k]["myopic_exact"] += int(p_m[k - 1] == y)
                hits[k]["ctx_then_myopic"] += int(p_c[k - 1] == y)
                hits[k]["ctx_free_myopic"] += int(p_g[k - 1] == y)
                hits[k]["context_first"] += int(p_f[k - 1] == y)
                hits[k]["latent_prefix"] += int(p_l[k - 1] == y)

        for k in BUDGETS:
            for a in ARMS:
                acc[k][a].append(hits[k][a] / len(test))
        ceiling.append(full / len(test))
        if seed == SEEDS[0]:
            mc_rows.append(mc_regression(bayes, ds, test))
        print(f"    seed {seed}: done")

    print("\n[3] When does the exact myopic policy buy the context (group 0)?")
    tot = sum(ctx_steps.values())
    print(f"    never, within {KMAX} acquisitions : {ctx_steps.get(0, 0)/tot:>6.1%}")
    for s in sorted(x for x in ctx_steps if x):
        print(f"    first bought at step {s:<2}          : {ctx_steps[s]/tot:>6.1%}")

    print("\n[4] Why the myopic arm is scored exactly (retained regression, seed 1)")
    print(f"    {'scorer':>8} {'acc @ budget 4':>16} {'acc @ budget 8':>16}")
    for m in (8, 32, 128, "exact"):
        r = mc_rows[0][m]
        print(f"    {str(m):>8} {r[4]:>16.3f} {r[8]:>16.3f}")
    print("    sampled scoring understates the myopic arm; the deficit shrinks as")
    print("    n_mc grows, so any advantage measured against it is partly artefact.")

    print("\n[5] Accuracy vs budget (all arms share the SAME exact Bayes predictor)")
    hdr = f"    {'budget':>6} " + " ".join(f"{a:>16}" for a in ARMS)
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    out = {}
    for k in BUDGETS:
        cells = []
        for a in ARMS:
            v = np.array(acc[k][a])
            cells.append(f"{v.mean():>16.3f}")
            out[f"{k}|{a}"] = {"mean": float(v.mean()),
                               "se": float(v.std(ddof=1) / np.sqrt(len(v))),
                               "per_seed": [float(x) for x in v]}
        print(f"    {k:>6} " + " ".join(cells))
    print(f"\n    information ceiling (all 55 features observed): "
          f"{np.mean(ceiling):.3f}")

    print("\n[6] Value of ONE non-myopic action: ctx_then_myopic - myopic_exact")
    print(f"    {'budget':>6} {'delta':>9} {'95% CI':>20}  {'free-context delta':>20}")
    verdict = {}
    for k in BUDGETS:
        d = np.array(acc[k]["ctx_then_myopic"]) - np.array(acc[k]["myopic_exact"])
        h = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if d.std(ddof=1) > 0 else 0.0
        g = np.array(acc[k]["ctx_free_myopic"]) - np.array(acc[k]["myopic_exact"])
        sig = "*" if d.mean() - h > 0 else " "
        print(f"    {k:>6} {d.mean():>+9.3f} {f'[{d.mean()-h:+.3f}, {d.mean()+h:+.3f}]':>20}{sig} "
              f"{g.mean():>+20.3f}")
        verdict[k] = {"delta": float(d.mean()), "lo": float(d.mean() - h),
                      "hi": float(d.mean() + h), "free_delta": float(g.mean())}

    Path("results").mkdir(exist_ok=True)
    Path("results/cube_nm_myopic.json").write_text(json.dumps(
        {"accuracy": out, "verdict": verdict,
         "ceiling": float(np.mean(ceiling)),
         "ctx_first_bought_at_step": {str(k): v for k, v in ctx_steps.items()},
         "mc_regression": [{str(m): r[m] for m in r} for r in mc_rows],
         "scorer": "exact", "n_test": N_TEST, "seeds": SEEDS}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
