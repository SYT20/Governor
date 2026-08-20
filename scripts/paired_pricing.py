#!/usr/bin/env python3
"""E0020 — does PRICING the resource beat merely RANKING difficulty?

Three end-to-end comparisons have failed to separate Governor from myopic. They
were the wrong instrument: the two policies choose the SAME budget level for
most items, so an episode-level mean averages the disagreements away with the
agreements and throws the pairing out.

This conditions on the disagreement set, which is defined purely by two FROZEN
policies and uses no held-out outcome to select items. On those items only:

    b = Governor correct, myopic wrong
    c = Governor wrong,  myopic correct
    McNemar exact:  b ~ Binomial(b + c, 1/2) under H0

The paired effect and its bootstrap CI are reported alongside, because a
significance test is a diagnostic and not an effect size.

NOTHING ELSE CHANGES: same data, split, B*, predictor, baseline, contract. No
probe.
"""
from __future__ import annotations

import pickle
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.harness.ledger import ExperimentRun, ExperimentSpec  # noqa: E402
from governor.harness.traps import (  # noqa: E402
    exact_token_counts, oracle_leakage, secret_scan, split_leakage,
)
from governor.phase4.s1data import FEATURE_NAMES, feature_vector, load  # noqa: E402
from governor.phase4.softbudget import (  # noqa: E402
    fit_predictors, governor_alloc, myopic_alloc, tune,
)

BENCH, B_STAR = "math", 846.0
KIND, CALIB = "logistic", "isotonic"      # selected in E0019 by calibration Brier
TOKEN_SOURCE = "simplescaling/s1-32B tokenizer (exact)"


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar. b, c are the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def main() -> int:
    data = pickle.load(open("results/s1_caps_exact.pkl", "rb"))[BENCH]
    levels = sorted(data)
    ids = sorted(data[levels[0]])
    items, _ = load(BENCH, budgets=[levels[0]])
    pr = {i.item_id: i.prompt for i in items}
    C = np.array([[data[b][i]["correct"] for b in levels] for i in ids], float)
    T = np.array([[data[b][i]["tokens"] for b in levels] for i in ids], float)
    X = np.array([feature_vector(pr[i]) for i in ids], float)
    cal = np.array([int(i[-5:]) % 2 == 0 for i in ids])
    ev = ~cal
    ev_ids = [i for i, e in zip(ids, ev) if e]

    print("=" * 92)
    print("E0020  PAIRED: does pricing beat ranking?")
    print("=" * 92)
    print(f"  {BENCH} | B*={B_STAR:.0f} | predictor {KIND}+{CALIB} | "
          f"{int(cal.sum())} cal / {int(ev.sum())} eval | frozen")

    lp = fit_predictors(X[cal], C[cal], T[cal], levels, kind=KIND, calibrate=CALIB)
    Qc, Tc = lp.predict(X[cal])
    Qe, Te = lp.predict(X[ev])
    lam_grid = np.concatenate([[0.0], np.geomspace(1e-7, 5e-2, 400)])
    tau_grid = np.linspace(0.0, 1.0, 401)
    lam = tune(lambda k: governor_alloc(Qc, Tc, k), lam_grid, T[cal], B_STAR)
    tau = tune(lambda k: myopic_alloc(Qc, k), tau_grid, T[cal], B_STAR)
    gi, mi = governor_alloc(Qe, Te, lam), myopic_alloc(Qe, tau)
    Ce, Tt = C[ev], T[ev]
    r = np.arange(len(ev_ids))
    gc, mc = Ce[r, gi].astype(int), Ce[r, mi].astype(int)
    gt, mt = Tt[r, gi], Tt[r, mi]

    dis = gi != mi
    n_dis = int(dis.sum())
    print(f"\n  lambda={lam:.3e}  tau={tau:.3f}")
    print(f"  policies agree on {len(ev_ids) - n_dis}/{len(ev_ids)} items; "
          f"DISAGREE on {n_dis} ({n_dis / len(ev_ids):.1%})")
    print(f"  mean tokens: Governor {gt.mean():.0f}, myopic {mt.mean():.0f} "
          f"(budget {B_STAR:.0f})")

    b = int(((gc == 1) & (mc == 0) & dis).sum())
    c = int(((gc == 0) & (mc == 1) & dis).sum())
    both1 = int(((gc == 1) & (mc == 1) & dis).sum())
    both0 = int(((gc == 0) & (mc == 0) & dis).sum())
    p = mcnemar_exact(b, c)
    print(f"\n  ON DISAGREEMENT ITEMS ONLY (n={n_dis})")
    print(f"    Governor right / myopic wrong : {b}")
    print(f"    Governor wrong / myopic right : {c}")
    print(f"    both right {both1}   both wrong {both0}")
    print(f"    McNemar exact two-sided p = {p:.4f}"
          f"  {'SIGNIFICANT' if p < 0.05 else 'not significant'}")

    d = (gc - mc).astype(float)
    rng = np.random.default_rng(0)
    boot_all, boot_dis = [], []
    idx_dis = np.where(dis)[0]
    for _ in range(5000):
        s = rng.integers(0, len(d), len(d))
        boot_all.append(float(d[s].mean()))
        if n_dis:
            s2 = rng.integers(0, n_dis, n_dis)
            boot_dis.append(float(d[idx_dis][s2].mean()))
    def ci(v):
        v = np.array(v)
        return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    ma, la, ha = ci(boot_all)
    md, ld, hd = ci(boot_dis) if boot_dis else (float("nan"),) * 3
    print(f"\n    paired mean(G - M), ALL items        : {ma:+.4f} [{la:+.4f}, {ha:+.4f}]")
    print(f"    paired mean(G - M), DISAGREEMENT only: {md:+.4f} [{ld:+.4f}, {hd:+.4f}]")

    # what is pricing actually doing?
    up = int((gi[dis] > mi[dis]).sum()); down = int((gi[dis] < mi[dis]).sum())
    print(f"\n  WHERE THEY DIVERGE: Governor picks a HIGHER level {up}x, "
          f"LOWER {down}x")
    if down:
        m_lower = dis & (gi < mi)
        print(f"    Governor cheaper on {int(m_lower.sum())} items: saves "
              f"{(mt[m_lower] - gt[m_lower]).mean():.0f} tok/item, "
              f"utility delta {float((gc - mc)[m_lower].mean()):+.3f}")
    if up:
        m_up = dis & (gi > mi)
        print(f"    Governor dearer  on {int(m_up.sum())} items: spends "
              f"{(gt[m_up] - mt[m_up]).mean():.0f} tok/item more, "
              f"utility delta {float((gc - mc)[m_up].mean()):+.3f}")

    rows = []
    for k in np.where(dis)[0]:
        rows.append({"item_id": ev_ids[k],
                     "governor_level": levels[gi[k]], "myopic_level": levels[mi[k]],
                     "q_gov": float(Qe[k, gi[k]]), "q_myo": float(Qe[k, mi[k]]),
                     "cost_pred_gov": float(Te[k, gi[k]]),
                     "cost_pred_myo": float(Te[k, mi[k]]),
                     "actual_tokens_gov": float(gt[k]),
                     "actual_tokens_myo": float(mt[k]),
                     "governor_correct": int(gc[k]), "myopic_correct": int(mc[k]),
                     "lambda": float(lam)})

    verdict = ("PRICING-ADDS-VALUE" if (p < 0.05 and b > c)
               else "PRICING-UNRESOLVED")
    print(f"\n  VERDICT: {verdict}")
    if verdict == "PRICING-UNRESOLVED":
        print("    Learned allocation is validated (E0019). The INCREMENTAL")
        print("    value of opportunity-cost pricing over ranking is not")
        print("    established -- which is not the same as showing it is zero.")

    traps = {"oracle_leakage": oracle_leakage(FEATURE_NAMES),
             "exact_token_counts": exact_token_counts(TOKEN_SOURCE),
             "split_leakage": split_leakage([i for i, x in zip(ids, cal) if x], ev_ids),
             "secret_scan": secret_scan()}
    spec = ExperimentSpec(
        exp_id="E0020-paired-pricing",
        title="Paired Governor-vs-myopic on decision disagreements (MATH)",
        model="s1-32B via simplescaling/results",
        budget={"contract": "SOFT_EXPECTED_BUDGET", "B_star": B_STAR,
                "charged": TOKEN_SOURCE},
        seeds={"split": "doc_id parity", "bootstrap": 0},
        split={"calibration_items": int(cal.sum()), "evaluation_items": len(ev_ids),
               "conditioning": "disagreement set defined by frozen policies; "
                               "no held-out outcome used to select items"},
        metric="McNemar exact on discordant correctness among items where the "
               "two frozen policies choose different budget levels; paired mean "
               "difference with bootstrap CI reported alongside",
        params={"predictor": f"{KIND}+{CALIB}", "lambda": float(lam),
                "tau": float(tau), "levels": levels},
        notes="Nothing changed from E0019 except the analysis.")
    run = ExperimentRun(spec, overwrite=True)
    for row in rows:
        run.append(row)
    run.finalize(summary={"n_evaluation": len(ev_ids), "n_disagree": n_dis,
                          "b_gov_right_myo_wrong": b, "c_gov_wrong_myo_right": c,
                          "mcnemar_p": p,
                          "paired_all_mean": ma, "paired_all_lo": la, "paired_all_hi": ha,
                          "paired_dis_mean": md, "paired_dis_lo": ld, "paired_dis_hi": hd,
                          "gov_higher_level": up, "gov_lower_level": down,
                          "verdict": verdict},
                 metrics={"disagreements": rows},
                 traps=traps, verdict=verdict)
    print(f"\n  recorded: experiments/E0020-paired-pricing/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
