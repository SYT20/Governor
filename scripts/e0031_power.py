#!/usr/bin/env python3
"""E0031 -- can this design detect an effect at all? SYNTHETIC ONLY.

Run BEFORE the evaluation set is touched, and using no real data whatsoever.
Every number here comes from simulated problems with a KNOWN difficulty effect,
so nothing it reports can leak information about the real outcome. Revising a
design on the strength of this is legitimate; that is what a power analysis is
for.

WHY IT WAS RUN. A first smoke test of the E0031 driver, on synthetic data with a
strong difficulty effect (rescue rates 0.45 / 0.20 / 0.07), produced:

    governor   vs fixed  +0.0283  [+0.0120, +0.0466]
    random     vs fixed  +0.0186  [+0.0009, +0.0365]
    governor - random    +0.0089  [-0.0089, +0.0267]

The preregistration requires the Governor to beat BOTH the cost-matched fixed
baseline AND a random subset of equal size. It cleared the first and missed the
second -- on data built to contain the effect. If that is the typical outcome,
the criterion is unattainable at n=225 and running it would burn the only
untouched evaluation set on a test that cannot succeed.

TWO THINGS THIS MEASURES

  power(effect)   how often gov-beats-random is detected, as the difficulty
                  effect grows from nothing to implausibly strong.
  false positives at zero effect, how often the criterion fires anyway. A
                  criterion that never fires is useless; one that fires on
                  noise is worse.

The `random` arm is not a formality. It already spends only on problems whose
first sample failed, so it captures the entire benefit of NOT WASTING TOKENS on
problems that are already solved. Beating fixed spend therefore proves only that
the eligibility filter works, which nobody doubts. Only gov-minus-random
isolates the difficulty signal.

Run:  python scripts/e0031_power.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / "results" / "e0029_problems.json"
OUT = ROOT / "results" / "E0031_power.json"

N_REPLICATES = 300
FRAC_FIXED = 0.5      # design constant; see the frac-vs-discordance analysis
N_BOOT = 800
LEVELS = ("easy", "medium", "hard")

# Anchored on the real marginal rates so the simulation is the right SHAPE:
# solve@1 0.289 and solve@all 0.427 on evaluation, i.e. ~13.8 points rescuable.
BASE_SOLVE1 = {"easy": 0.45, "medium": 0.28, "hard": 0.13}
BASE_RESCUE = 0.21          # mean P(rescuable | sample 1 failed)


def rescue_by_level(spread: float) -> dict:
    """`spread` scales how much difficulty predicts rescuability.

    0.0 -> every level identical: difficulty carries NOTHING, and any apparent
           advantage is the criterion firing on noise.
    1.0 -> roughly the separation E0029's calibration suggested.
    2.0 -> stronger than anything observed; if the design cannot detect THIS it
           cannot detect anything.
    """
    mult = {"easy": 1.0 + 0.9 * spread, "medium": 1.0,
            "hard": max(0.02, 1.0 - 0.9 * spread)}
    return {d: float(np.clip(BASE_RESCUE * mult[d], 0.0, 0.95)) for d in LEVELS}


def one_trial(qids_by_diff, rng, spread, frac_grid, k_grid):
    v_true = rescue_by_level(spread)
    cal, ev = [], []
    for d, qs in qids_by_diff.items():
        for i, q in enumerate(qs):
            solved1 = rng.random() < BASE_SOLVE1[d]
            resc = (not solved1) and (rng.random() < v_true[d])
            rec = (d, solved1, resc)
            (cal if i % 2 == 0 else ev).append(rec)

    def rates(rows):
        out = {}
        for d in LEVELS:
            el = [r for r in rows if r[0] == d and not r[1]]
            out[d] = (sum(r[2] for r in el) / len(el)) if el else 0.0
        return out

    def sim(rows, score, frac, k):
        """k extra samples; each has an independent chance of landing."""
        elig = [i for i, r in enumerate(rows) if not r[1]]
        take = set(np.array(elig)[np.argsort(-score[elig])[
            :int(round(frac * len(elig)))]]) if elig else set()
        U = np.empty(len(rows))
        for i, (d, s1, resc) in enumerate(rows):
            if s1:
                U[i] = 1.0
            elif i in take and resc:
                U[i] = 1.0 if rng.random() < 1 - (1 - 0.35) ** k else 0.0
            else:
                U[i] = 0.0
        return U

    v_hat = rates(cal)
    # FRAC IS FIXED BY DESIGN, not fitted. Optimising it against the fixed
    # baseline drives it to 1.0, where both arms take every eligible problem,
    # the ranking is never consulted, and gov-minus-random is zero by
    # construction. Only K_EXTRA is chosen on calibration.
    bf = FRAC_FIXED
    cal_score = np.array([v_hat[d] if not s1 else -np.inf for d, s1, _ in cal])
    best, bk = -9.0, k_grid[0]
    base_cal = np.array([1.0 if s1 else 0.0 for _, s1, _ in cal])
    for k in k_grid:
        u = sim(cal, cal_score, bf, k).mean() - base_cal.mean()
        if u > best:
            best, bk = u, k

    ev_gov = np.array([v_hat[d] if not s1 else -np.inf for d, s1, _ in ev])
    ev_rnd = np.array([rng.random() if not s1 else -np.inf for _, s1, _ in ev])
    Ug = sim(ev, ev_gov, bf, bk)
    Ur = sim(ev, ev_rnd, bf, bk)
    d = Ug - Ur

    # A: the preregistered statistic -- bootstrap the mean over ALL problems.
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean()
                   for _ in range(N_BOOT)])
    a_ok = float(np.percentile(bs, 2.5)) > 0

    # B: McNemar on DISCORDANT pairs. Most problems get the identical
    # allocation under both arms and contribute exactly zero; they add
    # denominator without adding evidence. Conditioning on the pairs where the
    # arms actually disagree is the standard paired-binary test and is what
    # E0020 used.
    b_win = int((d > 0).sum())
    b_los = int((d < 0).sum())
    n_disc = b_win + b_los
    if n_disc:
        # exact one-sided binomial tail, P(X >= b_win | p=0.5)
        from math import comb
        pval = sum(comb(n_disc, i) for i in range(b_win, n_disc + 1)) / 2 ** n_disc
    else:
        pval = 1.0
    b_ok = pval < 0.05

    # C: cross-fitted over ALL problems. Every problem is evaluated once with
    # v(d) estimated from the OTHER folds, so nothing informs its own
    # allocation, and the test set is the whole corpus rather than half of it.
    allrows = cal + ev
    folds = np.array([i % 5 for i in range(len(allrows))])
    rng.shuffle(folds)
    Ug_c = np.empty(len(allrows)); Ur_c = np.empty(len(allrows))
    for f in range(5):
        tr = [allrows[i] for i in range(len(allrows)) if folds[i] != f]
        te_idx = [i for i in range(len(allrows)) if folds[i] == f]
        te = [allrows[i] for i in te_idx]
        vh = rates(tr)
        gsc = np.array([vh[d] if not s1 else -np.inf for d, s1, _ in te])
        rsc = np.array([rng.random() if not s1 else -np.inf for _, s1, _ in te])
        Ug_c[te_idx] = sim(te, gsc, bf, bk)
        Ur_c[te_idx] = sim(te, rsc, bf, bk)
    dc = Ug_c - Ur_c
    w, l = int((dc > 0).sum()), int((dc < 0).sum())
    nd = w + l
    if nd:
        from math import comb as _c
        pc = sum(_c(nd, i) for i in range(w, nd + 1)) / 2 ** nd
    else:
        pc = 1.0
    c_ok = pc < 0.05
    return float(d.mean()), a_ok, b_ok, c_ok, n_disc


def main() -> int:
    probs = json.loads(PROBLEMS.read_text())
    by = {d: [p["qid"] for p in probs if p.get("difficulty") == d] for d in LEVELS}
    print("\nE0031 POWER -- synthetic only, no real data touched\n")
    print(f"  problem mix: " + "  ".join(f"{d}={len(by[d])}" for d in LEVELS))
    print(f"  replicates {N_REPLICATES}, bootstrap {N_BOOT}\n")

    frac_grid = list(np.linspace(0.2, 1.0, 5))
    k_grid = [1, 2, 3, 5, 9]
    rows = []
    print(f"  {'spread':>7} {'v(easy)':>8} {'v(hard)':>8} {'gov-rnd':>11} "
          f"{'discord':>8} {'A boot':>8} {'B McNem':>9} {'C xfit':>10}")
    print(f"  {'':>7} {'':>8} {'':>8} {'':>11} {'pairs':>8} {'n=225':>8} "
          f"{'n=225':>9} {'n=475':>10}")
    for spread in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        rng = np.random.default_rng(1000 + int(spread * 100))
        eff, ha, hb, hc, nds = [], 0, 0, 0, []
        for _ in range(N_REPLICATES):
            e, a, b, c, nd = one_trial(by, rng, spread, frac_grid, k_grid)
            eff.append(e); ha += a; hb += b; hc += c; nds.append(nd)
        v = rescue_by_level(spread)
        pa, pb, pc = ha / N_REPLICATES, hb / N_REPLICATES, hc / N_REPLICATES
        rows.append({"spread": spread, "v": v, "mean_diff": float(np.mean(eff)),
                     "power_A_bootstrap_n225": pa, "power_B_mcnemar_n225": pb,
                     "power_C_crossfit_n475": pc,
                     "mean_discordant_pairs": float(np.mean(nds))})
        print(f"  {spread:7.2f} {v['easy']:8.3f} {v['hard']:8.3f} "
              f"{np.mean(eff):+11.4f} {np.mean(nds):8.0f} "
              f"{pa:8.1%} {pb:9.1%} {pc:10.1%}")

    null = rows[0]["power_A_bootstrap_n225"]
    print(f"\n  FALSE POSITIVE RATE at zero effect: {null:.1%}")
    print(f"  (a 95% one-sided criterion should sit near 5%)")

    print(f"  false positives at zero effect: A={rows[0]['power_A_bootstrap_n225']:.1%}  "
          f"B={rows[0]['power_B_mcnemar_n225']:.1%}  "
          f"C={rows[0]['power_C_crossfit_n475']:.1%}")
    ok80 = [r for r in rows if r["power_A_bootstrap_n225"] >= 0.80]
    for key, label in (("power_B_mcnemar_n225", "B (McNemar, n=225)"),
                       ("power_C_crossfit_n475", "C (cross-fit, n=475)")):
        hit = [r for r in rows if r[key] >= 0.80]
        print(f"  80% power for {label:24s}: "
              + (f"spread>={hit[0]['spread']:.2f}" if hit else "NEVER"))
    print()
    if ok80:
        s = ok80[0]
        print(f"  80% power first reached at spread={s['spread']:.2f}: "
              f"v(easy)={s['v']['easy']:.3f} vs v(hard)={s['v']['hard']:.3f}")
        print("  The criterion is attainable, but only for effects at least")
        print("  this strong. Compare against what calibration actually shows")
        print("  before deciding the evaluation set is worth spending.")
    else:
        print("  80% power is NOT reached at ANY simulated effect size,")
        print("  including ones stronger than anything observed.")
        print("  The 'beats random' criterion cannot be met at n=225.")
        print("  Running it would spend the only untouched evaluation set on a")
        print("  test that is designed to fail.")
    OUT.write_text(json.dumps({"replicates": N_REPLICATES, "rows": rows}, indent=1))
    print(f"\n  wrote {OUT.relative_to(ROOT)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
