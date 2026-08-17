#!/usr/bin/env python3
"""E0021 — the SAME MATH test, with the budget actually enforced.

E0019's "+0.0282 BEATS" was withdrawn because the Governor spent 973 tokens
against a budget of 846. Nothing here changes except that the constraint is now
ENFORCED at runtime rather than merely tuned for on calibration, and the primary
comparison is at MATCHED REALISED COST.

Frozen and untouched: benchmark, split, B*, features, predictor
(logistic+isotonic), lambda/tau tuning on calibration only, fixed baseline,
oracle, soft-budget contract.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.harness.ledger import ExperimentRun, ExperimentSpec  # noqa: E402
from governor.harness.traps import (  # noqa: E402
    budget_adherence, exact_token_counts, oracle_leakage, secret_scan,
    split_leakage,
)
from governor.phase4.s1data import FEATURE_NAMES, feature_vector, load  # noqa: E402
from governor.phase4.softbudget import (  # noqa: E402
    enforced_alloc, fit_predictors, governor_alloc, myopic_alloc, tune,
)

KIND, CALIB, TOKEN_SOURCE = "logistic", "isotonic", "simplescaling/s1-32B tokenizer (exact)"


def env_fixed(C, T, B):
    pts = sorted((float(T[:, j].mean()), float(C[:, j].mean()))
                 for j in range(C.shape[1]))
    best = None
    for (c0, u0), (c1, u1) in zip(pts, pts[1:]):
        if c0 <= B <= c1 and c1 > c0:
            w = (B - c0) / (c1 - c0)
            best = max(best if best is not None else -1.0, u0 + w * (u1 - u0))
    for c, u in pts:
        if c <= B + 1e-9:
            best = max(best if best is not None else -1.0, u)
    return float(best if best is not None else pts[0][1])


def oracle_at(C, T, B):
    b = -1.0
    for lam in np.concatenate([[0.0], np.geomspace(1e-7, 1e-1, 400)]):
        i = np.argmax(C - lam * T, axis=1)
        r = np.arange(len(i))
        if float(T[r, i].mean()) <= B + 1e-9:
            b = max(b, float(C[r, i].mean()))
    return b


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="math", choices=("math", "gpqa"))
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    exp = f"E0021-enforced-{a.bench}"

    data = pickle.load(open("results/s1_caps_exact.pkl", "rb"))[a.bench]
    levels = sorted(data)
    ids = sorted(data[levels[0]])
    items, _ = load(a.bench, budgets=[levels[0]])
    pr = {i.item_id: i.prompt for i in items}
    C = np.array([[data[b][i]["correct"] for b in levels] for i in ids], float)
    T = np.array([[data[b][i]["tokens"] for b in levels] for i in ids], float)
    X = np.array([feature_vector(pr[i]) for i in ids], float)
    cal = np.array([int(i[-5:]) % 2 == 0 for i in ids])
    ev = ~cal
    ev_ids = [i for i, e in zip(ids, ev) if e]

    print("=" * 92)
    print(f"E0021  BUDGET-ENFORCED GOVERNOR — {a.bench.upper()}")
    print("=" * 92)

    lo, hi = float(T[:, 0].mean()), float(T[:, -1].mean())
    fine = np.linspace(lo * 1.02, hi * 0.9, 60)
    b_star = max((float(oracle_at(C[cal], T[cal], b) - env_fixed(C[cal], T[cal], b)),
                  float(b)) for b in fine)[1]
    print(f"  B* = {b_star:.0f} (chosen on calibration by ceiling)")

    lp = fit_predictors(X[cal], C[cal], T[cal], levels, kind=KIND, calibrate=CALIB)
    Qc, Tc = lp.predict(X[cal])
    Qe, Te = lp.predict(X[ev])
    lam = tune(lambda k: governor_alloc(Qc, Tc, k),
               np.concatenate([[0.0], np.geomspace(1e-7, 5e-2, 400)]), T[cal], b_star)
    tau = tune(lambda k: myopic_alloc(Qc, k), np.linspace(0, 1, 401), T[cal], b_star)
    print(f"  lambda={lam:.3e} tau={tau:.3f} (calibration only; never retuned)")

    Ce, Tt = C[ev], T[ev]
    n = len(ev_ids)
    r = np.arange(n)
    g_raw = governor_alloc(Qe, Te, lam)
    m_raw = myopic_alloc(Qe, tau)
    g_enf = enforced_alloc(range(n), lambda i: g_raw[i], Tt, levels, b_star)
    m_enf = enforced_alloc(range(n), lambda i: m_raw[i], Tt, levels, b_star)

    res = {}
    for name, idx in (("governor_unenforced", g_raw), ("GOVERNOR", g_enf),
                      ("myopic_unenforced", m_raw), ("myopic", m_enf)):
        res[name] = {"U": float(Ce[r, idx].mean()),
                     "tokens": float(Tt[r, idx].mean()), "idx": idx}
    print(f"\n  {'policy':<22}{'U':>8}{'tokens':>9}{'vs B*':>9}")
    for k in ("governor_unenforced", "GOVERNOR", "myopic_unenforced", "myopic"):
        v = res[k]
        print(f"  {k:<22}{v['U']:>8.4f}{v['tokens']:>9.0f}"
              f"{v['tokens'] - b_star:>+9.0f}")
    fx_B = env_fixed(Ce, Tt, b_star)
    cg = res["GOVERNOR"]["tokens"]
    fx_match = env_fixed(Ce, Tt, cg)
    print(f"\n  fixed @ B*={b_star:.0f}          : {fx_B:.4f}")
    print(f"  fixed @ Governor cost {cg:.0f} : {fx_match:.4f}   <-- MATCHED, primary")
    print(f"  oracle @ Governor cost      : {oracle_at(Ce, Tt, cg):.4f}")

    rng = np.random.default_rng(0)
    dmatch, dmy = [], []
    gi, mi = res["GOVERNOR"]["idx"], res["myopic"]["idx"]
    for _ in range(a.boot):
        s = rng.integers(0, n, n)
        ug = float(Ce[s, gi[s]].mean())
        dmatch.append(ug - env_fixed(Ce[s], Tt[s], float(Tt[s, gi[s]].mean())))
        dmy.append(ug - float(Ce[s, mi[s]].mean()))
    def ci(v):
        v = np.array(v)
        return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    m1, l1, h1 = ci(dmatch)
    m2, l2, h2 = ci(dmy)
    print(f"\n  BOOTSTRAP ({n} items, {a.boot} resamples)")
    print(f"    PRIMARY   GOV - fixed @ MATCHED cost: {m1:+.4f} [{l1:+.4f}, {h1:+.4f}]"
          f"  {'BEATS' if l1 > 0 else 'LOSES' if h1 < 0 else 'not separable'}")
    print(f"    SECONDARY GOV - myopic              : {m2:+.4f} [{l2:+.4f}, {h2:+.4f}]"
          f"  {'BEATS' if l2 > 0 else 'LOSES' if h2 < 0 else 'not separable'}")

    dis = gi != mi
    b_ = int(((Ce[r, gi] == 1) & (Ce[r, mi] == 0) & dis).sum())
    c_ = int(((Ce[r, gi] == 0) & (Ce[r, mi] == 1) & dis).sum())
    p_mc = mcnemar(b_, c_)
    print(f"    paired: {int(dis.sum())} disagreements, G-wins {b_}, M-wins {c_}, "
          f"McNemar p={p_mc:.4f}")

    traps = {"oracle_leakage": oracle_leakage(FEATURE_NAMES),
             "exact_token_counts": exact_token_counts(TOKEN_SOURCE),
             "split_leakage": split_leakage([i for i, x in zip(ids, cal) if x], ev_ids),
             "budget_adherence": budget_adherence(cg, b_star, baseline_cost=cg),
             "secret_scan": secret_scan()}
    red = [k for k, (ok, _) in traps.items() if not ok]
    for k, (ok, d) in traps.items():
        print(f"    {'GREEN' if ok else '  RED'}  {k:<22} {d}")
    passed = l1 > 0 and not red
    verdict = "PASS" if passed else ("FAIL" if h1 < 0 else "INCONCLUSIVE")
    print(f"\n  VERDICT: {verdict}")

    spec = ExperimentSpec(
        exp_id=exp, title=f"Budget-enforced Governor at matched realised cost ({a.bench})",
        model="s1-32B via simplescaling/results",
        budget={"contract": "SOFT_EXPECTED_BUDGET + hard runtime cap",
                "B_star": float(b_star), "charged": TOKEN_SOURCE},
        seeds={"split": "doc_id parity", "bootstrap": 0},
        split={"calibration_items": int(cal.sum()), "evaluation_items": n},
        metric="primary = Governor minus the fixed envelope AT THE GOVERNOR'S "
               "OWN REALISED COST; secondary = Governor minus myopic; 95% CI "
               "from a bootstrap over evaluation items with the controller frozen",
        params={"predictor": f"{KIND}+{CALIB}", "lambda": float(lam),
                "tau": float(tau), "enforcement": "sequential, reserve level, "
                                                  "charge actual"},
        notes="E0019 is withdrawn: it spent 15% over budget and was scored "
              "against a baseline at the nominal budget.")
    run = ExperimentRun(spec, overwrite=True)
    for k in range(n):
        run.append({"item_id": ev_ids[k], "gov_level": levels[gi[k]],
                    "myopic_level": levels[mi[k]],
                    "gov_correct": int(Ce[k, gi[k]]), "myopic_correct": int(Ce[k, mi[k]]),
                    "gov_tokens": float(Tt[k, gi[k]]), "myopic_tokens": float(Tt[k, mi[k]])})
    run.finalize(summary={"benchmark": a.bench, "B_star": float(b_star),
                          "governor_U": res["GOVERNOR"]["U"],
                          "governor_tokens": cg,
                          "unenforced_tokens": res["governor_unenforced"]["tokens"],
                          "fixed_at_matched_cost": fx_match,
                          "primary_mean": m1, "primary_lo": l1, "primary_hi": h1,
                          "secondary_mean": m2, "secondary_lo": l2, "secondary_hi": h2,
                          "mcnemar_p": p_mc, "n_disagree": int(dis.sum()),
                          "verdict": verdict},
                 metrics={"policies": {k: {kk: vv for kk, vv in v.items() if kk != "idx"}
                                       for k, v in res.items()}},
                 traps=traps, verdict=verdict)
    print(f"\n  recorded: experiments/{exp}/")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
