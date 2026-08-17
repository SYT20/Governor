#!/usr/bin/env python3
"""E0024 — the Governor on LiveCodeBench sample allocation.

The architecture is REUSED, not rewritten: `governor.phase4.softbudget` supplies
the predictor, the Lagrangian, the myopic baseline and the runtime budget
enforcement, unchanged from the MATH work. Only the family adapter is new. If
the controller had needed modifying to accept a different allocation axis, it
was overfitted to token budgets.

Gate passed first: E0023 measured a +0.0573 ceiling on this axis.
"""
from __future__ import annotations

import argparse
import json
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
from governor.phase4.lcbdata import FEATURE_NAMES, feature_vector, split_mask  # noqa: E402
from governor.phase4.softbudget import (  # noqa: E402
    enforced_alloc, fit_predictors, governor_alloc, myopic_alloc, tune,
)

TOKENS = "exact tokenizer count over published LiveCodeBench generations"


def env_fixed(U, C, B):
    pts = sorted((float(C[:, j].mean()), float(U[:, j].mean()))
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


def oracle(U, C, B):
    b = -1.0
    for lam in np.concatenate([[0.0], np.geomspace(1e-7, 1e-1, 500)]):
        i = np.argmax(U - lam * C, axis=1)
        r = np.arange(len(i))
        if float(C[r, i].mean()) <= B + 1e-9:
            b = max(b, float(U[r, i].mean()))
    return b


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    d = pickle.load(open("results/lcb_samples.pkl", "rb"))
    rows, U, C = d["rows"], d["U"], d["C"]
    levels = list(range(1, U.shape[1] + 1))
    qids = [r["qid"] for r in rows]

    # problem statements for features
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("livecodebench/submissions",
                        "Gemini-Pro-1.5 (May)/Scenario.codegeneration_10_0.2_eval_all.json",
                        repo_type="dataset")
    meta = {r["question_id"]: r for r in json.load(open(p))}
    X = np.array([feature_vector(meta[q]["question_content"],
                                 meta[q].get("difficulty", ""),
                                 meta[q].get("platform", ""),
                                 meta[q].get("starter_code", "")) for q in qids], float)
    cal = split_mask(qids)
    ev = ~cal
    print("=" * 88)
    print("E0024  GOVERNOR ON LIVECODEBENCH SAMPLE ALLOCATION")
    print("=" * 88)
    print(f"  {len(qids)} problems, k in 1..{levels[-1]}; "
          f"{int(cal.sum())} calibration / {int(ev.sum())} evaluation "
          f"(frozen by question-id hash)")
    print(f"  architecture REUSED from softbudget.py; only the adapter is new")

    # operating point from the CALIBRATION half, by ceiling
    fine = np.linspace(C[:, 0].mean() * 1.02, C[:, -1].mean() * 0.85, 40)
    b_star = max((float(oracle(U[cal], C[cal], b) - env_fixed(U[cal], C[cal], b)),
                  float(b)) for b in fine)[1]
    print(f"  B* = {b_star:.0f} expected tokens (calibration ceiling)")

    lp = fit_predictors(X[cal], U[cal], C[cal], levels, kind="logistic",
                        calibrate="isotonic")
    Qc, Tc = lp.predict(X[cal])
    Qe, Te = lp.predict(X[ev])
    lam = tune(lambda k: governor_alloc(Qc, Tc, k),
               np.concatenate([[0.0], np.geomspace(1e-7, 5e-2, 400)]), C[cal], b_star)
    tau = tune(lambda k: myopic_alloc(Qc, k), np.linspace(0, 1, 401), C[cal], b_star)
    Ue, Ce = U[ev], C[ev]
    n = len(Ue)
    r = np.arange(n)
    res = np.percentile(C[cal], 99, axis=0)
    gi = enforced_alloc(range(n), lambda i: governor_alloc(Qe, Te, lam)[i], Ce,
                        levels, b_star, prompt_tokens=0, reserve=res)
    mi = enforced_alloc(range(n), lambda i: myopic_alloc(Qe, tau)[i], Ce,
                        levels, b_star, prompt_tokens=0, reserve=res)
    ug, um = float(Ue[r, gi].mean()), float(Ue[r, mi].mean())
    cg, cm = float(Ce[r, gi].mean()), float(Ce[r, mi].mean())
    fx = env_fixed(Ue, Ce, cg)
    orc = oracle(Ue, Ce, cg)
    print(f"\n  {'policy':<14}{'U':>8}{'tokens':>9}{'mean k':>9}")
    for nm, idx, u, c in (("GOVERNOR", gi, ug, cg), ("myopic", mi, um, cm)):
        print(f"  {nm:<14}{u:>8.4f}{c:>9.0f}{np.mean([levels[j] for j in idx]):>9.2f}")
    print(f"  {'fixed@matched':<14}{fx:>8.4f}{cg:>9.0f}")
    print(f"  {'oracle':<14}{orc:>8.4f}{cg:>9.0f}")

    rng = np.random.default_rng(0)
    df, dm = [], []
    for _ in range(a.boot):
        s = rng.integers(0, n, n)
        u = float(Ue[s, gi[s]].mean())
        df.append(u - env_fixed(Ue[s], Ce[s], float(Ce[s, gi[s]].mean())))
        dm.append(u - float(Ue[s, mi[s]].mean()))
    def ci(v):
        v = np.array(v)
        return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    m1, l1, h1 = ci(df)
    m2, l2, h2 = ci(dm)
    dis = gi != mi
    b_ = int(((Ue[r, gi] == 1) & (Ue[r, mi] == 0) & dis).sum())
    c_ = int(((Ue[r, gi] == 0) & (Ue[r, mi] == 1) & dis).sum())
    pmc = mcnemar(b_, c_)
    print(f"\n  BOOTSTRAP ({n} problems, {a.boot} resamples)")
    print(f"    PRIMARY   GOV - fixed @ matched cost: {m1:+.4f} [{l1:+.4f}, {h1:+.4f}]"
          f"  {'BEATS' if l1 > 0 else 'LOSES' if h1 < 0 else 'not separable'}")
    print(f"    SECONDARY GOV - myopic              : {m2:+.4f} [{l2:+.4f}, {h2:+.4f}]"
          f"  {'BEATS' if l2 > 0 else 'LOSES' if h2 < 0 else 'not separable'}")
    print(f"    paired: {int(dis.sum())} disagreements, G {b_} / M {c_}, "
          f"McNemar p={pmc:.4f}")
    sd = float((Ue[r, gi] - Ue[r, mi]).std(ddof=1))
    need = (1.96 * sd / m2) ** 2 if m2 > 0 else float("inf")
    print(f"    power: sd={sd:.4f} -> N for GOV-myopic CI>0 = "
          f"{'unattainable' if need == float('inf') else f'{need:,.0f}'}")

    traps = {"oracle_leakage": oracle_leakage(FEATURE_NAMES),
             "exact_token_counts": exact_token_counts(TOKENS),
             "split_leakage": split_leakage([q for q, x in zip(qids, cal) if x],
                                            [q for q, x in zip(qids, ev) if x]),
             "budget_adherence": budget_adherence(cg, b_star, baseline_cost=cg),
             "secret_scan": secret_scan()}
    red = [k for k, (ok, _) in traps.items() if not ok]
    for k, (ok, dt) in traps.items():
        print(f"    {'GREEN' if ok else '  RED'}  {k:<20} {dt}")
    verdict = ("PASS" if (l1 > 0 and not red)
               else "FAIL" if h1 < 0 else "INCONCLUSIVE")
    print(f"\n  VERDICT: {verdict}")

    spec = ExperimentSpec(
        exp_id="E0024-lcb-governor",
        title="Governor on LiveCodeBench sample allocation (architecture reused)",
        model="Gemini-Pro-1.5 (May) generations, published by LiveCodeBench",
        budget={"axis": "samples k in 1..10", "B_star": float(b_star),
                "contract": "SOFT_EXPECTED_BUDGET + hard runtime cap",
                "charged": TOKENS},
        seeds={"split": "sha256(question_id) parity", "bootstrap": 0},
        split={"calibration": int(cal.sum()), "evaluation": n},
        metric="pass@k-style utility at matched realised cost; primary = "
               "Governor minus the randomised fixed envelope at the Governor's "
               "own cost; secondary = Governor minus myopic",
        params={"features": list(FEATURE_NAMES), "lambda": float(lam),
                "tau": float(tau), "predictor": "logistic+isotonic"},
        notes="softbudget.py reused unchanged; only the family adapter is new.")
    run = ExperimentRun(spec, overwrite=True)
    for k in range(n):
        run.append({"qid": [q for q, x in zip(qids, ev) if x][k],
                    "gov_k": levels[gi[k]], "myopic_k": levels[mi[k]],
                    "gov_pass": int(Ue[k, gi[k]]), "myopic_pass": int(Ue[k, mi[k]]),
                    "gov_tokens": float(Ce[k, gi[k]])})
    run.finalize(summary={"governor_U": ug, "myopic_U": um,
                          "fixed_matched": fx, "oracle": orc,
                          "governor_tokens": cg, "B_star": float(b_star),
                          "primary_mean": m1, "primary_lo": l1, "primary_hi": h1,
                          "secondary_mean": m2, "secondary_lo": l2, "secondary_hi": h2,
                          "mcnemar_p": pmc, "n_disagree": int(dis.sum()),
                          "n_required_gov_vs_myopic": float(need),
                          "verdict": verdict},
                 metrics={"levels": levels}, traps=traps, verdict=verdict)
    print(f"\n  recorded: experiments/E0024-lcb-governor/")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
