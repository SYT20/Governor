#!/usr/bin/env python3
"""E0025 — FINAL EXPERIMENT. Early-generation signal, and a hard stop.

Three axes have now shown a real oracle ceiling that the Governor cannot reach
from PROBLEM TEXT. E0024's diagnostic was decisive: 45 allocation disagreements
produced zero outcome differences. The hypothesis under test here is the one the
evidence points at -- that the missing signal appears only after you start
solving.

DESIGN. Every problem is given a PROBE of two samples, always, and charged for
them. Features are read from those two generations and from the problem text.
The Governor then decides how many further samples the problem gets.

WHAT IS AND IS NOT OBSERVABLE. Sample correctness on the hidden tests is NOT
available at runtime, so `graded_list` never enters a feature. What is available
is what the generations look like: their length, whether they parse as a code
block, how long the extracted program is, and -- the self-consistency signal --
whether the two probe samples produced the SAME program. Agreement between
independent samples is the classic cheap proxy for confidence and it costs
nothing beyond the probe already paid for.

HARD STOP. If the primary CI crosses zero, the allocation claim is recorded as
unresolved and the project stops. No further benchmarks.
"""
from __future__ import annotations

import argparse
import difflib
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
from governor.phase4.lcbdata import feature_vector as qfeat, split_mask  # noqa: E402
from governor.phase4.softbudget import (  # noqa: E402
    enforced_alloc, fit_predictors, governor_alloc, myopic_alloc, tune,
)

PROBE = 2
TOKENS = "exact tokenizer count over published LiveCodeBench generations"
PROBE_FEATURES = ("pr_tokens", "pr_len_ratio", "pr_has_code", "pr_code_lines",
                  "pr_identical", "pr_similarity", "pr_has_loop", "pr_has_try",
                  "pr_mean_chars")


def probe_feats(codes, outs, toks) -> list[float]:
    """Observable after the probe. Correctness is deliberately absent."""
    c0, c1 = (codes + ["", ""])[:2]
    o0, o1 = (outs + ["", ""])[:2]
    sim = difflib.SequenceMatcher(None, c0[:2000], c1[:2000]).ratio() if c0 and c1 else 0.0
    return [float(sum(toks[:2])),
            float(len(o0) / max(len(o1), 1)),
            float("```" in o0),
            float(c0.count("\n") + 1),
            float(c0.strip() == c1.strip()),
            float(sim),
            float("for " in c0 or "while " in c0),
            float("try:" in c0),
            float((len(c0) + len(c1)) / 2)]


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
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("simplescaling/s1-32B")
    p = hf_hub_download("livecodebench/submissions",
                        "Gemini-Pro-1.5 (May)/Scenario.codegeneration_10_0.2_eval_all.json",
                        repo_type="dataset")
    raw = json.load(open(p))
    d = pickle.load(open("results/lcb_samples.pkl", "rb"))
    keep = {r["qid"] for r in d["rows"]}
    recs = [r for r in raw if r["question_id"] in keep]
    K = 10
    qids = [r["question_id"] for r in recs]
    G = np.array([[bool(x) for x in r["graded_list"][:K]] for r in recs])
    T = np.array([[len(tok.encode(str(o))) for o in r["output_list"][:K]]
                  for r in recs], float)
    # levels are k = PROBE..K; the probe is always paid for
    levels = list(range(PROBE, K + 1))
    U = np.array([[G[i, :k].any() for k in levels] for i in range(len(recs))], float)
    C = np.array([[T[i, :k].sum() for k in levels] for i in range(len(recs))], float)
    X = np.array([np.concatenate([
        qfeat(r["question_content"], "", r.get("platform", ""), r.get("starter_code", "")),
        probe_feats(r["code_list"][:PROBE], r["output_list"][:PROBE], list(T[i]))])
        for i, r in enumerate(recs)], float)
    names = tuple(list(__import__("governor.phase4.lcbdata", fromlist=["x"]).FEATURE_NAMES)
                  + list(PROBE_FEATURES))
    cal = split_mask(qids)
    ev = ~cal
    n = int(ev.sum())
    print("=" * 88)
    print("E0025  FINAL — EARLY-GENERATION SIGNAL ON LIVECODEBENCH")
    print("=" * 88)
    print(f"  {len(qids)} problems, probe={PROBE} samples (always paid), "
          f"k in {levels[0]}..{levels[-1]}")
    print(f"  {int(cal.sum())} calibration / {n} evaluation; features = problem "
          f"text + probe generations, NO correctness")

    fine = np.linspace(C[:, 0].mean() * 1.02, C[:, -1].mean() * 0.85, 40)
    b_star = max((float(oracle(U[cal], C[cal], b) - env_fixed(U[cal], C[cal], b)),
                  float(b)) for b in fine)[1]
    lp = fit_predictors(X[cal], U[cal], C[cal], levels, kind="logistic",
                        calibrate="isotonic")
    Qc, Tc = lp.predict(X[cal])
    Qe, Te = lp.predict(X[ev])
    lam = tune(lambda k: governor_alloc(Qc, Tc, k),
               np.concatenate([[0.0], np.geomspace(1e-7, 5e-2, 400)]), C[cal], b_star)
    tau = tune(lambda k: myopic_alloc(Qc, k), np.linspace(0, 1, 401), C[cal], b_star)
    Ue, Ce = U[ev], C[ev]
    r = np.arange(n)
    res = C[cal].mean(axis=0)
    gi = enforced_alloc(range(n), lambda i: governor_alloc(Qe, Te, lam)[i], Ce,
                        levels, b_star, prompt_tokens=0, reserve=res)
    mi = enforced_alloc(range(n), lambda i: myopic_alloc(Qe, tau)[i], Ce,
                        levels, b_star, prompt_tokens=0, reserve=res)
    ug, cg = float(Ue[r, gi].mean()), float(Ce[r, gi].mean())
    um = float(Ue[r, mi].mean())
    fx, orc = env_fixed(Ue, Ce, cg), oracle(Ue, Ce, cg)
    print(f"\n  B* = {b_star:.0f}")
    print(f"  {'policy':<16}{'U':>8}{'tokens':>9}{'mean k':>9}")
    print(f"  {'GOVERNOR':<16}{ug:>8.4f}{cg:>9.0f}"
          f"{np.mean([levels[j] for j in gi]):>9.2f}")
    print(f"  {'myopic':<16}{um:>8.4f}{float(Ce[r, mi].mean()):>9.0f}"
          f"{np.mean([levels[j] for j in mi]):>9.2f}")
    print(f"  {'fixed@matched':<16}{fx:>8.4f}{cg:>9.0f}")
    print(f"  {'oracle':<16}{orc:>8.4f}{cg:>9.0f}   ceiling {orc - fx:+.4f}")

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
    print(f"\n  BOOTSTRAP ({n} problems, {a.boot} resamples)")
    print(f"    PRIMARY   GOV - fixed @ matched cost: {m1:+.4f} [{l1:+.4f}, {h1:+.4f}]"
          f"  {'BEATS' if l1 > 0 else 'LOSES' if h1 < 0 else 'not separable'}")
    print(f"    SECONDARY GOV - myopic              : {m2:+.4f} [{l2:+.4f}, {h2:+.4f}]")
    print(f"    paired: {int(dis.sum())} disagreements, G {b_} / M {c_}, "
          f"McNemar p={mcnemar(b_, c_):.4f}")
    print(f"    probe predictors (Brier by level): "
          f"{min(lp.cv_r2_q.values()):.4f}..{max(lp.cv_r2_q.values()):.4f}")

    traps = {"oracle_leakage": oracle_leakage(names),
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
    if verdict != "PASS":
        print("    HARD STOP. The early-generation signal was the final proposed")
        print("    mechanism. The allocation claim is recorded as unresolved and")
        print("    no further benchmarks will be run.")

    spec = ExperimentSpec(
        exp_id="E0025-lcb-probe",
        title="FINAL: early-generation signal for sample allocation",
        model="Gemini-Pro-1.5 (May) generations, published by LiveCodeBench",
        budget={"axis": f"samples k in {levels[0]}..{levels[-1]}, probe={PROBE} always paid",
                "B_star": float(b_star), "charged": TOKENS},
        seeds={"split": "sha256(question_id) parity", "bootstrap": 0},
        split={"calibration": int(cal.sum()), "evaluation": n},
        metric="pass-within-k at matched realised cost; primary = Governor minus "
               "the randomised fixed envelope at the Governor's own cost",
        params={"features": list(names), "probe_samples": PROBE,
                "lambda": float(lam), "tau": float(tau)},
        notes="Sample correctness never enters a feature; only generation text "
              "and cross-sample agreement. Hard stop on a non-PASS.")
    run = ExperimentRun(spec, overwrite=True)
    for k in range(n):
        run.append({"qid": [q for q, x in zip(qids, ev) if x][k],
                    "gov_k": levels[gi[k]], "myopic_k": levels[mi[k]],
                    "gov_pass": int(Ue[k, gi[k]]), "myopic_pass": int(Ue[k, mi[k]]),
                    "tokens": float(Ce[k, gi[k]])})
    run.finalize(summary={"governor_U": ug, "myopic_U": um, "fixed_matched": fx,
                          "oracle": orc, "ceiling": orc - fx,
                          "governor_tokens": cg, "B_star": float(b_star),
                          "primary_mean": m1, "primary_lo": l1, "primary_hi": h1,
                          "secondary_mean": m2, "secondary_lo": l2, "secondary_hi": h2,
                          "n_disagree": int(dis.sum()), "verdict": verdict,
                          "hard_stop": verdict != "PASS"},
                 metrics={"brier_by_level": lp.cv_r2_q}, traps=traps, verdict=verdict)
    print(f"\n  recorded: experiments/E0025-lcb-probe/")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
