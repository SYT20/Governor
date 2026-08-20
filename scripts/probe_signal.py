#!/usr/bin/env python3
"""E0018 — does early-reasoning state carry allocation signal, and did E0017
measure the question-only signal with the wrong instrument?

Two findings, and the second corrects the previous experiment.

1. PROBE FEATURES ADD SIGNAL ON MATH. Read off a cheap 500-token probe --
   whether the trace terminated, how many backtracking markers it contains, how
   long it ran -- AUC for "will more budget help this item" rises from 0.671
   (question only) to 0.741 (question + probe).

2. E0017 USED THE WRONG METRIC. It fitted ridge regressions to a target in
   {-1,0,+1} and reported R^2 near zero, which I read as "no signal". On the
   same features and the same calibration split, a logistic model scores AUC
   0.613-0.671. R^2 near zero on a sparse near-binary target is a property of
   the loss, not a statement about discriminability. The E0017 FAIL stands as a
   measured allocation result; its DIAGNOSIS -- "the features carry no signal"
   -- was wrong and is retracted here.

GPQA is different and shows no usable signal on either feature set (0.37-0.57,
several below chance). Whatever makes a GPQA item need more tokens is not
visible in the question or in a 500-token probe.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.harness.ledger import ExperimentRun, ExperimentSpec  # noqa: E402
from governor.harness.traps import (  # noqa: E402
    exact_token_counts, oracle_leakage, secret_scan, split_leakage,
)
from governor.phase4.s1data import FEATURE_NAMES, feature_vector, load  # noqa: E402

PROBE = 500
PROBE_FEATURES = ("p_tokens", "p_terminated", "p_wait", "p_alt", "p_eq",
                  "p_boxed", "p_lines", "p_digits", "p_recheck")
TOKEN_SOURCE = "simplescaling/s1-32B tokenizer (exact)"


def probe_feats(s: str, ntok: int) -> list[float]:
    """Observable at the end of the probe. Nothing here reads a later budget.

    Answer STABILITY across budgets would be the strongest feature available and
    it is deliberately absent: comparing the probe answer to a higher-budget
    answer reads the outcome the allocator is trying to predict.
    """
    low = s.lower()
    return [float(ntok), float("</think>" in s or "<|im_start|>answer" in s),
            float(low.count("wait")),
            float(low.count("alternatively") + low.count("hmm")),
            float(s.count("=")), float("boxed" in s), float(s.count("\n")),
            float(sum(c.isdigit() for c in s)),
            float(low.count("recheck") + low.count("double-check"))]


def main() -> int:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import (
        KFold, StratifiedKFold, cross_val_predict,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d_all = pickle.load(open("results/s1_text.pkl", "rb"))
    print("=" * 88)
    print("E0018  PROBE-STATE SIGNAL — AUC, and a retraction of the E0017 diagnosis")
    print("=" * 88)
    rows = []
    for bench in ("math", "gpqa"):
        d = d_all[bench]
        ids = sorted(d[PROBE])
        items, _ = load(bench, budgets=[PROBE])
        pr = {i.item_id: i.prompt for i in items}
        Xq = np.array([feature_vector(pr[i]) for i in ids], float)
        Xp = np.array([probe_feats(d[PROBE][i]["text"], d[PROBE][i]["tokens"])
                       for i in ids], float)
        Xb = np.hstack([Xq, Xp])
        cal = np.array([int(i[-5:]) % 2 == 0 for i in ids])
        print(f"\n  {bench.upper()}  ({int(cal.sum())} calibration items)")
        print(f"    {'target':>12}{'base':>7}{'AUC q':>8}{'AUC probe':>11}"
              f"{'AUC both':>10}{'ridge R2 q':>12}")
        for hb in (1000, 2000, 4000, 8000):
            gain = np.array([d[hb][i]["correct"] - d[PROBE][i]["correct"]
                             for i in ids], float)
            y = (gain > 0).astype(int)[cal]
            if y.sum() < 10 or y.sum() == len(y):
                continue
            aucs = []
            for X in (Xq[cal], Xp[cal], Xb[cal]):
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000))
                p = cross_val_predict(m, X, y,
                                      cv=StratifiedKFold(5, shuffle=True,
                                                         random_state=0),
                                      method="predict_proba")[:, 1]
                aucs.append(float(roc_auc_score(y, p)))
            yr = gain[cal]
            oof = cross_val_predict(Ridge(alpha=1.0), Xq[cal], yr,
                                    cv=KFold(5, shuffle=True, random_state=0))
            ss = float(np.sum((yr - yr.mean()) ** 2)) or 1e-12
            r2 = 1.0 - float(np.sum((yr - oof) ** 2)) / ss
            rows.append({"benchmark": bench, "target_budget": hb,
                         "base_rate": float(y.mean()), "auc_question": aucs[0],
                         "auc_probe": aucs[1], "auc_both": aucs[2],
                         "ridge_r2_question": r2})
            print(f"    gain@{hb:<7}{y.mean():>7.3f}{aucs[0]:>8.3f}"
                  f"{aucs[1]:>11.3f}{aucs[2]:>10.3f}{r2:>12.3f}")

    math_best = max(r["auc_both"] for r in rows if r["benchmark"] == "math")
    gpqa_best = max(r["auc_both"] for r in rows if r["benchmark"] == "gpqa")
    verdict = "PROBE-SIGNAL-ON-MATH" if math_best >= 0.65 else "NO-PROBE-SIGNAL"
    print(f"\n  best AUC(question+probe): math {math_best:.3f}, gpqa {gpqa_best:.3f}")
    print(f"  VERDICT: {verdict}")
    print("\n  RETRACTION: E0017 reported ridge R2 near zero and I read that as")
    print("  'the features carry no signal'. On the same features and split a")
    print("  logistic model scores AUC 0.613-0.671 question-only. R2 near zero")
    print("  on a sparse near-binary target is a property of the loss, not a")
    print("  statement about discriminability. The E0017 allocation FAIL stands;")
    print("  its diagnosis does not.")

    spec = ExperimentSpec(
        exp_id="E0018-probe-signal",
        title="Early-reasoning probe features: signal check and E0017 retraction",
        model="s1-32B via simplescaling/results",
        budget={"probe_tokens": PROBE, "charged": TOKEN_SOURCE},
        seeds={"split": "doc_id parity", "cv": 0},
        split={"rule": "doc_id parity; calibration half only"},
        metric="cross-validated AUC for P(gain>0) from question-only, "
               "probe-only and combined features; ridge R2 reported alongside "
               "to show the metric artefact",
        params={"probe_features": list(PROBE_FEATURES),
                "question_features": list(FEATURE_NAMES)},
        notes="Answer stability across budgets is deliberately excluded: it "
              "reads the outcome the allocator is trying to predict.")
    run = ExperimentRun(spec, overwrite=True)
    for r in rows:
        run.append(r)
    traps = {"oracle_leakage": oracle_leakage(FEATURE_NAMES + PROBE_FEATURES),
             "exact_token_counts": exact_token_counts(TOKEN_SOURCE),
             "split_leakage": split_leakage(["cal"], ["ev"]),
             "secret_scan": secret_scan()}
    run.finalize(summary={"verdict": verdict, "best_auc_math": math_best,
                          "best_auc_gpqa": gpqa_best,
                          "retracts": "E0017 diagnosis (not its result)"},
                 metrics={"rows": rows}, traps=traps, verdict=verdict)
    print(f"\n  recorded: experiments/E0018-probe-signal/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
