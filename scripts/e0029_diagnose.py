#!/usr/bin/env python3
"""E0029 -- SECONDARY analysis. Why did nothing separate at the allocation point?

THIS IS NOT THE PREREGISTERED RESULT. The preregistered run stopped at gate 3:

    observable ceiling (evaluation)          +0.1378
    allocation-point AUC                      0.415   (permutation null 0.592)
    pooled AUC                                0.646
    position-only baseline                    0.726

That stands. Everything here is a diagnostic run afterwards to understand it,
and anything it finds is a hypothesis for a future preregistration, never a
result to report in place of the one above.

It answers two questions.

1. WAS THE MODEL CHOSEN ON THE WRONG CRITERION? e0029_analyse.py selects by
   POOLED cross-validated AUC and then gates on ALLOCATION-POINT AUC. Those are
   different objectives, and the pooled figure is largely positional -- it sat
   BELOW a baseline using only attempt_idx and attempts_left. Choosing a ranker
   by a positional metric and then asking it to rank problems at sample 1 is a
   protocol error. Selection here uses the allocation-point metric directly.
   Still calibration-only, so no evaluation data is touched either way.

2. DOES ANY SINGLE OBSERVABLE FEATURE DISCRIMINATE AT SAMPLE 1? If none does,
   the ceiling is real but invisible to this feature set, and a better model
   cannot help. That is a different problem from a weak learner and it has a
   different fix.

WHY THE MAXIMUM IS TESTED SEPARATELY. Scanning 25 features and reporting the
best one is 25 chances to be lucky. The null for "best of 25" is much higher
than the null for one pre-chosen feature, so this permutes the labels and
rebuilds the whole max-statistic each time. A feature that beats the single-
feature null but not the max null has not been shown to do anything.

Run:
    python scripts/e0029_diagnose.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from governor.execfeedback.richfeatures import decision_features
from governor.models.calibration import auc

import scripts.e0029_analyse as A

N_PERM = 4000

# Code-size features are in the thousands, binaries are 0/1 -- roughly 30000x
# apart. lbfgs does not converge on that, and every logistic fold in the run
# that reported "no model clears the null" raised ConvergenceWarning: those
# models never fitted. Trees are scale-free and were fine.
def _lr(**kw):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=20000, **kw))


# The signal that survived multiplicity was entirely code SIZE, and it is one
# signal wearing several names -- code_lines, code_chars and ast_nodes measure
# the same thing and correlate hard. Handing 25 features to a model with 34
# positives buries it. This family is small enough to be preregistered.
SIZE_FEATURES = ("code_lines", "code_chars", "ast_nodes", "ast_call_nodes",
                 "ast_depth", "n_loops", "has_recursion")


def allocation_point(byq, qids):
    """Rows where the spend is actually committed: sample 1 has failed, and the
    question is whether to buy more. Every position feature is identical here,
    so any separation is evidence about the problem itself."""
    X, y, g = A.decision_rows(byq, qids)
    if len(y) == 0:
        raise SystemExit("no decision rows")
    first = np.array([i for i in range(len(g)) if i == 0 or g[i - 1] != g[i]])
    return X, y, g, first


def univariate_scan(X0, y0, names):
    rows = []
    for j, nm in enumerate(names):
        col = X0[:, j]
        if np.allclose(col, col[0]):
            rows.append((nm, 0.5, 0.0, "constant"))
            continue
        a = float(auc(y0, col))
        rows.append((nm, max(a, 1.0 - a), a, "higher" if a >= 0.5 else "lower"))
    return sorted(rows, key=lambda r: -r[1])


def main() -> int:
    print("\nE0029 -- SECONDARY DIAGNOSTIC (the preregistered run stopped at gate 3)\n")

    byq, cal, ev, meta = A.load_joined()
    print(f"  problems {meta['problems']}  cal {meta['cal']}  eval {meta['eval']}")
    if (meta["cal"], meta["eval"]) != (A.EXPECTED_CAL, A.EXPECTED_EVAL):
        raise SystemExit(f"wrong dataset: split {meta['cal']}/{meta['eval']}")

    names = sorted(decision_features(byq[cal[0]][:1]))
    X, y, g, first = allocation_point(byq, cal)
    X0, y0 = X[first], y[first]
    n, pos = len(y0), int(y0.sum())
    print(f"  allocation point: n={n}  positives={pos} ({pos/n:.1%})\n")

    rng = np.random.default_rng(0)

    # ---- 1. can ANY single observable feature separate at sample 1? ---------
    print("  " + "=" * 66)
    print("  1. SINGLE-FEATURE SEPARATION AT SAMPLE 1")
    print("  " + "=" * 66)
    scan = univariate_scan(X0, y0, names)

    # Null for ONE pre-chosen feature, and for the BEST OF ALL of them.
    single_null, max_null = [], []
    cols = [j for j in range(X0.shape[1]) if not np.allclose(X0[:, j], X0[0, j])]
    for _ in range(N_PERM):
        yp = rng.permutation(y0)
        aucs = [abs(float(auc(yp, X0[:, j])) - 0.5) + 0.5 for j in cols]
        single_null.append(aucs[0])
        max_null.append(max(aucs))
    single_hi = float(np.percentile(single_null, 95))
    max_hi = float(np.percentile(max_null, 95))

    live = [r for r in scan if r[3] != "constant"]
    dead = [r for r in scan if r[3] == "constant"]
    if dead:
        print(f"  {len(dead)} of {len(scan)} features are constant at sample 1 "
              f"and carry nothing:")
        print("    " + ", ".join(r[0] for r in dead[:8])
              + (" ..." if len(dead) > 8 else ""))
        print()
    # With ONE attempt, best_/last_/mean_pub_frac are the same number by
    # definition, so identical AUCs among them are expected, not a bug.
    print(f"  {'feature':28s} {'AUC':>6}  direction")
    for nm, a, raw, d in live[:12]:
        flag = ""
        if a > max_hi:
            flag = "  <- clears the BEST-OF-25 null"
        elif a > single_hi:
            flag = "  <- clears single-feature null only (not multiplicity)"
        print(f"  {nm:28s} {a:6.3f}  {d}{flag}")
    print(f"\n  null, one pre-chosen feature (95th pct): {single_hi:.3f}")
    print(f"  null, best of {len(cols)} features (95th pct): {max_hi:.3f}")
    best_name, best_a = scan[0][0], scan[0][1]
    if best_a > max_hi:
        print(f"\n  -> '{best_name}' at {best_a:.3f} survives multiplicity correction.")
        print("     Worth preregistering as a single-feature rule and testing once.")
    else:
        print(f"\n  -> the best of {len(cols)} features reaches {best_a:.3f} against a "
              f"best-of null of {max_hi:.3f}.")
        print("     NO observable feature separates at the allocation point.")
        print("     A better model cannot recover signal the features do not carry.")

    # ---- 2. selection on the criterion that matters ------------------------
    print("\n  " + "=" * 66)
    print("  2. MODEL SELECTION BY ALLOCATION-POINT AUC (calibration only)")
    print("  " + "=" * 66)
    print("  The preregistered run selected on POOLED AUC, which is positional.")
    print("  This selects on the metric the decision is actually made with.\n")

    size_idx = [j for j, nm in enumerate(names) if nm in SIZE_FEATURES]
    cands = {
        "logistic": (_lr(class_weight="balanced"), None),
        "gbt": (GradientBoostingClassifier(random_state=0, n_estimators=150,
                                           max_depth=2), None),
        "rf": (RandomForestClassifier(random_state=0, n_estimators=300,
                                      min_samples_leaf=5,
                                      class_weight="balanced"), None),
        "logistic-l1": (_lr(penalty="l1", solver="liblinear",
                            class_weight="balanced"), None),
        "logistic/size-only": (_lr(class_weight="balanced"), size_idx),
        "code_lines alone": (_lr(class_weight="balanced"),
                             [j for j, nm in enumerate(names)
                              if nm == "code_lines"]),
    }
    n_splits = min(5, len(set(g)))
    def oof_for(m, cols):
        sub = (lambda Z: Z[:, cols]) if cols else (lambda Z: Z)
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits).split(X, y, groups=g):
            oof[te] = (clone(m).fit(sub(X[tr]), y[tr])
                       .predict_proba(sub(X[te]))[:, 1])
        return oof

    print(f"  {'model':20s} {'feats':>5} {'pooled':>7} {'alloc-pt':>9}")
    results, oofs = {}, {}
    for nm, (m, cols) in cands.items():
        oof = oof_for(m, cols)
        oofs[nm] = oof
        pooled = float(auc(y, oof))
        alloc = float(auc(y0, oof[first]))
        results[nm] = alloc
        print(f"  {nm:20s} {len(cols) if cols else X.shape[1]:5d} "
              f"{pooled:7.3f} {alloc:9.3f}")

    # Null for the best of several models is likewise inflated.
    model_null = []
    sc = {nm: oofs[nm][first] for nm in cands}
    for _ in range(N_PERM):
        yp = rng.permutation(y0)
        model_null.append(max(float(auc(yp, s)) for s in sc.values()))
    model_hi = float(np.percentile(model_null, 95))

    best = max(results, key=results.get)
    print(f"\n  best by allocation point: {best} ({results[best]:.3f})")
    print(f"  null for best-of-{len(cands)}-models (95th pct): {model_hi:.3f}")
    print("  (the parsimonious rows are the point: 34 positives cannot support")
    print("   25 features, so a smaller family can beat the full vector)")
    if results[best] > model_hi:
        print(f"\n  -> {best} clears the null even after multiplicity.")
        print("     The original gate failure WAS a selection-criterion artifact.")
        print("     Preregister this criterion and re-run; do not report it as the")
        print("     primary result, because the criterion was chosen after seeing")
        print("     the first outcome.")
    else:
        print(f"\n  -> no model clears {model_hi:.3f} at the allocation point.")
        print("     The gate failure is NOT a selection artifact. Fixing the")
        print("     criterion changes nothing, which is worth knowing.")

    # ---- 3. what would it take? -------------------------------------------
    print("\n  " + "=" * 66)
    print("  3. POWER")
    print("  " + "=" * 66)
    print(f"  events per feature: {pos}/{X.shape[1]} = {pos/X.shape[1]:.1f}"
          f"   (E0028 called 7.0 data-starved)")
    a_best = max(results[best], best_a)
    if a_best > 0.5:
        # n needed for a one-sided test of this AUC against 0.5, roughly
        p1 = pos / n
        se_unit = np.sqrt((a_best * (1 - a_best)) / max(pos, 1))
        need = int(np.ceil(((1.96 + 0.84) / max(a_best - 0.5, 1e-6)) ** 2
                           * a_best * (1 - a_best) / max(p1, 1e-6)))
        print(f"  observed best AUC {a_best:.3f} (se~{se_unit:.3f})")
        print(f"  to resolve an effect this size at 80% power: n ~ {need} problems"
              f"  (have {n})")
    else:
        print(f"  best AUC is {a_best:.3f}, at or below chance -- no effect size to"
              f" power for.")

    print("\n  Everything above is SECONDARY. The reportable E0029 result remains:")
    print("  gate 3 failed, ceiling +0.1378, allocation-point AUC 0.415.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
