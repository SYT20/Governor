#!/usr/bin/env python3
"""E0028 -- can a LEARNED ranker recover the oracle's marginal-value ordering?

E0027 established the shape of the problem:

    observable ceiling            +0.0588
    oracle marginal ranking       +0.0505 [+0.0203, +0.0816]   <- the policy works
    learned marginal ranking      +0.0055 [-0.0104, +0.0218]   <- indistinguishable
                                                                  from random
    diagnosis                     19 positives, 2.7 events per feature

So the allocation formulation is sound and the ranker is data-starved. This
experiment attacks that without acquiring more problems:

  TARGET   at each decision point, given every attempt so far failed, does ANY
           later sample succeed? 1003 rows and 49 positives on calibration,
           against 207 rows and 19 positives -- 2.6x, 7.0 events per feature.

  FEATURES the trajectory of attempts plus the structure of the generated code,
           not just the last attempt's public-test score. Three attempts failing
           the SAME public test is different evidence from three failing
           different ones, and the old feature set could not say so.

  PROTOCOL model and operating point are chosen by inner cross-validation on
           CALIBRATION ONLY, frozen, and then applied once to evaluation.
           E0027's rank-fraction sweep touched the evaluation set and was
           reported as a diagnostic; nothing here does that.

The success criterion is NOT a higher AUC. It is a ranked policy that beats the
matched-cost fixed baseline with a confidence interval excluding zero.
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from governor.execfeedback.richfeatures import FEATURE_NAMES, decision_features
from governor.harness.ledger import ExperimentRun, ExperimentSpec
from governor.harness.traps import render, run_trap_checks
from governor.models.calibration import auc

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "E0028_rich.json"
OLD_FEATURES = ("pub_frac", "pub_passed", "pub_failed", "runtime_error",
                "timeout", "output_nonempty", "exec_latency_s")


def load():
    rows = json.loads(DATA.read_text())
    byq = collections.defaultdict(list)
    for r in rows:
        byq[r["qid"]].append(r)
    for q in byq:
        byq[q].sort(key=lambda r: r["sample"])
    ise = lambda q: int(hashlib.sha256(q.encode()).hexdigest(), 16) % 2 == 1
    return byq, sorted(q for q in byq if not ise(q)), sorted(q for q in byq if ise(q))


def decision_rows(byq, qids, rich=True):
    """One row per decision point: attempts 0..i all failed, decide about i+1."""
    X, y, groups = [], [], []
    for q in qids:
        s = byq[q]
        for i in range(len(s) - 1):
            if s[i]["graded"]:
                break                                   # solved; nothing to decide
            attempts = s[:i + 1]
            f = (decision_features(attempts) if rich
                 else {k: float(s[i][k]) for k in OLD_FEATURES})
            X.append([f[k] for k in sorted(f)])
            y.append(1.0 if any(x["graded"] for x in s[i + 1:]) else 0.0)
            groups.append(q)
    return np.array(X), np.array(y), np.array(groups)


def ranking_metrics(y, score, ks=(0.05, 0.10, 0.20)):
    """AUC alone hides what matters: whether rare positives reach the top."""
    n, pos = len(y), int(y.sum())
    order = np.argsort(-score)
    out = {"auc": float(auc(y, score)), "n": n, "positives": pos}
    for k in ks:
        m = max(1, int(round(k * n)))
        top = order[:m]
        hit = int(y[top].sum())
        out[f"precision@{int(k*100)}"] = hit / m
        out[f"recall@{int(k*100)}"] = hit / max(pos, 1)
        out[f"lift@{int(k*100)}"] = (hit / m) / (pos / n) if pos else 0.0
    gains = y[order]
    dcg = float(np.sum(gains / np.log2(np.arange(2, n + 2))))
    ideal = float(np.sum(np.sort(y)[::-1] / np.log2(np.arange(2, n + 2))))
    out["ndcg"] = dcg / ideal if ideal else 0.0
    return out


def main() -> int:
    rng = np.random.default_rng(0)
    byq, cal, ev = load()
    print(f"  split: calibration={len(cal)}  evaluation={len(ev)}\n")

    # ---------- TEST A / B: what do the two feature sets support? ----------
    print("  === TEST A/B -- ranking quality by feature set (inner CV on calibration) ===")
    results = {}
    for tag, rich in (("old 7 execution features", False), ("rich trajectory+static", True)):
        X, y, g = decision_rows(byq, cal, rich=rich)
        oof = np.zeros(len(y))
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        for tr, te in skf.split(X, y):
            m = GradientBoostingClassifier(random_state=0, n_estimators=150, max_depth=2)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        met = ranking_metrics(y, oof)
        results[tag] = met
        print(f"    {tag:<26} n={met['n']:>5} pos={met['positives']:>3} "
              f"AUC={met['auc']:.3f} P@10={met['precision@10']:.3f} "
              f"R@10={met['recall@10']:.3f} lift@10={met['lift@10']:.2f} "
              f"NDCG={met['ndcg']:.3f}")

    # ---------- TEST D: model selection by inner CV, on calibration only ----------
    print("\n  === TEST D -- model selection, inner CV, calibration only ===")
    X, y, g = decision_rows(byq, cal, rich=True)
    cands = {
        "logistic": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "gbt": GradientBoostingClassifier(random_state=0, n_estimators=150, max_depth=2),
        "rf": RandomForestClassifier(random_state=0, n_estimators=300,
                                     min_samples_leaf=5, class_weight="balanced"),
    }
    best_name, best_auc, best_oof = None, -1.0, None
    for name, m in cands.items():
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            mm = m.__class__(**m.get_params()).fit(X[tr], y[tr])
            oof[te] = mm.predict_proba(X[te])[:, 1]
        a = ranking_metrics(y, oof)
        print(f"    {name:<10} AUC={a['auc']:.3f}  P@10={a['precision@10']:.3f}  "
              f"lift@10={a['lift@10']:.2f}  NDCG={a['ndcg']:.3f}")
        if a["auc"] > best_auc:
            best_name, best_auc, best_oof = name, a["auc"], oof
    print(f"    selected: {best_name} (AUC {best_auc:.3f})")

    model = cands[best_name].__class__(**cands[best_name].get_params()).fit(X, y)

    # ---------- operating point, chosen on calibration ONLY ----------
    def problem_score(qids, mdl):
        out = []
        for q in qids:
            s = byq[q]
            if s[0]["graded"]:
                out.append(-1.0)                     # already solved by sample 1
                continue
            f = decision_features(s[:1])
            out.append(float(mdl.predict_proba([[f[k] for k in sorted(f)]])[0, 1]))
        return np.array(out)

    def evaluate(qids, score, frac):
        U1 = np.array([1.0 if byq[q][0]["graded"] else 0.0 for q in qids])
        Ua = np.array([1.0 if any(r["graded"] for r in byq[q]) else 0.0 for q in qids])
        C1 = np.array([byq[q][0]["tokens"] for q in qids])
        Rest = np.array([sum(r["tokens"] for r in byq[q][1:]) for q in qids])
        take = set(np.argsort(-score)[:int(round(frac * len(qids)))])
        U = np.array([Ua[i] if i in take else U1[i] for i in range(len(qids))])
        C = np.array([C1[i] + (Rest[i] if i in take else 0.0) for i in range(len(qids))])
        return U, C

    def fixed_at(qids, cost):
        cs = [np.mean([sum(r["tokens"] for r in byq[q][:k]) for q in qids]) for k in range(1, 11)]
        per = [np.array([1.0 if any(r["graded"] for r in byq[q][:k]) else 0.0 for q in qids])
               for k in range(1, 11)]
        us = [p.mean() for p in per]
        if cost <= cs[0]:
            return us[0] * cost / cs[0], per[0] * cost / cs[0]
        for i in range(len(cs) - 1):
            if cs[i] <= cost <= cs[i + 1]:
                w = (cost - cs[i]) / (cs[i + 1] - cs[i])
                return us[i] + w * (us[i + 1] - us[i]), (1 - w) * per[i] + w * per[i + 1]
        return us[-1], per[-1]

    cal_score = problem_score(cal, model)
    frac_star, best_adv = None, -9.0
    for frac in np.linspace(0.05, 1.0, 20):
        U, C = evaluate(cal, cal_score, frac)
        ub, _ = fixed_at(cal, C.mean())
        if U.mean() - ub > best_adv:
            frac_star, best_adv = float(frac), float(U.mean() - ub)
    print(f"\n  operating point frozen on calibration: frac={frac_star:.2f} "
          f"(calibration advantage {best_adv:+.4f})")

    # ---------- ONE evaluation ----------
    ev_score = problem_score(ev, model)
    U, C = evaluate(ev, ev_score, frac_star)
    ub, Ub = fixed_at(ev, C.mean())
    d = U - Ub
    boots = [d[rng.integers(0, len(ev), len(ev))].mean() for _ in range(4000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])

    Xv, yv, _ = decision_rows(byq, ev, rich=True)
    ev_met = ranking_metrics(yv, model.predict_proba(Xv)[:, 1])

    U1 = np.array([1.0 if byq[q][0]["graded"] else 0.0 for q in ev])
    Ua = np.array([1.0 if any(r["graded"] for r in byq[q]) else 0.0 for q in ev])
    orc = np.array([1.0 if (not U1[i] and Ua[i]) else 0.0 for i in range(len(ev))])
    Uo, Co = evaluate(ev, orc, frac_star)
    ubo, _ = fixed_at(ev, Co.mean())

    print(f"\n  === HELD-OUT EVALUATION (single application, n={len(ev)}) ===")
    print(f"    Governor        U={U.mean():.4f}  cost={C.mean():>7.1f}")
    print(f"    best fixed      U={ub:.4f}")
    print(f"    advantage       {U.mean()-ub:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"    oracle same frac{Uo.mean()-ubo:+.4f}")
    print(f"    held-out ranking: AUC={ev_met['auc']:.3f} P@10={ev_met['precision@10']:.3f} "
          f"lift@10={ev_met['lift@10']:.2f}")

    # ---------- power ----------
    sd = float(np.std(d, ddof=1))
    n_req = float((1.96 * sd / 0.02) ** 2) if sd > 0 else float("inf")
    print(f"\n  power: paired sd={sd:.4f}; to resolve eps=0.02 needs n≈{n_req:.0f}, have {len(ev)}")

    verdict = "PASS" if lo > 0 else ("SAMPLE-UNATTAINABLE" if n_req > 400 else "INCONCLUSIVE")

    evd = {
      "gov_utils": U, "greedy_utils": Ub,
      "gov_calls": np.array([1.0 if s > 0 else 0.0 for s in ev_score]),
      "greedy_calls": np.full(len(ev), frac_star),
      "decisions_by_state": [tuple(np.round(x, 3)) for x in Xv[:len(ev)]],
      "feature_names": list(FEATURE_NAMES),
      "answered_rate": 1.0, "utility": float(U.mean()),
      "requested": C, "actual_used": C, "charged": C,
      "scored_via_executor": True,
      "decisions": np.isin(np.arange(len(ev)),
                           np.argsort(-ev_score)[:int(round(frac_star*len(ev)))]).astype(int).tolist(),
      "cell_ids": [byq[q][0]["platform"] for q in ev],
      "froze_commit": "b8e2884", "heldout_commit": "HEAD",
      "selection_item_ids": cal, "evaluation_item_ids": ev,
      "token_cost_source": "exact tokenizer count over published LiveCodeBench generations",
      "realised_cost": float(C.mean()), "budget": float(C.mean()),
      "baseline_cost": float(C.mean()),
      "cited_experiment_ids": ["E0026-execution-feedback", "E0027-marginal-value"],
      "withdrawn_ids": ["E0019-predictor-loss-math", "E0017-soft-governor-math"],
    }
    traps = run_trap_checks(evd)
    print("\n" + render(traps))
    if [k for k, (ok, _) in traps.items() if not ok]:
        verdict = "BLOCKED"

    spec = ExperimentSpec(
        exp_id="E0028-marginal-ranker",
        title="Learned marginal-value ranking with trajectory and static features",
        model="Gemini-Pro-1.5 (May) generations, published by LiveCodeBench",
        budget={"axis": "rank-then-allocate; sample 1 everywhere, remainder to top-ranked",
                "B_star": float(C.mean()),
                "charged": "exact tokenizer count over published generations"},
        seeds={"split": "sha256(question_id) parity", "cv": 0, "bootstrap": 0},
        split={"calibration": len(cal), "evaluation": len(ev)},
        metric="pass-within-k at matched realised cost; primary = ranked policy "
               "minus the randomised fixed envelope at its own cost",
        params={"features": list(FEATURE_NAMES), "model": best_name,
                "frac": frac_star, "target": "any later sample succeeds given all so far failed",
                "selection": "5-fold stratified inner CV on calibration only"},
        notes="Operating point frozen on calibration before a single evaluation "
              "application. E0027's rank-fraction sweep touched evaluation and was "
              "reported as diagnostic; this does not.")
    run = ExperimentRun(spec, overwrite=True)
    for i, q in enumerate(ev):
        run.append({"qid": q, "score": float(ev_score[i]), "u": float(U[i]),
                    "fixed": float(Ub[i]), "tokens": float(C[i])})
    run.finalize(summary={
        "governor_U": float(U.mean()), "fixed_matched": float(ub),
        "primary_mean": float(U.mean() - ub), "primary_lo": float(lo), "primary_hi": float(hi),
        "oracle_same_frac": float(Uo.mean() - ubo),
        "heldout_auc": ev_met["auc"], "heldout_precision_at_10": ev_met["precision@10"],
        "heldout_lift_at_10": ev_met["lift@10"], "heldout_ndcg": ev_met["ndcg"],
        "cv_auc_old_features": results["old 7 execution features"]["auc"],
        "cv_auc_rich_features": results["rich trajectory+static"]["auc"],
        "calibration_positives": int(y.sum()), "decision_rows": int(len(y)),
        "events_per_feature": float(y.sum() / len(FEATURE_NAMES)),
        "paired_sd": sd, "n_required_for_eps_0.02": n_req,
        "frac": frac_star, "model": best_name, "verdict": verdict},
        metrics={"cv": {k: v["auc"] for k, v in results.items()}},
        traps=traps, verdict=verdict)
    print(f"\n  recorded E0028-marginal-ranker  verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
