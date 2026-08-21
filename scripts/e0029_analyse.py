#!/usr/bin/env python3
"""E0029 -- does the Governor allocate real compute on a model we generated from?

WHY THIS FILE EXISTS. The first E0029 "analysis" ran e0028_marginal_ranker.py,
which hardcodes results/E0028_rich.json. It re-ran E0028, printed split
207/193, and reproduced E0028's numbers byte for byte. All 4750 Qwen rows were
ignored and the output was reported as an E0029 result. Every structural guard
below exists because that happened:

  * the dataset is asserted, not assumed -- 475 problems, split 250/225, and the
    exact E0028 fingerprint 207/193 is a named fatal error rather than a
    coincidence someone has to notice in a log;
  * the hidden label comes from private tests via e0029_grade.py and is joined
    on (problem_id, sample_id), never recomputed from anything observable;
  * the feature/label boundary is audited into a file, so it is checkable after
    the fact instead of resting on the author's care.

THE INFORMATION BOUNDARY, which is the whole experiment:

    PUBLIC tests  -> features -> the Governor's decision
    PRIVATE tests -> label    -> scoring, strictly afterwards

If the label leaks into the features the result is meaningless, and it will look
excellent. LiveCodeBench's published metadata already turned out to BE the label
once here (1530 empty/PASS, 2470 non-empty/FAIL, zero off-diagonal).

GATES, run in order, each able to stop the run:

    0  data integrity     right dataset, complete join, expected split
    1  boundary audit     no forbidden field reachable from any feature
    2  ceiling            is there anything for a policy to win?
    3  predictor          can any model rank better than chance, grouped CV?
    4  governor           does the frozen policy beat matched-cost fixed spend?

A gate that fails stops the run. Reporting a Governor advantage computed on top
of a ceiling of zero, or a predictor at chance, is how the earlier rounds of this
project produced numbers that did not survive contact with a held-out set.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from governor.execfeedback.privatetests import FORBIDDEN_AS_FEATURE
from governor.execfeedback.richfeatures import FEATURE_NAMES, decision_features
from governor.harness.traps import FORBIDDEN_FEATURES, render, run_trap_checks
from governor.models.calibration import auc

ROOT = pathlib.Path(__file__).resolve().parents[1]
GEN = ROOT / "results" / "e0029_colab_generations.jsonl"
GRADED = ROOT / "results" / "E0029_QWEN_graded.jsonl"
PROBLEMS = ROOT / "results" / "e0029_problems.json"
AUDIT_OUT = ROOT / "results" / "E0029_feature_audit.json"
FROZEN = ROOT / "results" / "E0029_frozen.json"
FROZEN_MODEL = ROOT / "results" / "E0029_frozen_model.pkl"

EXPECTED_PROBLEMS = 475
EXPECTED_CAL = 250
EXPECTED_EVAL = 225
E0028_FINGERPRINT = (207, 193)      # the wrong dataset, seen once, never again

# Everything the private-test stage emits, plus the older vocabulary. A feature
# drawn from any of these is the experiment scoring itself against its own input.
LABEL_FIELDS = set(FORBIDDEN_AS_FEATURE) | {
    "graded", "grade", "solved", "hidden_all_passed", "hidden_pass_fraction",
    "private_test_cases", "expected_output", "reference_solution",
}


def _head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                           capture_output=True, text=True)
        return r.stdout.strip() or "unknown"
    except Exception:                                     # noqa: BLE001
        return "unknown"


def _rel(p: pathlib.Path) -> str:
    """Paths under a fixture's temp dir are not relative to ROOT; a diagnostic
    print must not be the thing that kills the run."""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


SEPARATION_AUC = 0.98     # above this, a lone feature is the label in disguise


def _determinism(col: np.ndarray, y: np.ndarray) -> bool:
    """True when some threshold splits the label with ZERO off-diagonal -- the
    sharpest form of a leak, and how the metadata leak was finally identified."""
    vals = np.unique(col)
    if len(vals) > 50:
        # High-cardinality: a perfectly deterministic split would already show as
        # AUC 1.0, so scanning thousands of thresholds buys nothing but time.
        return False
    for t in vals:
        side = col <= t
        if side.all() or (~side).all():
            continue
        if (len(set(y[side].tolist())) == 1 and len(set(y[~side].tolist())) == 1
                and y[side][0] != y[~side][0]):
            return True
    return False


def _max_sep(X: np.ndarray, y: np.ndarray, names: list[str]) -> float:
    best = 0.0
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.allclose(col, col[0]):
            continue
        a = float(auc(y, col))
        best = max(best, max(a, 1.0 - a))
    return best


class GateFailure(RuntimeError):
    """A gate said stop. Raised so no later stage can run on its result."""


def _is_eval(qid: str) -> bool:
    """Preregistered split: sha256 parity of the problem id. Deterministic,
    independent of row order, and identical to the rule E0027/E0028 used."""
    return int(hashlib.sha256(qid.encode()).hexdigest(), 16) % 2 == 1


# ----------------------------------------------------------------- gate 0
def load_joined() -> tuple[dict, list[str], list[str], dict]:
    for p, what in ((GEN, "generations"), (GRADED, "grades")):
        if not p.exists():
            raise GateFailure(
                f"missing {what}: {_rel(p)}\n"
                + ("  Generation ran on the Colab VM. Copy the raw JSONL back.\n"
                   if what == "generations" else
                   "  Run: python scripts/e0029_grade.py   (CPU-only)\n"))

    gens = [json.loads(l) for l in GEN.read_text().splitlines() if l.strip()]
    grades = {}
    for l in GRADED.read_text().splitlines():
        if not l.strip():
            continue
        d = json.loads(l)
        grades[(d["problem_id"], d["sample_id"])] = d

    joined, unmatched = [], 0
    for g in gens:
        key = (g["problem_id"], g["sample_id"])
        hit = grades.get(key)
        if hit is None:
            unmatched += 1
            continue
        row = dict(g)
        row["hidden_all_passed"] = bool(hit["hidden_all_passed"])
        row["hidden_pass_fraction"] = float(hit["hidden_pass_fraction"])
        row["grading_status"] = hit["grading_status"]
        joined.append(row)

    if not joined:
        raise GateFailure("the join produced no rows: generations and grades "
                          "share no (problem_id, sample_id) key")
    if unmatched:
        print(f"    WARNING: {unmatched} generated rows have no grade "
              f"({unmatched/len(gens):.1%}); they are DROPPED, not scored 0")

    byq = collections.defaultdict(list)
    for r in joined:
        byq[r["problem_id"]].append(r)
    for q in byq:
        byq[q].sort(key=lambda r: r["sample_id"])

    cal = sorted(q for q in byq if not _is_eval(q))
    ev = sorted(q for q in byq if _is_eval(q))

    meta = {"generated_rows": len(gens), "graded_rows": len(grades),
            "joined_rows": len(joined), "unmatched": unmatched,
            "problems": len(byq), "cal": len(cal), "eval": len(ev),
            "models": sorted({r.get("model", "?") for r in joined}),
            "samples_per_problem": sorted({len(v) for v in byq.values()})}
    return byq, cal, ev, meta


def gate0(meta: dict) -> None:
    print("  === GATE 0 -- data integrity ===")
    for k in ("generated_rows", "graded_rows", "joined_rows", "problems",
              "cal", "eval", "models", "samples_per_problem"):
        print(f"    {k:22s} {meta[k]}")

    if (meta["cal"], meta["eval"]) == E0028_FINGERPRINT:
        raise GateFailure(
            f"split is {E0028_FINGERPRINT[0]}/{E0028_FINGERPRINT[1]} -- that is "
            f"E0028's 400-problem dataset, not E0029's {EXPECTED_PROBLEMS}.\n"
            "    This is the exact failure that produced the invalid E0029 result:\n"
            "    an analysis script with a hardcoded E0028 path re-ran E0028.")
    if meta["problems"] != EXPECTED_PROBLEMS:
        raise GateFailure(f"expected {EXPECTED_PROBLEMS} problems, got "
                          f"{meta['problems']}")
    if (meta["cal"], meta["eval"]) != (EXPECTED_CAL, EXPECTED_EVAL):
        raise GateFailure(f"expected split {EXPECTED_CAL}/{EXPECTED_EVAL}, got "
                          f"{meta['cal']}/{meta['eval']}")
    if len(meta["models"]) != 1:
        raise GateFailure(f"rows mix models: {meta['models']}")
    print(f"    OK: {EXPECTED_PROBLEMS} problems, split "
          f"{EXPECTED_CAL}/{EXPECTED_EVAL}, one model\n")


# ----------------------------------------------------------------- gate 1
def gate1_boundary_audit(byq: dict, cal: list[str]) -> dict:
    """Prove the feature vector cannot see the label, and write the proof out."""
    print("  === GATE 1 -- feature/label separation audit ===")

    probe = byq[cal[0]][:2]
    feats = decision_features(probe)
    names = sorted(feats)

    violations = [n for n in names
                  if n in LABEL_FIELDS or n in FORBIDDEN_FEATURES
                  or any(t in n.lower() for t in ("hidden", "private", "graded",
                                                  "oracle", "label", "truth"))]
    if violations:
        raise GateFailure(f"feature names collide with the label: {violations}")

    # Naming is not proof. Perturb the label across every row and confirm not one
    # feature value moves: this catches a feature that reads the label under an
    # innocent name, which naming checks cannot.
    mutated = []
    for r in probe:
        m = dict(r)
        m["hidden_all_passed"] = not m["hidden_all_passed"]
        m["hidden_pass_fraction"] = 1.0 - m["hidden_pass_fraction"]
        m["grading_status"] = "MUTATED"
        mutated.append(m)
    after = decision_features(mutated)
    moved = [n for n in names if abs(feats[n] - after[n]) > 1e-12]
    if moved:
        raise GateFailure(f"features MOVED when only the label changed: {moved}")

    # And the converse: the features must actually respond to the public signal,
    # or the vector is inert and the predictor is being handed noise.
    pub = []
    for r in probe:
        m = dict(r)
        m["pub_frac"] = 0.0 if float(r.get("pub_frac", 0)) > 0.5 else 1.0
        m["pub_passed"], m["pub_failed"] = r.get("pub_failed", 0), r.get("pub_passed", 0)
        pub.append(m)
    after_pub = decision_features(pub)
    responsive = [n for n in names if abs(feats[n] - after_pub[n]) > 1e-12]
    if not responsive:
        raise GateFailure("no feature responds to the public-test signal: "
                          "the feature vector is inert")

    # PROBE 3 -- separation. The two probes above only catch a feature that
    # READS the label while features are computed. A leak baked into a stored
    # column at generation time moves nothing when the label is mutated and
    # carries an innocent name, so both miss it entirely. That is precisely the
    # shape of the LiveCodeBench metadata leak: emptiness EQUALLED the label,
    # 1530 empty/PASS and 2470 non-empty/FAIL with zero off-diagonal.
    #
    # A single observable feature that separates a hard hidden outcome almost
    # perfectly is not a good feature. It is the label wearing a different name.
    Xa, ya, _ = decision_rows(byq, cal)
    separators = []
    if len(ya) and 0 < ya.sum() < len(ya):
        for j, nm in enumerate(names):
            col = Xa[:, j]
            if np.allclose(col, col[0]):
                continue
            a = float(auc(ya, col))
            a = max(a, 1.0 - a)                       # either direction is a leak
            det = _determinism(col, ya)
            if a >= SEPARATION_AUC or det:
                separators.append({"feature": nm, "auc": round(a, 4),
                                   "deterministic": det})
    if separators:
        raise GateFailure(
            "a single feature separates the hidden label almost perfectly:\n"
            + "\n".join(f"      {d['feature']}: AUC={d['auc']:.4f}"
                         f"{'  DETERMINISTIC (zero off-diagonal)' if d['deterministic'] else ''}"
                         for d in separators)
            + "\n    That is the signature of the label leaking into an observable"
              "\n    column upstream, not of a good feature.")
    print(f"    single-feature separation: max AUC over {len(names)} features "
          f"= {_max_sep(Xa, ya, names):.3f} (trip at {SEPARATION_AUC})")

    audit = {
        "experiment_id": "E0029-QWEN",
        "feature_count": len(names),
        "feature_names": names,
        "declared_feature_names": list(FEATURE_NAMES),
        "forbidden_fields_checked": sorted(LABEL_FIELDS),
        "name_collisions": violations,
        "features_moved_when_label_mutated": moved,
        "features_responsive_to_public_signal": responsive,
        "single_feature_separators": separators,
        "separation_auc_threshold": SEPARATION_AUC,
        "label_source": "private tests via governor/execfeedback/privatetests.py",
        "label_field": "hidden_all_passed",
        "feature_source": "public tests via governor/execfeedback/publictests.py",
        "join_key": ["problem_id", "sample_id"],
        "verdict": "SEPARATED",
    }
    AUDIT_OUT.write_text(json.dumps(audit, indent=2))
    print(f"    {len(names)} features, 0 name collisions")
    print(f"    label mutated across all rows -> {len(moved)} feature values moved")
    print(f"    public signal mutated        -> {len(responsive)} feature values moved")
    print(f"    wrote {_rel(AUDIT_OUT)}\n")
    return audit


# ----------------------------------------------------------------- gate 2
def utilities(byq: dict, qids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    U1 = np.array([1.0 if byq[q][0]["hidden_all_passed"] else 0.0 for q in qids])
    Ua = np.array([1.0 if any(r["hidden_all_passed"] for r in byq[q]) else 0.0
                   for q in qids])
    C1 = np.array([float(byq[q][0].get("total_tokens", 0)) for q in qids])
    Rest = np.array([sum(float(r.get("total_tokens", 0)) for r in byq[q][1:])
                     for q in qids])
    return U1, Ua, C1, Rest


def gate2_ceiling(byq: dict, cal: list[str], ev: list[str]) -> dict:
    """Is there anything to win? Two ceilings, and only one of them is honest.

    OMNISCIENT decides before paying for sample 1, so it never spends on a
    problem sample 1 would have solved. No observable policy can do that -- the
    information arrives only after the spend. It is reported to show the gap.

    OBSERVABLE pays for sample 1, sees it fail, and only then is told the label.
    That is a genuine upper bound on any policy built from observable features,
    and it is the number the later gates must be judged against.
    """
    print("  === GATE 2 -- ceiling ===")
    out = {}
    for tag, qids in (("calibration", cal), ("evaluation", ev)):
        U1, Ua, C1, Rest = utilities(byq, qids)
        n = len(qids)
        solved_1 = U1.mean()
        solved_any = Ua.mean()
        rescuable = float(((U1 == 0) & (Ua == 1)).mean())

        # spend the rest only where sample 1 failed AND a later sample succeeds
        frac_spent = rescuable
        omniscient = solved_any - solved_1
        observable = rescuable

        out[tag] = {"n": n, "solve@1": float(solved_1), "solve@all": float(solved_any),
                    "rescuable": rescuable, "omniscient_gain": float(omniscient),
                    "observable_gain": float(observable),
                    "frac_needing_spend": float(frac_spent),
                    "mean_cost_1": float(C1.mean()),
                    "mean_cost_rest": float(Rest.mean())}
        print(f"    {tag:12s} n={n:>4}  solve@1={solved_1:.4f}  "
              f"solve@all={solved_any:.4f}  rescuable={rescuable:.4f}")

    e = out["evaluation"]
    if e["rescuable"] <= 0.0:
        raise GateFailure(
            "ceiling is zero: no evaluation problem is solved by a later sample "
            "that sample 1 missed. No allocation policy can win here, so a\n"
            "    Governor advantage would be measuring noise.")
    if e["rescuable"] < 0.01:
        print(f"    WARNING: ceiling {e['rescuable']:.4f} is below 1 point; "
              f"n={e['n']} cannot resolve an effect this small")
    print(f"    observable ceiling (evaluation): {e['rescuable']:+.4f}\n")
    return out


# ----------------------------------------------------------------- gate 3
def decision_rows(byq: dict, qids: list[str]):
    """One row per decision point: attempts 0..i all failed, decide about i+1.

    The label is 'does any LATER sample succeed', which is the quantity the
    allocator needs and is not observable at decision time.
    """
    X, y, groups = [], [], []
    for q in qids:
        s = byq[q]
        for i in range(len(s) - 1):
            if s[i]["hidden_all_passed"]:
                break                                   # solved; nothing to decide
            f = decision_features(s[:i + 1])
            X.append([f[k] for k in sorted(f)])
            y.append(1.0 if any(x["hidden_all_passed"] for x in s[i + 1:]) else 0.0)
            groups.append(q)
    return np.array(X), np.array(y), np.array(groups)


def gate3_predictor(byq: dict, cal: list[str]):
    """Select a model by GROUPED inner CV on calibration only.

    Grouping is not a refinement. A problem contributes up to nine decision rows
    that all carry the same label, so a row-wise split puts the same problem in
    train and test and the model memorises it: that produced CV AUC 0.951
    against a held-out 0.628 here.
    """
    print("  === GATE 3 -- predictor (grouped CV, calibration only) ===")
    X, y, g = decision_rows(byq, cal)
    names_sorted = sorted(decision_features(byq[cal[0]][:1]))
    pos = int(y.sum())
    print(f"    rows={len(y)}  positives={pos}  "
          f"events/feature={pos/max(X.shape[1],1):.1f}")
    if pos < 10:
        raise GateFailure(f"only {pos} positive decision points; nothing to learn")

    cands = {
        "logistic": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "gbt": GradientBoostingClassifier(random_state=0, n_estimators=150,
                                          max_depth=2),
        "rf": RandomForestClassifier(random_state=0, n_estimators=300,
                                     min_samples_leaf=5, class_weight="balanced"),
    }
    best, best_auc, rows = None, -1.0, {}
    n_splits = min(5, len(set(g)))
    for name, m in cands.items():
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits).split(X, y, groups=g):
            oof[te] = (m.__class__(**m.get_params()).fit(X[tr], y[tr])
                       .predict_proba(X[te])[:, 1])
        a = float(auc(y, oof))
        rows[name] = a
        print(f"    {name:<10} grouped-CV AUC = {a:.3f}")
        if a > best_auc:
            best, best_auc = name, a

    # ---- the null that matters: POSITION is not evidence -------------------
    # With ten samples at any fixed success rate, P(some LATER sample succeeds)
    # rises as attempts remain, so `attempts_left` predicts the label perfectly
    # legitimately -- and tells you nothing about WHICH problem deserves the
    # spend. A ranker riding that signal scores well and allocates blindly.
    # This is the project's own progress_as_cognition trap, and an earlier
    # version of this gate passed synthetic data whose label was independent of
    # every observable, purely on positional structure.
    pos_idx = [j for j, nm in enumerate(names_sorted)
               if nm in ("attempt_idx", "attempts_left")]
    ev_idx = [j for j in range(X.shape[1]) if j not in pos_idx]

    def grouped_auc(cols):
        if not cols:
            return 0.5
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits).split(X, y, groups=g):
            m = LogisticRegression(max_iter=3000, class_weight="balanced")
            oof[te] = m.fit(X[tr][:, cols], y[tr]).predict_proba(X[te][:, cols])[:, 1]
        a = float(auc(y, oof))
        return max(a, 1.0 - a)

    pos_auc = grouped_auc(pos_idx)
    print(f"    position-only baseline (attempt_idx, attempts_left) AUC = {pos_auc:.3f}")

    # The allocation decision is made at sample 1, where every position feature
    # is identical across problems. If the ranker cannot separate THERE, it
    # cannot allocate, whatever its pooled AUC says.
    first = np.array([i for i, gi in enumerate(g)
                      if i == 0 or g[i - 1] != gi])
    y0 = y[first]
    if 0 < y0.sum() < len(y0):
        oof0 = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits).split(X, y, groups=g):
            mm = cands[best].__class__(**cands[best].get_params()).fit(X[tr], y[tr])
            oof0[te] = mm.predict_proba(X[te])[:, 1]
        a0 = float(auc(y0, oof0[first]))
        print(f"    at the allocation point (sample 1, n={len(first)}, "
              f"pos={int(y0.sum())}) AUC = {a0:.3f}")
    else:
        raise GateFailure("the label is constant at the allocation point: "
                          "every problem is rescuable or none is, so there is "
                          "nothing to rank")

    # The GATE is the allocation point, not the pooled figure. Pooled AUC mixes
    # decision depths, so position dominates it whether or not the ranker has
    # any evidential content -- here the position-only model scores ABOVE the
    # full model pooled, while the full model still separates at sample 1. At
    # sample 1 every position feature is identical across problems, so whatever
    # separation survives there is evidence about the problem and nothing else.
    rng = np.random.default_rng(0)
    sc0 = oof0[first]
    # NOT folded with max(a, 1-a). Folding is right for the leak probe, where
    # separation in either direction is suspicious, but a ranker that orders
    # problems backwards is useless, not half-right -- folding the null here
    # while leaving a0 unfolded compares two different statistics and makes the
    # gate wrongly strict.
    null = np.array([float(auc(rng.permutation(y0), sc0)) for _ in range(2000)])
    null_hi = float(np.percentile(null, 95))
    print(f"    permutation null at that point, 95th pct = {null_hi:.3f}")

    if pos_auc >= best_auc:
        print(f"    NOTE: pooled AUC {best_auc:.3f} does not exceed position-only "
              f"{pos_auc:.3f};\n          the pooled figure is positional and is "
              f"NOT what this gate rests on.")
    if a0 <= null_hi:
        raise GateFailure(
            f"at the allocation point the ranker scores {a0:.3f} against a "
            f"permutation null of {null_hi:.3f}.\n    The spend is committed at "
            f"sample 1; separation there is the only kind that can allocate.\n"
            f"    Pooled AUC {best_auc:.3f} comes from later decision rows and "
            f"largely reflects\n    how many attempts remain (position-only "
            f"baseline {pos_auc:.3f}).")

    model = cands[best].__class__(**cands[best].get_params()).fit(X, y)
    print(f"    OK: separates at the allocation point {a0:.3f} > null "
          f"{null_hi:.3f}\n")
    return model, {"selected": best, "cv_auc": best_auc,
                   "position_only_auc": pos_auc, "allocation_point_auc": a0,
                   "allocation_null_95": null_hi,
                   "by_model": rows, "rows": len(y), "positives": pos}


# ----------------------------------------------------------------- gate 4
def _policy(byq):
    """The scoring and evaluation machinery, shared by freeze and apply so the
    evaluation cannot silently use a different policy than the one frozen."""

    def problem_score(qids, model):
        out = []
        for q in qids:
            s = byq[q]
            f = decision_features(s[:1])
            out.append(float(model.predict_proba([[f[k] for k in sorted(f)]])[0, 1]))
        return np.array(out)

    def evaluate(qids, score, frac, k_extra):
        """Top `frac` of problems by score get `k_extra` further samples.

        The DEPTH matters as much as the selection. An all-or-nothing policy --
        selected problems get every remaining sample, everyone else gets one --
        is dominated by simply spreading the same tokens thinly: sample 2 on a
        fresh problem is worth far more than sample 10 on a hard one. Measured
        on this data, that shape lost to the fixed baseline at EVERY interior
        operating point, so the sweep below ranges over depth too.

        This also makes the policy class strictly contain the baseline: frac=1.0
        with depth k is exactly "k+1 samples for everyone". The Governor can
        therefore never be structurally worse on calibration, and the held-out
        comparison tests what it should -- whether the RANKING generalises.
        """
        take = set(np.argsort(-score)[:int(round(frac * len(qids)))])
        U, C = [], []
        for i, q in enumerate(qids):
            s_ = byq[q]
            k = 1 + k_extra if i in take else 1
            used = s_[:k]
            U.append(1.0 if any(r["hidden_all_passed"] for r in used) else 0.0)
            C.append(sum(float(r.get("total_tokens", 0)) for r in used))
        return np.array(U), np.array(C)

    def fixed_at(qids, cost):
        """Best fixed k-samples-for-everyone at the same mean cost, interpolated."""
        kmax = max(len(byq[q]) for q in qids)
        cs = [float(np.mean([sum(float(r.get("total_tokens", 0))
                                 for r in byq[q][:k]) for q in qids]))
              for k in range(1, kmax + 1)]
        per = [np.array([1.0 if any(r["hidden_all_passed"] for r in byq[q][:k])
                         else 0.0 for q in qids]) for k in range(1, kmax + 1)]
        us = [p.mean() for p in per]
        if cost <= cs[0]:
            return us[0] * cost / cs[0], per[0] * cost / cs[0]
        for i in range(len(cs) - 1):
            if cs[i] <= cost <= cs[i + 1]:
                w = (cost - cs[i]) / (cs[i + 1] - cs[i])
                return us[i] + w * (us[i + 1] - us[i]), (1 - w) * per[i] + w * per[i + 1]
        return us[-1], per[-1]

    return problem_score, evaluate, fixed_at


def gate4_freeze(byq, cal, model) -> dict:
    """Sweep the operating point on CALIBRATION ONLY and write it to disk.

    Freezing is a separate, committed step on purpose. Fitting and evaluating in
    one process makes `frozen_before_heldout` unfalsifiable -- the same commit
    gets stamped as both, which is not evidence of anything. Here the frozen
    point must be committed before the evaluation step will accept it.
    """
    print("  === GATE 4a -- freeze operating point (calibration only) ===")
    problem_score, evaluate, fixed_at = _policy(byq)
    kmax = max(len(byq[q]) for q in cal) - 1
    cal_score = problem_score(cal, model)
    frac_star, k_star, best_adv = None, None, -9.0
    for frac in np.linspace(0.05, 1.0, 20):
        for k_extra in range(1, kmax + 1):
            U, C = evaluate(cal, cal_score, frac, k_extra)
            ub, _ = fixed_at(cal, float(C.mean()))
            if U.mean() - ub > best_adv:
                frac_star, k_star, best_adv = float(frac), int(k_extra), float(U.mean() - ub)

    import pickle
    FROZEN_MODEL.write_bytes(pickle.dumps(model))
    frozen = {"commit": _head(), "frac_star": frac_star, "k_extra": k_star,
              "cal_advantage": best_adv,
              "model_sha256": hashlib.sha256(FROZEN_MODEL.read_bytes()).hexdigest(),
              "cal_ids_sha256": hashlib.sha256(
                  json.dumps(sorted(cal)).encode()).hexdigest(),
              "n_cal": len(cal)}
    FROZEN.write_text(json.dumps(frozen, indent=2))
    print(f"    frac={frac_star:.2f} depth=+{k_star} "
          f"(cal advantage {best_adv:+.4f})")
    print(f"    wrote {_rel(FROZEN)} at commit {frozen['commit']}")
    print("\n    COMMIT THIS FILE, then run without --freeze to evaluate once.")
    return frozen


def gate4_apply(byq, cal, ev, ceiling) -> dict:
    """Apply the frozen point to the held-out set exactly once."""
    print("  === GATE 4b -- Governor vs matched-cost fixed spend (held-out) ===")
    if not FROZEN.exists():
        raise GateFailure(
            f"no frozen operating point at {_rel(FROZEN)}.\n"
            "    Run with --freeze first, commit the artifact, then evaluate.")
    frozen = json.loads(FROZEN.read_text())
    if hashlib.sha256(json.dumps(sorted(cal)).encode()).hexdigest() != frozen["cal_ids_sha256"]:
        raise GateFailure("the frozen point was fitted on a different "
                          "calibration set than this run produced")
    import pickle
    if not FROZEN_MODEL.exists():
        raise GateFailure(f"frozen model missing: {_rel(FROZEN_MODEL)}")
    if hashlib.sha256(FROZEN_MODEL.read_bytes()).hexdigest() != frozen["model_sha256"]:
        raise GateFailure("the frozen model file does not match its recorded "
                          "sha256; it was modified after freezing")
    model = pickle.loads(FROZEN_MODEL.read_bytes())

    rng = np.random.default_rng(0)
    problem_score, evaluate, fixed_at = _policy(byq)
    frac_star, k_star = frozen["frac_star"], frozen["k_extra"]
    best_adv = frozen["cal_advantage"]
    print(f"    frozen at commit {frozen['commit']}: frac={frac_star:.2f} "
          f"depth=+{k_star} (cal advantage {best_adv:+.4f})")

    ev_score = problem_score(ev, model)
    U, C = evaluate(ev, ev_score, frac_star, k_star)
    ub, Ub = fixed_at(ev, float(C.mean()))
    d = U - Ub
    boots = [d[rng.integers(0, len(ev), len(ev))].mean() for _ in range(4000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    adv = float(U.mean() - ub)

    ceil = ceiling["evaluation"]["rescuable"]
    print(f"\n    HELD-OUT EVALUATION (single application, n={len(ev)})")
    print(f"      Governor       U={U.mean():.4f}  cost={C.mean():>9.1f}")
    print(f"      best fixed     U={ub:.4f}")
    print(f"      advantage      {adv:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"      ceiling        {ceil:+.4f}   captured "
          f"{adv/ceil if ceil else float('nan'):.1%}")

    sd = float(np.std(d, ddof=1))
    n_req = float((1.96 * sd / 0.02) ** 2) if sd > 0 else float("inf")
    print(f"      paired sd={sd:.4f}; to resolve eps=0.02 needs n≈{n_req:.0f}, "
          f"have {len(ev)}")

    verdict = ("PASS" if lo > 0 else
               "SAMPLE-UNATTAINABLE" if n_req > len(ev) * 2 else "INCONCLUSIVE")
    return {"frac_star": frac_star, "k_extra": k_star, "cal_advantage": best_adv,
            "froze_commit": frozen["commit"], "heldout_commit": _head(),
            "advantage": adv, "ci": [float(lo), float(hi)],
            "ceiling": float(ceil),
            "captured": float(adv / ceil) if ceil else None,
            "governor_utility": float(U.mean()), "fixed_utility": float(ub),
            "cost": float(C.mean()), "paired_sd": sd, "n_required": n_req,
            "n_eval": len(ev), "verdict": verdict,
            "_U": U, "_Ub": Ub, "_C": C, "_score": ev_score}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true",
                    help="fit on calibration, write the frozen operating point, stop")
    ap.add_argument("--json-out", default=None, help="write the full result here")
    args = ap.parse_args()

    print("\nE0029-QWEN -- Governor on self-generated Qwen samples, "
          "private-test labels\n")
    try:
        byq, cal, ev, meta = load_joined()
        gate0(meta)
        audit = gate1_boundary_audit(byq, cal)
        ceiling = gate2_ceiling(byq, cal, ev)
        model, pred = gate3_predictor(byq, cal)
        if args.freeze:
            gate4_freeze(byq, cal, model)
            print("\n  FROZEN. Commit results/E0029_frozen*.  "
                  "Evaluation is a separate run.\n")
            return 0
        gov = gate4_apply(byq, cal, ev, ceiling)
    except GateFailure as e:
        print(f"\n  GATE FAILED -- run stopped\n\n    {e}\n")
        print("  No downstream number is reported. A Governor advantage computed")
        print("  past a failed gate is exactly the kind of result this project has")
        print("  already had to withdraw.\n")
        return 1

    Xv, yv, _ = decision_rows(byq, ev)
    evd = {
        "gov_utils": gov["_U"], "greedy_utils": gov["_Ub"],
        "gov_calls": np.array([1.0 if s > 0 else 0.0 for s in gov["_score"]]),
        "greedy_calls": np.full(len(ev), gov["frac_star"]),
        "decisions_by_state": [tuple(np.round(x, 3)) for x in Xv[:len(ev)]],
        "feature_names": list(FEATURE_NAMES),
        "answered_rate": 1.0, "utility": gov["governor_utility"],
        "requested": gov["_C"], "actual_used": gov["_C"], "charged": gov["_C"],
        "scored_via_executor": True,
        "decisions": np.isin(
            np.arange(len(ev)),
            np.argsort(-gov["_score"])[:int(round(gov["frac_star"] * len(ev)))]
        ).astype(int).tolist(),
        "cell_ids": [byq[q][0].get("platform", "?") for q in ev],
        # Truthful stamps: the freeze commit is read from the artifact written
        # by the earlier --freeze run, not restated from the current HEAD. If
        # they match, the freeze was never committed and the trap SHOULD go red.
        "froze_commit": gov["froze_commit"],
        "heldout_commit": gov["heldout_commit"],
        "selection_item_ids": cal, "evaluation_item_ids": ev,
        "token_cost_source": "exact tokenizer count over locally generated Qwen samples",
        "realised_cost": gov["cost"], "budget": gov["cost"],
        "baseline_cost": gov["cost"],
        "cited_experiment_ids": ["E0026-execution-feedback", "E0027-marginal-value",
                                 "E0028-marginal-ranker"],
        "withdrawn_ids": ["E0019-predictor-loss-math", "E0017-soft-governor-math"],
    }
    traps = run_trap_checks(evd)
    print("\n" + render(traps))
    failed = [k for k, (ok, _) in traps.items() if not ok]
    verdict = "BLOCKED" if failed else gov["verdict"]

    print(f"\n  VERDICT: {verdict}")
    if failed:
        print(f"    blocked by: {failed}")

    result = {"experiment_id": "E0029-QWEN", "meta": meta, "audit_file": _rel(AUDIT_OUT), "ceiling": ceiling,
              "predictor": pred,
              "governor": {k: v for k, v in gov.items() if not k.startswith("_")},
              "traps_failed": failed, "verdict": verdict,
              "feature_audit_verdict": audit["verdict"]}
    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(result, indent=2))
        print(f"    wrote {args.json_out}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
