#!/usr/bin/env python3
"""E0030 -- is the code-size signal real, or an artifact? CALIBRATION ONLY.

WHY THIS RUNS BEFORE ANY PREREGISTRATION IS EXECUTED.

The diagnostic reported `code_lines` at AUC 0.647 against a best-of-18-features
null of 0.646 and flagged it "clears the BEST-OF-25 null". True, and misleading:
the margin is +0.001. A binary flag on a threshold the statistic is sitting
exactly on is not evidence, and the model search that followed was a further
look on top of the feature search, so the honest threshold is higher still.

Spending the untouched 225-problem evaluation set on a +0.001 margin would
convert the project's only confirmatory resource into a coin flip.

So this asks whether there is anything worth spending it on, using calibration
only. Four tests, each able to kill the hypothesis:

  T1 TRUNCATION   generation was capped. A truncated program is long BY
                  CONSTRUCTION and fails BY CONSTRUCTION. If the signal lives
                  entirely in truncated samples it measures the token cap.
  T2 DIFFICULTY   if short code just means easy problems, the stated mechanism
                  ("a short program is a near miss") is false even where the
                  rule works, and `difficulty` would be the honest feature.
  T3 JOINT NULL   the real null for "best of F features, then best of M models",
                  built by permuting labels through the WHOLE search.
  T4 POLICY       AUC is not advantage. Simulate the actual allocation on
                  calibration and bootstrap it. A ranking can discriminate and
                  still buy nothing.

Nothing here touches evaluation. Nothing here is a result.

Run:  python scripts/e0030_validate.py
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from governor.execfeedback.richfeatures import decision_features
from governor.models.calibration import auc

import scripts.e0029_analyse as A

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / "results" / "e0029_problems.json"
N_PERM = 4000
FEATURE = "code_lines"


def calibration_frame(byq, cal):
    """One row per problem where sample 1 failed: the allocation decision."""
    rows = []
    for q in cal:
        s = byq[q]
        if s[0]["hidden_all_passed"]:
            continue                                   # nothing to buy
        f = decision_features(s[:1])
        rows.append({
            "qid": q,
            "code_lines": f["code_lines"],
            "code_chars": f["code_chars"],
            "pub_frac": float(s[0].get("pub_frac", 0.0)),
            "completion_tokens": float(s[0].get("completion_tokens", 0.0)),
            "truncated": bool(s[0].get("output_truncated", 0.0)),
            "y": 1.0 if any(r["hidden_all_passed"] for r in s[1:]) else 0.0,
        })
    return rows


def _auc(y, x):
    a = float(auc(np.asarray(y), np.asarray(x)))
    return max(a, 1.0 - a)


def t1_truncation(rows, cap_tokens):
    print("  " + "=" * 66)
    print("  T1 -- is the signal just the generation cap?")
    print("  " + "=" * 66)
    tok = np.array([r["completion_tokens"] for r in rows])
    near_cap = tok >= (cap_tokens - 10)
    flagged = np.array([r["truncated"] for r in rows])
    hit = near_cap | flagged
    print(f"  generation cap        {cap_tokens} tokens")
    print(f"  at/near the cap       {int(hit.sum())}/{len(rows)} "
          f"({hit.mean():.1%})")
    if hit.sum():
        yy = np.array([r["y"] for r in rows])
        print(f"  rescue rate, capped   {yy[hit].mean():.3f}")
        print(f"  rescue rate, not      {yy[~hit].mean():.3f}")

    keep = ~hit
    if keep.sum() < 40 or len(set(np.array([r['y'] for r in rows])[keep])) < 2:
        print("  too few uncapped rows to re-test -- INCONCLUSIVE")
        return None
    y2 = np.array([r["y"] for r in rows])[keep]
    x2 = np.array([r[FEATURE] for r in rows])[keep]
    a_all = _auc([r["y"] for r in rows], [r[FEATURE] for r in rows])
    a_cut = _auc(y2, x2)
    print(f"\n  {FEATURE} AUC, all rows      {a_all:.3f}  (n={len(rows)})")
    print(f"  {FEATURE} AUC, uncapped only {a_cut:.3f}  (n={int(keep.sum())})")
    if a_cut < 0.55:
        print("  -> the signal COLLAPSES once capped samples are removed.")
        print("     It was measuring the token limit, not the problem.")
    elif a_cut >= a_all - 0.02:
        print("  -> survives. Not an artifact of the cap.")
    else:
        print("  -> weakened but not gone; the cap explains part of it.")
    return a_cut


def t2_difficulty(rows):
    print("\n  " + "=" * 66)
    print("  T2 -- is short code just easy problems?")
    print("  " + "=" * 66)
    if not PROBLEMS.exists():
        print("  e0029_problems.json absent -- skipped")
        return
    diff = {p["qid"]: p.get("difficulty", "?")
            for p in json.loads(PROBLEMS.read_text())}
    by = collections.defaultdict(list)
    for r in rows:
        by[diff.get(r["qid"], "?")].append(r)

    print(f"  {'difficulty':10s} {'n':>4} {'rescue':>7} {'mean lines':>11}")
    for d in sorted(by):
        v = by[d]
        print(f"  {d:10s} {len(v):4d} "
              f"{np.mean([r['y'] for r in v]):7.3f} "
              f"{np.mean([r[FEATURE] for r in v]):11.1f}")

    # If code_lines only works BETWEEN difficulty strata, it is a proxy for
    # difficulty. If it works WITHIN each stratum, it carries its own signal.
    print(f"\n  {FEATURE} AUC within each difficulty stratum:")
    kept = []
    for d in sorted(by):
        v = by[d]
        ys = [r["y"] for r in v]
        if len(v) < 25 or len(set(ys)) < 2:
            print(f"    {d:10s} n={len(v):3d}  too small")
            continue
        a = _auc(ys, [r[FEATURE] for r in v])
        kept.append((d, a, len(v)))
        print(f"    {d:10s} n={len(v):3d}  AUC={a:.3f}")
    if kept:
        w = sum(a * n for _, a, n in kept) / sum(n for _, _, n in kept)
        print(f"\n  pooled-within-difficulty AUC = {w:.3f}")
        if w < 0.55:
            print("  -> within strata it does nothing. It is a difficulty proxy,")
            print("     and `difficulty` would be the honest feature to use.")
        else:
            print("  -> holds within strata. Not merely a difficulty proxy.")


def t3_joint_null(byq, cal, rows):
    print("\n  " + "=" * 66)
    print("  T3 -- the null for the SEARCH that was actually run")
    print("  " + "=" * 66)
    names = sorted(decision_features(byq[cal[0]][:1]))
    X = np.array([[decision_features(byq[r["qid"]][:1])[k] for k in names]
                  for r in rows])
    y = np.array([r["y"] for r in rows])
    live = [j for j in range(X.shape[1]) if not np.allclose(X[:, j], X[0, j])]
    obs = max(_auc(y, X[:, j]) for j in live)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(N_PERM):
        yp = rng.permutation(y)
        null.append(max(_auc(yp, X[:, j]) for j in live))
    hi = float(np.percentile(null, 95))
    p = float((np.array(null) >= obs).mean())
    print(f"  live features            {len(live)}")
    print(f"  best observed AUC        {obs:.3f}")
    print(f"  null 95th pct            {hi:.3f}")
    print(f"  permutation p-value      {p:.3f}")
    print(f"  margin                   {obs - hi:+.3f}")
    if p >= 0.05:
        print("\n  -> NOT significant once the feature search is accounted for.")
        print("     Reporting the best of many features as if it were chosen in")
        print("     advance is the error this test exists to prevent.")
    else:
        print("\n  -> survives the feature search.")
    return p


def t4_policy(byq, cal, rows):
    print("\n  " + "=" * 66)
    print("  T4 -- does the ranking BUY anything? (calibration simulation)")
    print("  " + "=" * 66)
    order = {r["qid"]: r[FEATURE] for r in rows}
    qids = list(cal)
    U1 = np.array([1.0 if byq[q][0]["hidden_all_passed"] else 0.0 for q in qids])
    # shortest first among unsolved; solved-at-1 never gets extra spend
    score = np.array([-order.get(q, np.inf) if q in order else -np.inf
                      for q in qids])

    def sim(frac, k):
        take = set(np.argsort(-score)[:int(round(frac * len(qids)))])
        U, C = [], []
        for i, q in enumerate(qids):
            s = byq[q]
            used = s[:1 + k] if i in take else s[:1]
            U.append(1.0 if any(r["hidden_all_passed"] for r in used) else 0.0)
            C.append(sum(float(r.get("total_tokens", 0)) for r in used))
        return np.array(U), np.array(C)

    kmax = max(len(byq[q]) for q in qids) - 1
    cs = [float(np.mean([sum(float(r.get("total_tokens", 0))
                             for r in byq[q][:k]) for q in qids]))
          for k in range(1, kmax + 2)]
    per = [np.array([1.0 if any(r["hidden_all_passed"] for r in byq[q][:k])
                     else 0.0 for q in qids]) for k in range(1, kmax + 2)]
    us = [p.mean() for p in per]

    def fixed(cost):
        if cost <= cs[0]:
            return us[0] * cost / cs[0], per[0] * cost / cs[0]
        for i in range(len(cs) - 1):
            if cs[i] <= cost <= cs[i + 1]:
                w = (cost - cs[i]) / (cs[i + 1] - cs[i])
                return us[i] + w * (us[i+1] - us[i]), (1-w)*per[i] + w*per[i+1]
        return us[-1], per[-1]

    rng = np.random.default_rng(0)
    best = None
    print(f"  {'frac':>5} {'k':>3} {'gov U':>7} {'fixed':>7} {'adv':>8} "
          f"{'95% CI':>20}")
    for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
        for k in (1, 2, 3, 5, 9):
            if k > kmax:
                continue
            U, C = sim(frac, k)
            ub, Ub = fixed(float(C.mean()))
            d = U - Ub
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1500)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            adv = float(U.mean() - ub)
            if best is None or adv > best[0]:
                best = (adv, frac, k, lo, hi)
            if frac in (0.4, 0.8) and k in (2, 5):
                print(f"  {frac:5.1f} {k:3d} {U.mean():7.4f} {ub:7.4f} "
                      f"{adv:+8.4f}   [{lo:+.4f},{hi:+.4f}]")
    adv, frac, k, lo, hi = best
    print(f"\n  best on calibration: frac={frac:.1f} depth=+{k}  "
          f"advantage {adv:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    print("  (this is the SELECTED maximum on the set it was chosen from, so it")
    print("   is optimistic by construction -- it is an upper bound on what")
    print("   evaluation could show, not a prediction)")
    if hi <= 0:
        print("\n  -> the ranking buys NOTHING even where it was fitted.")
    elif adv < 0.01:
        print(f"\n  -> best advantage {adv:+.4f} against a ceiling of +0.1378:")
        print("     under 10% of the headroom, on the set it was chosen from.")
    return best


def main() -> int:
    print("\nE0030 VALIDATION -- calibration only, evaluation untouched\n")
    byq, cal, ev, meta = A.load_joined()
    if (meta["cal"], meta["eval"]) != (A.EXPECTED_CAL, A.EXPECTED_EVAL):
        raise SystemExit(f"wrong dataset: {meta['cal']}/{meta['eval']}")
    rows = calibration_frame(byq, cal)
    pos = int(sum(r["y"] for r in rows))
    print(f"  calibration decisions: n={len(rows)}  rescued={pos} "
          f"({pos/len(rows):.1%})\n")

    cap = 2500
    cfg = ROOT / "configs" / "e0029_split.json"
    if cfg.exists():
        cap = int(json.loads(cfg.read_text()).get("max_completion_tokens", cap))

    t1_truncation(rows, cap)
    t2_difficulty(rows)
    p = t3_joint_null(byq, cal, rows)
    t4_policy(byq, cal, rows)

    print("\n  " + "=" * 66)
    print("  VERDICT")
    print("  " + "=" * 66)
    if p is not None and p >= 0.05:
        print("  DO NOT SPEND THE EVALUATION SET.")
        print("  The feature does not survive its own search on calibration.")
    else:
        print("  The feature survives calibration. Read T1 and T2 before")
        print("  deciding whether the MECHANISM is what the preregistration says.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
