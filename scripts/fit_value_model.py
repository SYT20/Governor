#!/usr/bin/env python3
"""Stage 3: fit the value model, settle D1/D2, and run the hardest gate.

    python3 scripts/fit_value_model.py

The gate: beat a train-base-rate predictor on BOTH Brier and ECE, evaluated on
families the model never saw. Fail it and no policy gets built -- section 3's
fallback applies and the contribution becomes the negative result.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.corpus.build import Checkpoint  # noqa: E402
from governor.models.calibration import evaluate  # noqa: E402
from governor.models.value import (  # noqa: E402
    EXCLUDED_FEATURES,
    audit_terminal_dependence,
    fit_composed,
    fit_model,
    usable_feature_names,
)


def load(db: str) -> list[Checkpoint]:
    conn = sqlite3.connect(db)
    rows = conn.execute(
        """SELECT episode_id,decision_id,family,split,seed,action,mode,tier,
                  was_random,n_admissible,features,label FROM checkpoints"""
    ).fetchall()
    conn.close()
    return [
        Checkpoint(
            episode_id=r[0], decision_id=r[1], family=r[2], split=r[3], seed=r[4],
            action=r[5], mode=r[6], tier=r[7], was_random=bool(r[8]),
            n_admissible=r[9], features=json.loads(r[10]), label=r[11],
        )
        for r in rows
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="results/corpus.db")
    p.add_argument("--tune", type=int, default=0, help="optuna trials (0 = skip)")
    args = p.parse_args()

    print("=" * 84)
    print("GOVERNOR — Stage 3: value model, calibration, and the base-rate gate")
    print("=" * 84)

    ck = load(args.db)
    train = [c for c in ck if c.split == "train"]
    held = [c for c in ck if c.split == "heldout"]
    names = usable_feature_names(train)
    tr_base = float(np.mean([c.label for c in train]))

    print(f"\n[1] Data")
    print(f"    train    {len(train):>5} checkpoints  {len(set(c.episode_id for c in train)):>4} episodes  "
          f"{len(set(c.family for c in train))} families")
    print(f"    held-out {len(held):>5} checkpoints  {len(set(c.episode_id for c in held)):>4} episodes  "
          f"{len(set(c.family for c in held))} families "
          f"({', '.join(sorted(set(c.family for c in held)))})")
    print(f"    features ({len(names)}): {', '.join(names)}")
    print(f"    excluded: {', '.join(sorted(EXCLUDED_FEATURES))}  "
          f"(family fingerprint — would leak the regime)")
    print(f"    train base rate = {tr_base:.4f}  -> this is the number to beat")

    # ---- fit the candidates -------------------------------------------------
    print("\n[2] Fitting candidates (all CV grouped by episode)")
    models = []
    for kind in ("logistic", "gbm"):
        m = fit_model(train, kind=kind, uses_actions=True, data_version="corpus-v1")
        models.append(m)
        print(f"    {m.name:<16} fitted   n_effective={m.n_effective} episodes")
    comp = fit_composed(train, kind="logistic", data_version="corpus-v1")
    models.append(comp)
    cov = {k: v for k, v in sorted(comp.effect.n_obs.items())}
    print(f"    {comp.name:<16} fitted   effect model from randomised transitions only")
    print(f"      randomised transitions per action class: {cov}")

    # ---- D2: hyperparameter search on TRAIN families only -------------------
    if args.tune:
        from governor.models.tune import tune
        print(f"\n[2b] Optuna search ({args.tune} trials), leave-one-family-out on TRAIN only")
        print("     The held-out families are never seen here. Tuning against them would")
        print("     turn the gate into a training objective and it would measure nothing.")
        tr = tune(train, n_trials=args.tune)
        print(tr.render())
        bp = dict(tr.best_params)
        kind = bp.pop("kind"); ncf = bp.pop("n_calib_folds", 4)
        tuned = fit_model(train, kind=kind, uses_actions=True,
                          data_version="corpus-v1-tuned", n_calib_folds=ncf,
                          estimator_kwargs=bp)
        tuned.name = f"Q_{kind}_tuned"
        models.append(tuned)

    # ---- the gate -----------------------------------------------------------
    print("\n[3] Held-out families — the gate (must beat base rate on Brier AND ECE)")
    results = {}
    for m in models:
        pred = m.predict(held)
        rep = evaluate([c.label for c in held], [float(x) for x in pred],
                       train_base_rate=tr_base)
        results[m.name] = (rep, pred)
        print()
        print(rep.render(f"{m.name}   {'PASS' if rep.beats_base_rate else 'FAIL'}"))

    # ---- in-corpus comparison, to expose overfitting -------------------------
    print("\n[4] In-corpus vs held-out Brier skill (gap reveals regime memorisation)")
    print(f"    {'model':<16} {'in-corpus':>11} {'held-out':>10} {'gap':>9}")
    for m in models:
        ins = evaluate([c.label for c in train], [float(x) for x in m.predict(train)],
                       train_base_rate=tr_base)
        out = results[m.name][0]
        gap = ins.skill["brier"] - out.skill["brier"]
        print(f"    {m.name:<16} {ins.skill['brier']:>+11.1%} {out.skill['brier']:>+10.1%} "
              f"{gap:>+9.1%}")

    # ---- leakage audit ------------------------------------------------------
    print("\n[5] Leakage audit — where does the skill actually come from?")
    best_name = max(results, key=lambda k: results[k][0].skill["brier"])
    aud = audit_terminal_dependence(held, results[best_name][1])
    print(f"    best model: {best_name}")
    for seg, label in (("pre_resolution", "before any passing test"),
                       ("post_pass", "after a passing test")):
        d = aud.get(seg) or {}
        if d:
            print(f"      {label:<26} n={d['n']:<5} brier_skill={d['brier_skill']:>+7.1%} "
                  f"auc={d['auc']:.3f} base={d['base_rate']:.2f}")
        else:
            print(f"      {label:<26} too few rows to score")
    pre = (aud.get("pre_resolution") or {}).get("brier_skill")
    if pre is not None and pre <= 0:
        print("      WARNING: all skill comes from post-verification states. The model")
        print("               reads a thermometer; it does not forecast. A policy built")
        print("               on it gets no guidance early, which is when it matters.")

    # ---- reliability --------------------------------------------------------
    print(f"\n[6] Reliability of {best_name} on held-out families")
    print(f"    {'bin':<12} {'n':>6} {'predicted':>10} {'observed':>10} {'gap':>8}")
    for b in results[best_name][0].bins:
        if b["n"]:
            print(f"    [{b['lo']:.1f},{b['hi']:.1f}){'':<4} {b['n']:>6} {b['conf']:>10.3f} "
                  f"{b['acc']:>10.3f} {b['gap']:>8.3f}")

    # ---- does the ranking transfer while only the mapping shifts? -----------
    print("\n[6b] Regime-shift diagnostic: recalibrate on a slice of the new regime")
    print("     AUC transfers but ECE does not, which points at a shifted mapping")
    print("     rather than a broken model. Testing that directly: refit ONLY the")
    print("     isotonic mapping on 25% of held-out episodes, score the other 75%.")
    from sklearn.linear_model import LogisticRegression as _LR

    def _logit(v):
        v = np.clip(np.asarray(v, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(v / (1 - v)).reshape(-1, 1)

    # Calibrate WITHIN each held-out family. Episode ids sort by family name, so a
    # naive head-slice calibrates on one regime and scores on the others -- which
    # is a different (and much harder) question than the deployment one. What a
    # policy actually faces is: arrive in a new regime, spend a few episodes, then
    # recalibrate against *that* regime.
    cal, tst = [], []
    for fam in sorted({c.family for c in held}):
        eps = sorted({c.episode_id for c in held if c.family == fam})
        k = max(1, len(eps) // 4)
        cal_eps = set(eps[:k])
        cal += [c for c in held if c.family == fam and c.episode_id in cal_eps]
        tst += [c for c in held if c.family == fam and c.episode_id not in cal_eps]

    print(f"     calibration slice: {len(set(c.episode_id for c in cal))} episodes "
          f"/ {len(cal)} checkpoints (stratified across all 3 families)")
    print(f"     scored on:         {len(set(c.episode_id for c in tst))} episodes "
          f"/ {len(tst)} checkpoints")
    print(f"     {'model':<16} {'ece before':>11} {'ece after':>10} {'floor p95':>10} "
          f"{'brier skill':>12}")
    recal_pass = []
    for m in models:
        # CONTROL: the same test slice, WITHOUT recalibration. Without this the
        # before/after comparison is between different row sets and says nothing.
        r0 = evaluate([c.label for c in tst], [float(x) for x in m.predict(tst)],
                      train_base_rate=tr_base)
        # Platt scaling, not isotonic. m.predict() is ALREADY isotonic-calibrated;
        # stacking a second isotonic fit on ~60 episodes overfits into a coarse step
        # function and made ECE worse than doing nothing. A 2-parameter sigmoid is
        # the right recalibrator for a slice this small.
        platt = _LR(max_iter=1000)
        platt.fit(_logit(m.predict(cal)), [c.label for c in cal])
        p2 = [float(x) for x in platt.predict_proba(_logit(m.predict(tst)))[:, 1]]
        r2 = evaluate([c.label for c in tst], p2, train_base_rate=tr_base)
        ok = r2.beats_base_rate
        recal_pass.append((m.name, ok))
        print(f"     {m.name:<16} {r0.model['ece']:>11.4f} {r2.model['ece']:>10.4f} "
              f"{r2.noise_floor['p95']:>10.4f} {r2.skill['brier']:>+12.1%}")

    # ---- verdict ------------------------------------------------------------
    passed = [n for n, (r, _) in results.items() if r.beats_base_rate]
    print("\n[7] Decisions")
    d1 = "direct Q" if best_name.startswith("Q") else "V composed with T"
    d2 = "gbm" if "gbm" in best_name else "logistic"
    print(f"    D1 (estimator)  -> {d1:<22} ({best_name} wins on held-out Brier skill)")
    print(f"    D2 (model type) -> {d2}")

    ok = bool(passed)
    print(f"\n[8] GATE: {'PASS' if ok else 'FAIL'} — "
          f"{len(passed)}/{len(results)} models beat the base rate on held-out families")
    if not ok:
        print("    Stage 4 must not begin. Section 3 fallback: the contribution becomes")
        print("    the negative result, with the corpus and methodology as deliverable.")
    print()

    Path("results").mkdir(exist_ok=True)
    Path("results/stage3_calibration.json").write_text(json.dumps(
        {n: {"model": r.model, "baseline": r.baseline, "skill": r.skill,
             "beats_base_rate": r.beats_base_rate, "bins": r.bins}
         for n, (r, _) in results.items()}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
