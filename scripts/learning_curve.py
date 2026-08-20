#!/usr/bin/env python3
"""Does Stage 3's calibration failure shrink with more data, or is it structural?

    python3 scripts/learning_curve.py

Stage 3 found strong discrimination (AUC 0.86) alongside held-out ECE well above
the perfect-calibration noise floor. Two explanations fit that evidence equally well
and imply opposite next steps:

  undertrained   the mapping needs more episodes -> collect more, the gate passes
                 on data rather than on redefinition
  untransferable the mapping does not cross regimes at any n -> section 3's
                 fallback applies and the negative result is the contribution

A learning curve separates them. Subsampling is by EPISODE and stratified by family,
because episodes are the unit of information here (labels are episode-constant, so
effective sample size is the episode count -- see J.8). Hyperparameters are held
fixed across every point so the curve isolates data quantity from tuning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.corpus.build import Checkpoint  # noqa: E402
from governor.models.calibration import evaluate  # noqa: E402
from governor.models.value import fit_model  # noqa: E402
from scripts.fit_value_model import load  # noqa: E402

# Held fixed across the curve: the winning configuration from Stage 3's optuna
# search. Re-tuning at each point would confound "more data helps" with
# "different hyperparameters help".
TUNED_GBM = dict(max_depth=5, learning_rate=0.2026866305748189, max_iter=100,
                 min_samples_leaf=20, l2_regularization=0.0684044923060367)


def subsample_by_episode(train: list[Checkpoint], n_episodes: int) -> list[Checkpoint]:
    """Take n episodes, spread evenly across families.

    Subsampling checkpoints instead would be wrong twice over: it would break
    episodes apart, and it would pretend the effective sample size grew when it
    did not.
    """
    fams = sorted({c.family for c in train})
    per = max(1, n_episodes // len(fams))
    keep: set[str] = set()
    for f in fams:
        eps = sorted({c.episode_id for c in train if c.family == f})
        keep.update(eps[:per])
    return [c for c in train if c.episode_id in keep]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="results/corpus_big.db")
    ap.add_argument("--points", default="125,250,500,1000,2000")
    args = ap.parse_args()

    ck = load(args.db)
    train = [c for c in ck if c.split == "train"]
    held = [c for c in ck if c.split == "heldout"]
    y_held = [c.label for c in held]
    n_train_eps = len({c.episode_id for c in train})

    print("=" * 84)
    print("GOVERNOR — Stage 3b: calibration learning curve")
    print("=" * 84)
    print(f"\n  corpus: {n_train_eps} train episodes across "
          f"{len({c.family for c in train})} families")
    print(f"  held-out: {len({c.episode_id for c in held})} episodes across "
          f"{len({c.family for c in held})} families (never fitted on)")
    print("\n  Question: as episodes increase, does held-out ECE fall toward the")
    print("  noise floor (undertrained) or stay flat above it (untransferable)?\n")

    points = [int(x) for x in args.points.split(",") if int(x) <= n_train_eps]
    rows = []
    print(f"  {'episodes':>9} {'model':<14} {'AUC':>7} {'brier skill':>12} "
          f"{'ECE':>8} {'floor':>8} {'ECE/floor':>10}")
    print("  " + "-" * 74)

    for n in points:
        sub = subsample_by_episode(train, n)
        n_eps = len({c.episode_id for c in sub})
        base = float(np.mean([c.label for c in sub]))
        for label, kind, kw in (("Q_gbm_tuned", "gbm", TUNED_GBM),
                                ("Q_logistic", "logistic", {"C": 0.5})):
            m = fit_model(sub, kind=kind, uses_actions=True,
                          data_version=f"lc-{n_eps}", n_calib_folds=3,
                          estimator_kwargs=kw)
            p = [float(x) for x in m.predict(held)]
            r = evaluate(y_held, p, train_base_rate=base)
            ratio = r.model["ece"] / max(r.noise_floor["p95"], 1e-9)
            rows.append({"episodes": n_eps, "model": label, "auc": r.model["auc"],
                         "brier_skill": r.skill["brier"], "ece": r.model["ece"],
                         "floor_p95": r.noise_floor["p95"], "ece_over_floor": ratio})
            print(f"  {n_eps:>9} {label:<14} {r.model['auc']:>7.3f} "
                  f"{r.skill['brier']:>+12.1%} {r.model['ece']:>8.4f} "
                  f"{r.noise_floor['p95']:>8.4f} {ratio:>10.2f}x")
        print()

    # ---- verdict --------------------------------------------------------------
    print("  " + "=" * 74)
    print("  Trend (ECE as a multiple of the noise floor; 1.0x = as calibrated as")
    print("  this much held-out data can demonstrate)\n")
    verdicts = {}
    for label in ("Q_gbm_tuned", "Q_logistic"):
        seq = [r for r in rows if r["model"] == label]
        if len(seq) < 2:
            continue
        first, last = seq[0], seq[-1]
        trend = last["ece_over_floor"] - first["ece_over_floor"]
        halves = len(seq) // 2
        early = np.mean([r["ece_over_floor"] for r in seq[:halves]])
        late = np.mean([r["ece_over_floor"] for r in seq[-halves:]])
        improving = late < early - 0.10
        verdicts[label] = improving
        arrow = "falling" if improving else ("flat" if abs(trend) < 0.3 else "rising")
        print(f"    {label:<14} {first['episodes']:>5}ep {first['ece_over_floor']:.2f}x  ->  "
              f"{last['episodes']:>5}ep {last['ece_over_floor']:.2f}x   [{arrow}]")
        print(f"    {'':<14} first-half mean {early:.2f}x   second-half mean {late:.2f}x")

    print()
    if any(verdicts.values()):
        print("  VERDICT: UNDERTRAINED — calibration improves with episodes.")
        print("           Collect more and re-run the Stage 3 gate. Do not redefine it.")
    else:
        print("  VERDICT: NOT EXPLAINED BY DATA VOLUME — the gap does not close as")
        print("           episodes grow. Discrimination transfers across regimes and the")
        print("           probability mapping does not. Section 3's fallback applies:")
        print("           this is the finding, reproduced in a setting where the cause")
        print("           is isolable, and Stage 4 must not assume calibrated absolute")
        print("           probabilities. Ranking-based action choice remains defensible;")
        print("           the STOP rule and the cost tradeoff do not.")
    print()

    Path("results").mkdir(exist_ok=True)
    Path("results/learning_curve.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
