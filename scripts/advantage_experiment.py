#!/usr/bin/env python3
"""Stage 4B: does an explicit advantage model rank ACTIONS better than pooled Q?

    python3 scripts/advantage_experiment.py

Stage 4A established that the pooled Q model ranks states well (AUC 0.75) and
actions poorly (0.58 overall, 0.52 on training regimes). This tests the fix the
diagnosis implies: fit V(s) first, then regress the residual on the action, using
only randomised decisions where the action is unconfounded with the state.

Branch points are collected once and cached, so both models are scored on the
identical set of counterfactuals -- otherwise the comparison measures which model
got the easier states.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.accounting.meter import Envelope  # noqa: E402
from governor.envs.families import heldout_families, train_families  # noqa: E402
from governor.envs.synthbug import Action, Mode, SynthConfig, Tier  # noqa: E402
from governor.experiments.branch import (  # noqa: E402
    BranchPoint,
    collect_branch_points,
    decision_regret,
    kendall_tau,
    oracle_spread,
    pairwise_ranking,
)
from governor.models.calibration import auc  # noqa: E402
from governor.models.value import fit_advantage, fit_model  # noqa: E402
from scripts.build_corpus import fit_profile_over_families  # noqa: E402
from scripts.fit_value_model import load  # noqa: E402
from scripts.run_experiment import fit_channels  # noqa: E402

TUNED_GBM = dict(max_depth=5, learning_rate=0.2026866305748189, max_iter=100,
                 min_samples_leaf=20, l2_regularization=0.0684044923060367)
CACHE = Path("results/branch_points.json")


def parse_action(s: str) -> Action:
    mode_s, rest = s.split("@")
    tier_s = rest.split("->")[0]
    tgt = int(rest.split("->h")[1]) if "->h" in rest else None
    return Action(Mode(mode_s), Tier(tier_s), tgt)


def build_or_load(args) -> list[BranchPoint]:
    if CACHE.exists() and not args.rebuild:
        raw = json.loads(CACHE.read_text())
        pts = [BranchPoint(**{k: v for k, v in r.items() if k != "predicted"}) for r in raw]
        print(f"    loaded {len(pts)} cached branch points from {CACHE}")
        return pts

    profile = fit_profile_over_families(train_families(), 40)
    channels, _ = fit_channels(SynthConfig(), 120)
    envelope = Envelope(tokens=60_000, cost=0.30, wall_s=200.0, tool_calls=25)
    fams = [("train", f) for f in train_families()[: args.families]]
    fams += [("heldout", f) for f in heldout_families()]

    pts: list[BranchPoint] = []
    for split, fam in fams:
        got = collect_branch_points(
            family=fam, split=split, seeds=range(90_000, 90_000 + args.episodes),
            profile=profile, channels=channels, envelope=envelope,
            n_replicates=args.replicates,
        )
        pts += got
        print(f"    {split:<8} {fam.name:<18} {len(got):>4} branch points")
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps([{
        "family": p.family, "split": p.split, "episode_seed": p.episode_seed,
        "decision_id": p.decision_id, "features": p.features,
        "realised": p.realised, "n_replicates": p.n_replicates} for p in pts], indent=1))
    return pts


def score(points: list[BranchPoint], model, label: str, se: float) -> dict:
    for bp in points:
        bp.predicted = {}
        for a in bp.realised:
            bp.predicted[a] = float(model.predict([bp.checkpoint_for(parse_action(a))])[0])
    out = {}
    for split in ("train", "heldout", "ALL"):
        pts = points if split == "ALL" else [p for p in points if p.split == split]
        if not pts:
            continue
        out[split] = {
            "loose": pairwise_ranking(pts, min_gap=0.10),
            "strict": pairwise_ranking(pts, min_gap=round(2 * se, 2)),
            "tau": kendall_tau(pts),
            "regret": decision_regret(pts),
        }
    ys, ps = [], []
    for bp in points:
        for a, v in bp.realised.items():
            ys.append(1 if v >= 0.5 else 0)
            ps.append(bp.predicted[a])
    out["pooled_auc"] = auc(ys, ps)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="results/corpus_big.db")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--replicates", type=int, default=16)
    ap.add_argument("--families", type=int, default=4)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    print("=" * 84)
    print("GOVERNOR — Stage 4B: explicit advantage model vs pooled Q")
    print("=" * 84)

    ck = load(args.db)
    train = [c for c in ck if c.split == "train"]
    n_rand = sum(1 for c in train if c.was_random)
    print(f"\n[1] Corpus: {len({c.episode_id for c in train})} episodes, "
          f"{len(train)} checkpoints, {n_rand} randomised "
          f"({n_rand/len(train):.0%}) — stage 2 fits on the randomised subset only")

    print("\n[2] Fitting both models on identical data")
    pooled = fit_model(train, kind="gbm", uses_actions=True, data_version="big",
                       n_calib_folds=3, estimator_kwargs=TUNED_GBM)
    pooled.name = "pooled_Q"
    adv = fit_advantage(train, data_version="big")
    print(f"    pooled_Q          P(success | state, action) in one model")
    print(f"    V_plus_advantage  V(s) then residual regression on randomised rows")

    print("\n[3] Branch points (collected once, both models scored on the same set)")
    points = build_or_load(args)
    se = 0.707 / np.sqrt(args.replicates)
    sp = oracle_spread(points)
    print(f"    {len(points)} points   mean spread {sp.get('mean_spread',0):.3f}   "
          f"decision-relevant {sp.get('frac_states_with_real_choice',0):.0%}")

    print("\n[4] Within-state action ranking  (0.500 = cannot rank actions at all)")
    results = {}
    for m in (pooled, adv):
        results[m.name] = score(points, m, m.name, se)
    print(f"\n    {'model':<18} {'split':<9} {'pooled AUC':>11} {'pairwise':>10} "
          f"{'strict':>8} {'tau':>8} {'regret':>8}")
    print("    " + "-" * 76)
    for name, r in results.items():
        for split in ("train", "heldout", "ALL"):
            if split not in r:
                continue
            d = r[split]
            pa = f"{r['pooled_auc']:.3f}" if split == "ALL" else ""
            print(f"    {name if split=='train' else '':<18} {split:<9} {pa:>11} "
                  f"{d['loose']['accuracy']:>10.3f} {d['strict']['accuracy']:>8.3f} "
                  f"{d['tau']:>8.3f} {d['regret'].get('mean',0):>8.3f}")
        print()

    # ---- verdict --------------------------------------------------------------
    a = results["pooled_Q"]["ALL"]["loose"]["accuracy"]
    b = results["V_plus_advantage"]["ALL"]["loose"]["accuracy"]
    n_pairs = results["pooled_Q"]["ALL"]["loose"]["pairs"]
    se_acc = float(np.sqrt(0.25 / max(n_pairs, 1)))
    print("[5] VERDICT")
    print(f"    pooled Q          {a:.3f}")
    print(f"    V + advantage     {b:.3f}")
    print(f"    difference        {b-a:+.3f}   (SE on each ~{se_acc:.3f}; pairs are")
    print(f"                      clustered within state, so the true SE is larger)")
    if b - a > 2 * se_acc:
        print("    The explicit advantage decomposition improves action ranking.")
        print("    Adopt V + A as the policy's scoring model (Decision Record G.2).")
    elif abs(b - a) <= 2 * se_acc:
        print("    No detectable difference. The decomposition does not rescue action")
        print("    ranking, so the limitation is the state representation itself, not")
        print("    how the target is factored. Adding features that describe the ACTION's")
        print("    expected effect is the next lever, not a different estimator.")
    else:
        print("    The advantage model is WORSE. Residual regression on the randomised")
        print("    subset is noisier than direct fitting at this sample size.")

    Path("results/stage4b.json").write_text(json.dumps(results, indent=2, default=float))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
