#!/usr/bin/env python3
"""Stage 4A: can the model rank ACTIONS, or only rank STATES?

    python3 scripts/branch_experiment.py

Stage 3's AUC ~0.84 was over-read as evidence for action selection. Pooled AUC is
dominated by easy-state / hard-state separation; a model can achieve it while
ordering actions at a fixed state no better than chance. This script measures the
quantity the policy actually depends on, using live branched counterfactuals from
identical states.

The headline number is pairwise within-state ranking accuracy. 0.5 means the model
cannot rank actions at all.
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
from governor.envs.synthbug import SynthConfig  # noqa: E402
from governor.experiments.branch import (  # noqa: E402
    collect_branch_points,
    decision_regret,
    kendall_tau,
    oracle_spread,
    pairwise_ranking,
)
from governor.models.calibration import auc  # noqa: E402
from governor.models.value import fit_model  # noqa: E402
from scripts.build_corpus import fit_profile_over_families  # noqa: E402
from scripts.fit_value_model import load  # noqa: E402
from scripts.run_experiment import fit_channels  # noqa: E402

TUNED_GBM = dict(max_depth=5, learning_rate=0.2026866305748189, max_iter=100,
                 min_samples_leaf=20, l2_regularization=0.0684044923060367)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="results/corpus_big.db")
    ap.add_argument("--episodes", type=int, default=40, help="base episodes per family")
    ap.add_argument("--replicates", type=int, default=24)
    ap.add_argument("--families", type=int, default=4, help="train families to sample")
    args = ap.parse_args()

    print("=" * 84)
    print("GOVERNOR — Stage 4A: does the model rank ACTIONS or only STATES?")
    print("=" * 84)

    ck = load(args.db)
    train_ck = [c for c in ck if c.split == "train"]
    print(f"\n[1] Fitting the Stage 3 winner on {len({c.episode_id for c in train_ck})} episodes")
    model = fit_model(train_ck, kind="gbm", uses_actions=True,
                      data_version="corpus-big", n_calib_folds=3,
                      estimator_kwargs=TUNED_GBM)
    print(f"    {model.name} (tuned GBM) — the model whose AUC 0.84 was over-read")

    profile = fit_profile_over_families(train_families(), 40)
    channels, _ = fit_channels(SynthConfig(), 120)
    envelope = Envelope(tokens=60_000, cost=0.30, wall_s=200.0, tool_calls=25)

    fams = [("train", f) for f in train_families()[: args.families]]
    fams += [("heldout", f) for f in heldout_families()]

    print(f"\n[2] Branching: {args.episodes} base episodes/family, forks at 30% and 70%,")
    print(f"    every admissible action tried, {args.replicates} replicates each.")
    print(f"    Replicate count sets the resolution: a Bernoulli rate near 0.5 from")
    print(f"    {args.replicates} draws has standard error ~{0.5/np.sqrt(args.replicates):.3f}.")

    all_points = []
    for split, fam in fams:
        pts = collect_branch_points(
            family=fam, split=split, seeds=range(90_000, 90_000 + args.episodes),
            profile=profile, channels=channels, envelope=envelope,
            n_replicates=args.replicates,
        )
        for bp in pts:
            for a in bp.realised:
                from governor.envs.synthbug import Action, Mode, Tier
                mode_s, rest = a.split("@")
                tier_s = rest.split("->")[0]
                tgt = int(rest.split("->h")[1]) if "->h" in rest else None
                act = Action(Mode(mode_s), Tier(tier_s), tgt)
                bp.predicted[a] = float(model.predict([bp.checkpoint_for(act)])[0])
        all_points += pts
        print(f"    {split:<8} {fam.name:<18} {len(pts):>4} branch points")

    # ---- is the choice even decision-relevant? ------------------------------
    print("\n[3] Sanity: does the action choice matter at these states?")
    sp = oracle_spread(all_points)
    print(f"    mean best-worst spread   {sp.get('mean_spread', 0):.3f}")
    print(f"    median spread            {sp.get('median_spread', 0):.3f}")
    print(f"    states where it matters  {sp.get('frac_states_with_real_choice', 0):.1%} "
          f"(spread >= 0.10)")
    if sp.get("mean_spread", 0) < 0.05:
        print("    WARNING: actions barely differ here. Ranking accuracy would be")
        print("             unmeasurable and any number below is noise.")

    # ---- the headline -------------------------------------------------------
    se = 0.707 / np.sqrt(args.replicates)   # SE of a difference of two Bernoulli rates
    print("\n[4] Within-state action ranking — the number that actually matters")
    print(f"    Realised values are Bernoulli rates from {args.replicates} draws, so the")
    print(f"    difference between two actions carries SE ~{se:.3f}. Noise in the realised")
    print(f"    value attenuates pairwise accuracy TOWARD 0.5, so these are conservative:")
    print(f"    a reading above 0.5 understates the true ranking ability.")
    print(f"    Reporting at several min-gap thresholds; the largest exceeds the noise.")
    summary = {}
    for gap in (0.0, 0.10, round(2 * se, 2)):
        print(f"\n    min realised gap > {gap:.2f}")
        print(f"    {'split':<10} {'points':>7} {'pairs':>7} {'pairwise acc':>13} "
              f"{'kendall tau':>12} {'mean regret':>12} {'frac optimal':>13}")
        print("    " + "-" * 80)
        for split in ("train", "heldout", "ALL"):
            pts = all_points if split == "ALL" else [p for p in all_points if p.split == split]
            if not pts:
                continue
            pr = pairwise_ranking(pts, min_gap=gap)
            kt = kendall_tau(pts)
            rg = decision_regret(pts)
            if gap == 0.10:
                summary[split] = {"pairwise": pr, "tau": kt, "regret": rg,
                                  "n_points": len(pts)}
            print(f"    {split:<10} {len(pts):>7} {pr['pairs']:>7} {pr['accuracy']:>13.3f} "
                  f"{kt:>12.3f} {rg.get('mean', 0):>12.3f} {rg.get('frac_optimal', 0):>13.1%}")

    # ---- the contrast that makes the point ----------------------------------
    print("\n[5] The contrast: pooled AUC vs within-state ranking")
    ys, ps = [], []
    for bp in all_points:
        for a, v in bp.realised.items():
            if a in bp.predicted:
                ys.append(1 if v >= 0.5 else 0)
                ps.append(bp.predicted[a])
    pooled = auc(ys, ps)
    within = summary.get("ALL", {}).get("pairwise", {}).get("accuracy", 0.5)
    print(f"    pooled AUC over all (state, action) rows   {pooled:.3f}")
    print(f"    within-state pairwise ranking accuracy     {within:.3f}")
    print(f"    gap                                        {pooled - within:+.3f}")
    print("    Pooled AUC can be high while within-state accuracy sits at 0.5 —")
    print("    that is precisely the failure mode this experiment exists to detect.")

    # ---- verdict ------------------------------------------------------------
    held = summary.get("heldout", {}).get("pairwise", {}).get("accuracy", 0.5)
    print("\n[6] VERDICT")
    if held >= 0.60:
        print(f"    Within-state action ranking on UNSEEN regimes: {held:.3f}")
        print("    The model ranks actions meaningfully better than chance. Ranking-based")
        print("    action selection is supported by direct counterfactual evidence, not")
        print("    inferred from pooled AUC.")
    elif held >= 0.55:
        print(f"    Within-state action ranking on UNSEEN regimes: {held:.3f}")
        print("    Weak but positive. Usable only with wide margins; treat near-ties as")
        print("    indistinguishable and break them on cost.")
    else:
        print(f"    Within-state action ranking on UNSEEN regimes: {held:.3f}")
        print("    The model does NOT rank actions. AUC 0.84 was measuring easy-vs-hard")
        print("    states. The representation is insufficient for the decision problem,")
        print("    and no amount of recalibration fixes that — the missing piece is an")
        print("    action-effect model, not a better probability mapping.")

    Path("results").mkdir(exist_ok=True)
    Path("results/branch_ranking.json").write_text(json.dumps(
        {"summary": summary, "pooled_auc": pooled, "spread": sp}, indent=2, default=float))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
