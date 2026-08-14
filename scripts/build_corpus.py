#!/usr/bin/env python3
"""Stage 2: collect a labelled checkpoint corpus and evaluate the J.8 gate.

    python3 scripts/build_corpus.py --per-family 90

Produces `results/corpus.db` plus a verdict. Stage 3 must not begin until the gate
is green -- fitting a value model on an insufficient corpus produces a number that
looks like a result and is not one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.accounting.meter import Envelope  # noqa: E402
from governor.corpus.build import build_corpus, write_corpus  # noqa: E402
from governor.corpus.gate import check_gate  # noqa: E402
from governor.envs.families import (  # noqa: E402
    HELDOUT,
    describe,
    heldout_families,
    train_families,
)
from governor.envs.synthbug import Action, Mode, SynthConfig, Tier, make_task  # noqa: E402
from governor.models.cost import CostProfile  # noqa: E402
from scripts.run_experiment import fit_channels  # noqa: E402


def fit_profile_over_families(fams, n: int) -> CostProfile:
    """Cost profile pooled across training regimes.

    Fitting on one family would bake that regime's cost scale into every estimate;
    the `costly` family alone is 2.4x the baseline.
    """
    profile = CostProfile(min_observations=8)
    for fam in fams:
        for seed in range(50_000, 50_000 + n):
            task = fam.task(seed)
            for tier in Tier:
                for mode in (Mode.EXPLORE, Mode.EXPLOIT, Mode.VERIFY):
                    a = Action(mode, tier, 0 if mode is not Mode.VERIFY else None)
                    profile.observe(a.action_class, task.cost_of(a))
            for m in (Mode.STOP_VERIFIED, Mode.STOP_UNVERIFIED, Mode.STOP_FAILURE):
                a = Action(m, Tier.T0)
                profile.observe(a.action_class, task.cost_of(a))
    profile.freeze(data_version=f"pooled-{len(fams)}fam-{n}")
    return profile


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--per-family", type=int, default=90)
    p.add_argument("--epsilon", type=float, default=0.30)
    p.add_argument("--warmup", type=int, default=40)
    p.add_argument("--db", default="results/corpus.db")
    args = p.parse_args()

    print("=" * 84)
    print("GOVERNOR — Stage 2: randomised corpus collection + J.8 sufficiency gate")
    print("=" * 84)

    train, held = train_families(), heldout_families()

    print(f"\n[1] Task families ({len(train)} train, {len(held)} held out)")
    print(f"    {'family':<18} {'split':<9} {'k':>2} {'a_T1':>6} {'b_T1':>6} {'pfix':>6}  note")
    for r in describe():
        print(f"    {r['family']:<18} {r['split']:<9} {r['k']:>2} {r['alpha_T1']:>6.2f} "
              f"{r['beta_T1']:>6.2f} {r['p_fix_T1']:>6.2f}  {r['note']}")
    print(f"    held-out families are never collected from: {', '.join(HELDOUT)}")

    print("\n[2] Fitting cost profile and evidence channels (pooled over train families)")
    profile = fit_profile_over_families(train, args.warmup)
    channels, book = fit_channels(SynthConfig(), args.warmup * 3)
    for r in book.report():
        print(f"    {r['channel']:<14} alpha={r['alpha']:.3f} beta={r['beta']:.3f} "
              f"LR+={r['lr_plus']:>5.2f}  usable={r['usable']}")

    envelope = Envelope(tokens=60_000, cost=0.30, wall_s=200.0, tool_calls=25)

    print(f"\n[3] Collecting under epsilon-greedy (eps={args.epsilon}), "
          f"{args.per_family} episodes/family")
    checkpoints = build_corpus(
        families=train, heldout=held,
        episodes_per_family=args.per_family,
        profile=profile, channels=channels, envelope=envelope,
        epsilon=args.epsilon,
    )
    write_corpus(checkpoints, args.db)
    print(f"    wrote {len(checkpoints)} checkpoints -> {args.db}")

    print("\n[4] J.8 sufficiency gate (train split)")
    res = check_gate(checkpoints, split="train")
    print(res.render())

    s = res.stats
    print(f"\n[5] Corpus statistics")
    print(f"    checkpoints={s['checkpoints']}  episodes={s['episodes']}  "
          f"mean/episode={s['mean_checkpoints_per_episode']}")
    print(f"    ICC(label)={s['icc_label']}  design effect={s['design_effect']}  "
          f"ESS={s['ess']}")
    print(f"    success rate={s['success_rate']:.1%}  "
          f"randomised actions={s['randomised_fraction']:.1%}")
    if s["icc_label"] >= 0.999:
        print("    NOTE: labels are constant within an episode, so ICC = 1 and the")
        print("          effective sample size equals the EPISODE count, not the")
        print("          checkpoint count. More checkpoints from the same episodes")
        print("          would not move this number. Collect more episodes instead.")
    if s["uncovered_cells"]:
        print(f"    UNCOVERED (no randomised observation): {s['uncovered_cells']}")

    # Held-out split reported for information; never gated on.
    held_res = check_gate(checkpoints, split="heldout")
    print(f"\n[6] Held-out split (for Stage 3 generalisation testing, not gated)")
    print(f"    episodes={held_res.stats.get('episodes')}  "
          f"success rate={held_res.stats.get('success_rate', 0):.1%}")

    verdict = "PASS — Stage 3 may begin" if res.passed else "FAIL — do not fit a value model yet"
    print(f"\n[7] VERDICT: {verdict}\n")

    Path("results").mkdir(exist_ok=True)
    Path("results/stage2_gate.json").write_text(
        json.dumps({"passed": res.passed, "checks": res.checks, "stats": s}, indent=2)
    )
    return 0 if res.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
