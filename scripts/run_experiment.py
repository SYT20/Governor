#!/usr/bin/env python3
"""End-to-end Stage 0/1 experiment on SynthBug.

    python3 scripts/run_experiment.py --episodes 200 --warmup 150

What this does and does not show
--------------------------------
It validates the *machinery*: accounting, hard enforcement, admissibility, the
corrected Bayes update, decision records, and the arm-comparison harness. It says
nothing about real code-fixing skill -- SynthBug is a simulator with known ground
truth, and every number below inherits that caveat.

The degradation sweep is the headline experiment of the Decision Record (section B,
H1): the effect of a better policy is predicted to appear under *scarcity*, not at
full budget where every reasonable policy converges.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.accounting.meter import Envelope  # noqa: E402
from governor.arms.baselines import (  # noqa: E402
    EpsilonGreedyCollector,
    FixedArm,
    HeuristicArm,
    OracleArm,
    StaticRoutedArm,
)
from governor.cognitive.belief import Channel, ChannelBook, conservative_channel  # noqa: E402
from governor.envs.synthbug import Action, Mode, SynthConfig, Tier, make_task  # noqa: E402
from governor.models.cost import CostProfile  # noqa: E402
from governor.policy.runner import config_hash, run_episode  # noqa: E402
from governor.record.store import open_store  # noqa: E402


def fit_cost_profile(cfg: SynthConfig, n: int) -> CostProfile:
    """Stage 2 in miniature: learn what actions cost by observing them."""
    profile = CostProfile(min_observations=8)
    for seed in range(10_000, 10_000 + n):
        task = make_task(seed, cfg)
        for tier in Tier:
            for mode in (Mode.EXPLORE, Mode.EXPLOIT, Mode.VERIFY):
                a = Action(mode, tier, 0 if mode is not Mode.VERIFY else None)
                profile.observe(a.action_class, task.cost_of(a))
        for m in (Mode.STOP_VERIFIED, Mode.STOP_UNVERIFIED, Mode.STOP_FAILURE):
            a = Action(m, Tier.T0)
            profile.observe(a.action_class, task.cost_of(a))
    profile.freeze(data_version=f"warmup-{n}")
    return profile


def fit_channels(cfg: SynthConfig, n: int) -> tuple[dict[Tier, Channel], ChannelBook]:
    """Estimate the two-parameter channel reliability from resolved episodes.

    Section G.3: alpha and beta are estimated independently. Labelling requires
    knowing which hypothesis was actually true, which in the real system is only
    available for resolved episodes -- here the simulator provides it directly, and
    that difference is exactly why D5 ships a conservative fallback first.
    """
    book = ChannelBook()
    for seed in range(20_000, 20_000 + n):
        task = make_task(seed, cfg)
        for tier in Tier:
            for h in range(cfg.n_hypotheses):
                obs = task.step(Action(Mode.EXPLORE, tier, h))
                book.observe(
                    f"explore@{tier}",
                    target_was_true=(h == task.true_cause),
                    observation=obs.value,
                )
    channels = {}
    for tier in Tier:
        est = book.estimator(f"explore@{tier}")
        channels[tier] = (
            est.to_channel() if est.is_usable() else conservative_channel(f"explore@{tier}")
        )
    return channels, book


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200, help="eval episodes per arm per budget")
    p.add_argument("--warmup", type=int, default=150, help="episodes for cost/channel fitting")
    p.add_argument("--hypotheses", type=int, default=4)
    p.add_argument("--db", default="results/governor.db")
    args = p.parse_args()

    cfg = SynthConfig(n_hypotheses=args.hypotheses)

    print("=" * 78)
    print("GOVERNOR — Stage 0/1 end-to-end on SynthBug (machinery validation only)")
    print("=" * 78)

    # ---- offline fitting, frozen before any evaluation -----------------------
    profile = fit_cost_profile(cfg, args.warmup)
    channels, book = fit_channels(cfg, args.warmup)

    print("\n[1] Cost profile (fitted, p50 / p90 tokens)")
    for row in profile.report():
        if row["action_class"].startswith("STOP"):
            continue
        print(f"    {row['action_class']:<14} n={row['n']:<5} "
              f"p50={row['tokens_p50']:>9.0f}  p90={row['tokens_p90']:>9.0f}")

    print("\n[2] Evidence channels — fitted vs. the values that generated the data")
    print(f"    {'channel':<14} {'alpha_fit':>9} {'alpha_true':>10} "
          f"{'beta_fit':>9} {'beta_true':>10} {'LR+':>6}")
    max_err = 0.0
    for r in book.report():
        tier = Tier(r["channel"].split("@")[1])
        ta, tb = cfg.alpha[tier], cfg.beta[tier]
        max_err = max(max_err, abs(r["alpha"] - ta), abs(r["beta"] - tb))
        print(f"    {r['channel']:<14} {r['alpha']:>9.3f} {ta:>10.3f} "
              f"{r['beta']:>9.3f} {tb:>10.3f} {r['lr_plus']:>6.2f}")
    print(f"    max |fitted - true| = {max_err:.4f}   "
          f"{'PASS' if max_err < 0.05 else 'FAIL — channel estimator is biased'}")

    # ---- full-budget reference ----------------------------------------------
    ref = Envelope(tokens=60_000, cost=0.30, wall_s=200.0, tool_calls=25)
    cfg_h = config_hash(cfg, ref, {t: (c.alpha, c.beta) for t, c in channels.items()})
    print(f"\n[3] Envelope (100%): {ref.as_dict()}")
    print(f"    config_hash = {cfg_h}   (arm-parity assertion, drift check #6)")

    # ---- degradation sweep ---------------------------------------------------
    scales = [1.0, 0.75, 0.5, 0.25, 0.10]
    arms = [FixedArm(), StaticRoutedArm(), HeuristicArm(), OracleArm(),
            EpsilonGreedyCollector(epsilon=0.30, seed=1)]

    Path("results").mkdir(exist_ok=True)
    with open_store(args.db) as store:
        for scale in scales:
            env = ref.scaled(scale)
            for arm in arms:
                for seed in range(args.episodes):
                    task = make_task(seed, cfg)
                    run_episode(
                        task=task, arm=arm, envelope=env,
                        profile=profile, channel_for=channels, store=store,
                        episode_id=f"{arm.name}|{scale}|{seed}",
                        budget_scale=scale, cfg_hash=cfg_h,
                    )
            store.commit()

        rows = store.arm_summary()
        n_eps = store.episode_count()

    # ---- report --------------------------------------------------------------
    print(f"\n[4] Degradation sweep — {n_eps} episodes total\n")
    hdr = f"    {'budget':>7} {'arm':<14} {'TSR':>7} {'cost/win':>9} {'BVR':>5} {'trunc':>6} {'steps':>6}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    by_scale: dict[float, dict[str, float]] = {}
    bvr_total = 0.0
    for r in rows:
        cps = "  inf" if r["cost_per_success"] == float("inf") else f"{r['cost_per_success']:.4f}"
        print(f"    {r['budget']:>6.0%} {r['arm']:<14} {r['tsr']:>7.1%} {cps:>9} "
              f"{r['bvr']:>5.0%} {r['truncation_rate']:>6.2f} {r['avg_decisions']:>6.1f}")
        by_scale.setdefault(r["budget"], {})[r["arm"]] = r["tsr"]
        bvr_total += r["bvr"]

    print("\n[5] Gates")
    print(f"    BVR = 0 across every arm and budget ........ "
          f"{'PASS' if bvr_total == 0 else 'FAIL'}")

    # Headroom = what the best non-cheating baseline leaves on the table relative
    # to the achievable ceiling. This, not heuristic-minus-fixed, is the quantity
    # H1 predicts should grow as the envelope tightens: under scarcity, decision
    # quality is what separates policies.
    baselines = ("A_fixed", "B_static", "C_heuristic")
    headroom: dict[float, float] = {}
    for s, v in by_scale.items():
        best = max((v.get(b, 0.0) for b in baselines), default=0.0)
        if v.get("F_oracle", 0.0) > 0.0:  # skip degenerate levels
            headroom[s] = v["F_oracle"] - best

    live = {s: h for s, h in headroom.items() if s >= 0.25}
    widen = live.get(0.25, 0) > live.get(1.0, 0)
    print(f"    Headroom widens as budget tightens .......... "
          f"{'PASS' if widen else 'INCONCLUSIVE'}"
          f"   (100%: {live.get(1.0, 0):+.1%} -> 25%: {live.get(0.25, 0):+.1%})")
    if live:
        print(f"    Mean headroom over best baseline ............ "
              f"{statistics.mean(live.values()):+.1%}"
              f"   (the gap a learned policy has to close)")

    dead = [s for s, v in by_scale.items() if v.get("F_oracle", 0.0) == 0.0]
    if dead:
        print(f"    Degenerate budget levels (oracle also 0%) ... "
              f"{sorted(dead)}  — below the floor where any policy can act;\n"
              f"        exclude from claims, and bound the useful range above them.")

    print(f"\n    Records: {args.db}  —  query with sqlite3, or DuckDB later.")
    print("    Reminder: SynthBug validates machinery, not code-fixing skill.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
