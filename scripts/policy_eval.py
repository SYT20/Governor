#!/usr/bin/env python3
"""Stage 5: the question the whole project exists to answer.

Not "can we build a good value model" but:

    does an explicit policy layer make an agent better under the SAME budget?

Held constant across every arm: environment, task seeds, cost profile, evidence
channels, envelope, candidate generator, accountant. The only thing that varies is
who chooses the action.

Evaluated on HELD-OUT families only -- regimes no model was fitted on -- and swept
across budget levels, because H1 predicts the effect appears under scarcity and
vanishes at full budget where every reasonable policy converges.
"""
from __future__ import annotations
import json, sys, math, random
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from governor.accounting.meter import Envelope
from governor.arms.baselines import FixedArm, HeuristicArm, OracleArm, StaticRoutedArm
from governor.arms.governor import GovernorArm
from governor.envs.families import heldout_families, train_families
from governor.envs.synthbug import SynthConfig
from governor.models.value import fit_model
from governor.policy.runner import config_hash, run_episode
from scripts.build_corpus import fit_profile_over_families
from scripts.fit_value_model import load
from scripts.run_experiment import fit_channels

TUNED = dict(max_depth=5, learning_rate=0.2027, max_iter=100,
             min_samples_leaf=20, l2_regularization=0.0684)

def mcnemar(a: list[int], b: list[int]) -> tuple[int, int, float]:
    """Exact-ish McNemar on paired binary outcomes (same tasks, same seeds)."""
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return n01, n10, min(1.0, p)

def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--db", default="results/corpus_af.db")
    args = ap.parse_args()

    print("=" * 88)
    print("GOVERNOR — Stage 5: does the policy layer beat the baselines under scarcity?")
    print("=" * 88)

    ck = load(args.db); tr = [c for c in ck if c.split == "train"]
    ch, _ = fit_channels(SynthConfig(), 120); chs = {str(k): v for k, v in ch.items()}
    model = fit_model(tr, kind="gbm", uses_actions=True, data_version="af",
                      n_calib_folds=3, estimator_kwargs=TUNED, channels=chs)
    profile = fit_profile_over_families(train_families(), 40)
    ref = Envelope(tokens=60_000, cost=0.30, wall_s=200.0, tool_calls=25)
    cfg_h = config_hash(ref, "stage5")

    print(f"\n  model fitted on {len({c.episode_id for c in tr})} train episodes")
    print(f"  evaluated on HELD-OUT families: "
          f"{', '.join(f.name for f in heldout_families())}")
    print(f"  {args.episodes} tasks per family, identical seeds across all arms")
    print(f"  config_hash {cfg_h} (arm parity)")

    arms = {
        "A_fixed": FixedArm(), "B_static": StaticRoutedArm(), "C_heuristic": HeuristicArm(),
        "E_governor": GovernorArm(model=model, channels=chs),
        "F_oracle": OracleArm(),
    }
    scales = [1.0, 0.5, 0.35, 0.25]
    res: dict = {}

    for scale in scales:
        env = ref.scaled(scale)
        for name, arm in arms.items():
            wins, cost, trunc, viol = [], 0.0, 0, 0
            for fam in heldout_families():
                for seed in range(70_000, 70_000 + args.episodes):
                    r = run_episode(task=fam.task(seed), arm=arm, envelope=env,
                                    profile=profile, channel_for=ch, store=None,
                                    budget_scale=scale, cfg_hash=cfg_h)
                    wins.append(int(r.succeeded)); cost += r.consumed["cost"]
                    trunc += r.truncations; viol += int(r.violated)
            res.setdefault(scale, {})[name] = {
                "wins": wins, "tsr": float(np.mean(wins)), "cost": cost,
                "cost_per_win": cost / max(sum(wins), 1), "trunc": trunc, "viol": viol}

    print(f"\n  {'budget':>7} {'arm':<13} {'TSR':>7} {'cost/win':>9} {'BVR':>5} "
          f"{'vs C':>8} {'McNemar p':>10}")
    print("  " + "-" * 70)
    for scale in scales:
        base = res[scale]["C_heuristic"]["wins"]
        for name in arms:
            d = res[scale][name]
            n01, n10, p = mcnemar(base, d["wins"])
            delta = d["tsr"] - res[scale]["C_heuristic"]["tsr"]
            ds = "" if name == "C_heuristic" else f"{delta:+.1%}"
            ps = "" if name == "C_heuristic" else f"{p:.4f}"
            print(f"  {scale:>6.0%} {name:<13} {d['tsr']:>7.1%} {d['cost_per_win']:>9.4f} "
                  f"{d['viol']/(args.episodes*3):>5.0%} {ds:>8} {ps:>10}")
        print()

    print("  " + "=" * 70)
    print("  H1: the policy layer's advantage should GROW as the budget tightens.")
    for name in ("A_fixed", "B_static", "E_governor"):
        gaps = {s: res[s][name]["tsr"] - res[s]["C_heuristic"]["tsr"] for s in scales}
        trend = gaps[0.25] - gaps[1.0]
        print(f"    {name:<13} vs C:  100%={gaps[1.0]:+.1%}  25%={gaps[0.25]:+.1%}  "
              f"trend {trend:+.1%}")
    g = res[0.25]["E_governor"]["tsr"]; c = res[0.25]["C_heuristic"]["tsr"]
    o = res[0.25]["F_oracle"]["tsr"]
    _, _, p25 = mcnemar(res[0.25]["C_heuristic"]["wins"], res[0.25]["E_governor"]["wins"])
    print(f"\n  At 25% budget: Governor {g:.1%}, heuristic {c:.1%}, oracle ceiling {o:.1%}")
    print(f"  Governor closes {((g-c)/max(o-c,1e-9)):.0%} of the heuristic-to-oracle gap "
          f"(McNemar p={p25:.4f})")
    if p25 < 0.05 and g > c:
        print("  VERDICT: H1 SUPPORTED at 25% budget.")
    elif g > c:
        print("  VERDICT: Governor ahead but not significant. Underpowered or a small effect.")
    else:
        print("  VERDICT: H1 NOT SUPPORTED. The heuristic is as good or better.")
    json.dump({str(s): {k: {kk: vv for kk, vv in v.items() if kk != "wins"}
                        for k, v in res[s].items()} for s in scales},
              open("results/stage5_policy.json", "w"), indent=2)
main()
