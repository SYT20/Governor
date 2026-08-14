#!/usr/bin/env python3
"""Stage 7: the policy-improvement estimand, Delta(s,a;pi_0).

Every previous attempt estimated the wrong thing.

  Stage 3     P(success | s, a) from observational trajectories. Dominated by
              state difficulty; within-state action ranking 0.61.
  Rule bank   marginal associations of action vs a heterogeneous pooled control,
              composed additively. Catastrophic: 15.6% vs the heuristic's 82.2%.

The right quantity for improving on a known baseline is the INTERVENTIONAL
advantage relative to that baseline:

    Delta(s, a; pi_0) = E[Y | do(A=a), s, then pi_0]  -  E[Y | do(A=pi_0(s)), s, then pi_0]

The branch harness already produces exactly this and it was being used only as an
evaluation metric: fork the state, force action a, continue under a FIXED
HeuristicArm. Subtracting the baseline arm's realised value turns those rollouts
into training targets for a policy-improvement model.

The resulting policy is a SAFE OVERRIDE: follow the baseline everywhere, and
deviate only where the predicted advantage is positive with confidence and the
resource cost is justified. That is the project's original question stated
precisely -- when is it worth deviating from the current strategy?
"""
from __future__ import annotations
import copy, json, sys
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from governor.accounting.meter import Accountant, Envelope
from governor.arms.baselines import HeuristicArm
from governor.envs.families import heldout_families, train_families
from governor.envs.synthbug import Action, Mode, SynthConfig, Tier
from governor.experiments.branch import _rollout, _fresh_context
from governor.models.cost import Reserve
from governor.policy.runner import (_apply_observation, _charge_or_truncate,
                                    deterministic_candidates)
from scripts.build_corpus import fit_profile_over_families
from scripts.run_experiment import fit_channels

CACHE = Path("results/delta_branches.json")

def collect(fams, split, seeds, profile, channels, envelope, reps=14, tiers=tuple(Tier)):
    """Branch, recording the BASELINE action so Delta is computable."""
    out = []
    for fam_label, fam in fams:
        n0 = len(out)
        for seed in seeds:
            task = fam.task(seed); acc = Accountant(envelope=envelope)
            ctx = _fresh_context(task, acc, profile)
            arm = HeuristicArm(); arm.reset(task); reserve = Reserve(profile)
            snaps = []
            while not task.terminated and ctx.step < 40:
                cands = deterministic_candidates(ctx, tiers)
                need = reserve.required(verified=ctx.verified)
                adm = [a for a in cands if acc.admissible(profile.vector(a.action_class),
                       reserve={} if a.mode.is_terminal else need)]
                if not adm: break
                chosen, _ = arm.act(ctx, adm)
                nt = [a for a in adm if not a.mode.is_terminal]
                if len(nt) >= 2 and not chosen.mode.is_terminal:
                    snaps.append((ctx.step, copy.deepcopy(task), copy.deepcopy(ctx),
                                  list(nt), chosen))
                realised = task.cost_of(chosen)
                _, tr = _charge_or_truncate(acc, chosen.action_class, realised)
                obs = task.step(chosen); _apply_observation(ctx, chosen, obs, channels, tr)
                ctx.step += 1
            if not snaps: continue
            for frac in (0.3, 0.7):
                i = min(len(snaps)-1, int(frac*len(snaps)))
                step_i, st, sc, cands, baseline = snaps[i]
                cands = cands[:4]
                if baseline not in cands: cands = [baseline] + cands[:3]
                rec = {"family": fam.name, "split": split, "seed": seed,
                       "decision_id": step_i, "features": dict(sc.features()),
                       "belief": list(sc.belief), "baseline": str(baseline),
                       "admissible": [str(a) for a in cands], "realised": {}}
                for cand in cands:
                    wins = 0
                    for r in range(reps):
                        t2 = copy.deepcopy(st); c2 = copy.deepcopy(sc)
                        c2.accountant = copy.deepcopy(sc.accountant)
                        t2._rng.seed(seed*100_003 + step_i*997 + r*31)   # CRN
                        rl = t2.cost_of(cand)
                        _, tr = _charge_or_truncate(c2.accountant, cand.action_class, rl)
                        ob = t2.step(cand); _apply_observation(c2, cand, ob, channels, tr)
                        c2.step += 1
                        wins += int(_rollout(t2, c2, profile, channels, tiers, 40))
                    rec["realised"][str(cand)] = wins/reps
                out.append(rec)
        print(f"    {fam_label:<8} {fam.name:<18} +{len(out)-n0:>3} branch points "
              f"(total {len(out)})", flush=True)
    return out

def main() -> int:
    ch,_ = fit_channels(SynthConfig(), 120)
    prof = fit_profile_over_families(train_families(), 40)
    env = Envelope(tokens=60_000, cost=0.30, wall_s=200.0, tool_calls=25)
    if CACHE.exists():
        pts = json.loads(CACHE.read_text()); print(f"  loaded {len(pts)} cached branch points")
    else:
        fams = [("train", f) for f in train_families()[:5]] + \
               [("heldout", f) for f in heldout_families()]
        pts = collect(fams, "x", range(90_000, 90_018), prof, ch, env, reps=10)
        for p in pts: p["split"] = "heldout" if p["family"] in \
            {f.name for f in heldout_families()} else "train"
        CACHE.write_text(json.dumps(pts, indent=1)); print(f"  collected {len(pts)} branch points")

    # ---- Q1 admissibility, Q2 pairwise interventional effects ----------------
    from collections import Counter, defaultdict
    print(f"\n  [Q1] admissible-set sizes: "
          f"{dict(sorted(Counter(len(p['admissible']) for p in pts).items()))}")
    print(f"  [Q2] PAIRWISE interventional effects (same state, both actions forced,")
    print(f"       common random numbers). NOT action-vs-pooled-rest.")
    pair = defaultdict(list)
    for p in pts:
        ks = list(p["realised"])
        for i in range(len(ks)):
            for j in range(len(ks)):
                if i != j:
                    pair[(ks[i].split("->")[0], ks[j].split("->")[0])].append(
                        p["realised"][ks[i]] - p["realised"][ks[j]])
    rows = [(a,b,np.mean(v),len(v)) for (a,b),v in pair.items() if len(v) >= 60]
    rows.sort(key=lambda r: -r[2])
    print(f"       {'action A':<14} {'vs action B':<14} {'mean Delta':>11} {'n':>6}")
    for a,b,m,n in rows[:6]:
        print(f"       {a:<14} {b:<14} {m:>+11.3f} {n:>6}")

    # ---- Delta relative to the BASELINE action ------------------------------
    D = []
    for p in pts:
        b = p["baseline"]
        if b not in p["realised"]: continue
        base = p["realised"][b]
        for a, v in p["realised"].items():
            if a == b: continue
            D.append({"split": p["split"], "family": p["family"], "action": a,
                      "features": p["features"], "belief": p["belief"],
                      "delta": v - base})
    tr = [d for d in D if d["split"]=="train"]; hd = [d for d in D if d["split"]=="heldout"]
    print(f"\n  Delta(s,a;pi_0) samples: {len(tr)} train / {len(hd)} held-out")
    print(f"  mean Delta {np.mean([d['delta'] for d in D]):+.4f}  "
          f"frac positive {np.mean([d['delta']>0 for d in D]):.1%}")
    print(f"  -> most deviations from the baseline HURT, which is why a policy that")
    print(f"     overrides everywhere loses. The question is whether the positive")
    print(f"     minority is PREDICTABLE.")

    # ---- can Delta be predicted? --------------------------------------------
    from sklearn.ensemble import HistGradientBoostingRegressor
    FE = sorted(tr[0]["features"])
    def X(rows):
        acts = sorted({d["action"].split("->")[0] for d in D})
        M = []
        for d in rows:
            v = [d["features"][k] for k in FE]
            v += [1.0 if d["action"].split("->")[0]==a else 0.0 for a in acts]
            b = d["belief"]; v += [max(b) if b else 0.0, len(b)]
            M.append(v)
        return np.array(M)
    ytr = np.array([d["delta"] for d in tr]); yhd = np.array([d["delta"] for d in hd])
    m = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05,
                                      min_samples_leaf=20, random_state=0)
    m.fit(X(tr), ytr)
    ph = m.predict(X(hd))
    ss = 1 - np.sum((yhd-ph)**2)/max(np.sum((yhd-yhd.mean())**2), 1e-9)
    sign = np.mean((ph>0)==(yhd>0))
    print(f"\n  Delta model on HELD-OUT regimes:  R^2 {ss:+.3f}   sign accuracy {sign:.1%}")
    for thr in (0.0, 0.05, 0.10):
        sel = ph > thr
        if sel.sum() >= 10:
            print(f"    predicted Delta > {thr:.2f}:  n={int(sel.sum()):>4}  "
                  f"actual mean Delta {yhd[sel].mean():+.4f}  "
                  f"frac actually positive {np.mean(yhd[sel]>0):.1%}")
    print(f"\n  GATE: an override is only safe if 'predicted Delta > t' selects a set")
    print(f"        whose ACTUAL mean Delta is positive. If it is negative at every")
    print(f"        threshold, no safe override exists and the baseline is unimprovable")
    print(f"        by this estimand.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
