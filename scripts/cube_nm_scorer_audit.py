#!/usr/bin/env python3
"""Hardened audit of the exact myopic scorer, across posterior regimes.

WHY MORE THAN THE 12-STATE CHECK. The exact scorer is about to become the
teacher: every future claim of the form "Governor approaches / beats optimal
myopic" is measured against it. A component in that position deserves more
testing than the policy it judges, and the original check probed 12 states
reached by two random acquisitions -- one narrow regime.

WHAT IS DIFFERENT HERE.

1. REGIME STRATIFICATION. States are constructed to span the posterior shapes
   the scorer will actually meet: empty, context-resolved, evidence from the
   wrong block only, partial true-block evidence, and deliberately conflicting
   evidence assembled from two blocks at once.

2. A CALIBRATED THRESHOLD, NOT A GUESSED ONE. The previous gate asserted
   |exact - sampled| < 0.02. That number was invented, and inventing thresholds
   is what made the first likelihood gate "fail" at its own Bayes floor. The
   sampled estimate is a mean over n_mc draws, so it carries a standard error
   that is computable from the same draws. The test is therefore whether the
   discrepancy is within 4 SE of the sampled value -- if the exact scorer is
   right, the gap IS sampling error and must scale like one.

3. EVERY CANDIDATE, NOT A SLICE. All available groups are scored at each state,
   so the argmax comparison is over the real decision, not a subset of it.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from governor.envs.cube_nm_bayes import CubeNMBayes  # noqa: E402
from governor.envs.cube_nm_repro import BLOCK_SIZE, N_LABELS, CubeNMRepro  # noqa: E402

N_MC = 20000
N_PER_REGIME = 25


def build_states(ds, bayes, rng):
    """Posterior states spanning the regimes the scorer must handle."""
    out = []

    def state(i, groups, tag):
        logL = np.zeros(bayes.H)
        avail = list(range(bayes.n_groups))
        for g in groups:
            if g in avail:
                avail.remove(g)
                logL = logL + bayes.loglik_cols(ds.features[i], bayes.group_cols[g])
        return (tag, logL, avail)

    for i in range(N_PER_REGIME):
        c = int(ds.context[i])
        y = int(ds.labels[i])
        true_blk = ds.block_group_ids(c)
        wrong_blk = ds.block_group_ids((c + 2) % ds.n_contexts)
        informative = [true_blk[(y + j) % BLOCK_SIZE] for j in range(3)]

        out.append(state(i, [], "empty"))
        out.append(state(i, [0] + true_blk[:2], "context_resolved"))
        out.append(state(i, wrong_blk[:4], "wrong_block_only"))
        out.append(state(i, informative[:2], "true_block_partial"))
        # conflicting: real evidence from the true block, plus four draws from a
        # different block that the posterior must weigh against it
        out.append(state(i, informative[:2] + wrong_blk[:4], "conflicting"))
        # deep: seven acquisitions, the regime late-budget decisions live in
        out.append(state(i, rng.choice(np.arange(1, bayes.n_groups), 7,
                                       replace=False).tolist(), "deep_random"))
    return out


def sampled_scores(bayes, logL, groups, rng, n_mc):
    """MC score per candidate, WITH the standard error of each estimate."""
    post = np.exp(logL - logL.max())
    post /= post.sum()
    hs = rng.choice(bayes.H, size=n_mc, p=post)
    mean, se = {}, {}
    for g in groups:
        cols = bayes.group_cols[g]
        mu_s, sd_s = bayes.MU[np.ix_(hs, cols)], bayes.SD[np.ix_(hs, cols)]
        x = mu_s + sd_s * rng.standard_normal(mu_s.shape)
        d = (x[:, None, :] - bayes.MU[None, :, cols]) / bayes.SD[None, :, cols]
        ll = (-0.5 * d * d - bayes.LOGSD[None, :, cols]).sum(axis=-1)
        ent = bayes._entropy_y(logL[None, :] + ll)
        mean[g] = -float(ent.mean())
        se[g] = float(ent.std(ddof=1) / np.sqrt(n_mc))
    return mean, se


def exact_scores(bayes, logL, groups):
    """Production scorer's value for every candidate (not just its argmax)."""
    post = np.exp(logL - logL.max())
    post /= post.sum()
    ex = {}
    single = [g for g in groups if g != 0]
    if single:
        slots = np.array([bayes._col_slot[bayes.group_cols[g][0]] for g in single])
        joint = post[:, None, None] * bayes._pdf[:, slots, :]
        px = joint.sum(axis=0)
        w = px / px.sum(axis=1, keepdims=True)
        cond = joint / np.maximum(px[None, :, :], 1e-300)
        py = cond.reshape(bayes.K, N_LABELS, len(single), -1).sum(axis=0)
        ent = -(py * np.log(np.maximum(py, 1e-300))).sum(axis=0)
        for j, g in enumerate(single):
            ex[g] = -float((w[j] * ent[j]).sum())
    if 0 in groups:
        P = post.reshape(bayes.K, N_LABELS)
        pc = P.sum(axis=1)
        pyc = P / np.maximum(pc[:, None], 1e-300)
        hc = -(pyc * np.log(np.maximum(pyc, 1e-300))).sum(axis=1)
        ex[0] = -float((pc * hc).sum())
    return ex


def main() -> int:
    print("=" * 84)
    print("EXACT SCORER AUDIT — regime-stratified, calibrated against MC standard error")
    print("=" * 84)

    ds = CubeNMRepro(n_samples=2000, seed=7)
    bayes = CubeNMBayes(ds)
    rng = np.random.default_rng(0)
    states = build_states(ds, bayes, rng)

    per = defaultdict(lambda: {"n": 0, "argmax_ok": 0, "worst_z": 0.0,
                               "worst_gap": 0.0, "cands": 0, "over4se": 0})
    print(f"\n  {len(states)} states x up to {bayes.n_groups} candidates, "
          f"n_mc={N_MC} per candidate")
    print("  (this scores EVERY available candidate, so argmax agreement is the "
          "real decision)\n")

    for tag, logL, avail in states:
        ex = exact_scores(bayes, logL, avail)
        mc, se = sampled_scores(bayes, logL, avail, rng, N_MC)
        d = per[tag]
        d["n"] += 1
        d["cands"] += len(avail)
        for g in avail:
            gap = abs(ex[g] - mc[g])
            z = gap / max(se[g], 1e-12)
            d["worst_gap"] = max(d["worst_gap"], gap)
            d["worst_z"] = max(d["worst_z"], z)
            d["over4se"] += int(z > 4.0)
        # argmax agreement, tolerant of a genuine near-tie inside MC noise
        ga = max(ex, key=ex.get)
        gb = max(mc, key=mc.get)
        d["argmax_ok"] += int(ga == gb or abs(ex[ga] - ex[gb]) < 4 * se[gb])

    print(f"  {'regime':<20} {'states':>7} {'cands':>7} {'argmax':>8} "
          f"{'worst gap':>10} {'worst z':>9} {'>4 SE':>7}")
    print("  " + "-" * 74)
    ok = True
    for tag in ("empty", "context_resolved", "wrong_block_only",
                "true_block_partial", "conflicting", "deep_random"):
        d = per[tag]
        frac = d["over4se"] / max(d["cands"], 1)
        good = d["argmax_ok"] == d["n"] and frac < 0.01
        ok &= good
        print(f"  {tag:<20} {d['n']:>7} {d['cands']:>7} "
              f"{d['argmax_ok']}/{d['n']:<6} {d['worst_gap']:>10.5f} "
              f"{d['worst_z']:>9.2f} {frac:>6.2%}{'' if good else '  <-- FAIL'}")

    tot_c = sum(d["cands"] for d in per.values())
    tot_o = sum(d["over4se"] for d in per.values())
    print(f"\n  {tot_c} candidate scores compared; {tot_o} ({tot_o/tot_c:.3%}) "
          f"exceeded 4 SE of the sampled estimate")
    print(f"  (at 4 SE, pure sampling noise alone predicts ~0.006% exceedances)")
    print(f"\n  SCORER AUDIT: {'PASS' if ok else 'FAIL'}")

    Path("results").mkdir(exist_ok=True)
    Path("results/cube_nm_scorer_audit.json").write_text(json.dumps(
        {k: dict(v) for k, v in per.items()}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
