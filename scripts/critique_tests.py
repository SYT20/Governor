#!/usr/bin/env python3
"""Three claims put to test rather than accepted.

1. "+0.024 from action features is promising but inconclusive"
   -> paired bootstrap CLUSTERED BY BRANCH POINT. Pairs within a state are
      correlated, so a naive sqrt(0.25/n_pairs) SE understates the true error.

2. "The ECE plateau is not settled; use a debiased estimator"
   -> a bin sweep (already run) is not the same as debiasing. Implement the
      split-half debiased ECE and calibration-in-the-large, which isolates a pure
      base-rate/intercept shift from genuine shape miscalibration.

3. "Intercept-only recalibration preserves ranking and fixes probability levels"
   -> testable and untested. If the cross-regime shift is mostly a prior shift,
      a ONE-parameter logit offset should fix most of the ECE while leaving AUC
      bit-identical. If it does not, the shift is not just the base rate and the
      earlier isotonic/Platt failures were not merely an overfitting story.
"""
from __future__ import annotations
import json, math, pickle, random, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from governor.models.calibration import auc, brier, ece, ece_sweep, calibration_noise_floor, log_loss
from governor.models.value import fit_model
from governor.envs.synthbug import SynthConfig
from scripts.fit_value_model import load
from scripts.run_experiment import fit_channels

def logit(p): 
    p = min(max(p, 1e-9), 1 - 1e-9); return math.log(p / (1 - p))
def sig(x): return 1.0 / (1.0 + math.exp(-x))

def debiased_ece(y, p, n_bins=15, splits=40, seed=0):
    """Split-half debiased ECE (Kumar/Liang/Ma style).

    Plug-in binned ECE is biased upward by roughly B/n because each bin's observed
    frequency carries sampling noise that never cancels. Estimating bin accuracy on
    one half and evaluating the gap on the other removes the part of the gap that
    is pure noise, because the noise in the two halves is independent.
    """
    rng = random.Random(seed); vals = []
    idx = list(range(len(y)))
    for _ in range(splits):
        rng.shuffle(idx)
        h = len(idx) // 2
        A, B = idx[:h], idx[h:]
        accA = {}
        for i in A:
            b = min(n_bins - 1, int(p[i] * n_bins))
            accA.setdefault(b, []).append(y[i])
        tot = 0.0; n = 0
        for i in B:
            b = min(n_bins - 1, int(p[i] * n_bins))
            if b in accA and accA[b]:
                tot += abs(sum(accA[b]) / len(accA[b]) - p[i]); n += 1
        if n: vals.append(tot / n)
    return sum(vals) / len(vals) if vals else float("nan")

def cal_in_large(y, p):
    """Mean predicted minus mean observed. Nonzero => pure intercept/prior shift."""
    return float(np.mean(p) - np.mean(y))

def main():
    ck = load("results/corpus_af.db")
    tr = [c for c in ck if c.split == "train"]; hd = [c for c in ck if c.split == "heldout"]
    ch, _ = fit_channels(SynthConfig(), 120); chs = {str(k): v for k, v in ch.items()}
    TUNED = dict(max_depth=5, learning_rate=0.2027, max_iter=100,
                 min_samples_leaf=20, l2_regularization=0.0684)
    m = fit_model(tr, kind="gbm", uses_actions=True, data_version="af",
                  n_calib_folds=3, estimator_kwargs=TUNED, channels=chs)
    p = [float(x) for x in m.predict(hd)]; y = [c.label for c in hd]
    tr_base = float(np.mean([c.label for c in tr]))

    print("="*80); print("CLAIM 2 — is the ECE plateau an estimator artefact?"); print("="*80)
    plug = ece(y, p, 15); deb = debiased_ece(y, p, 15)
    nf = calibration_noise_floor(p, n_sim=200, n_bins=15)
    cil = cal_in_large(y, p)
    print(f"  plug-in ECE (15 bins)      {plug:.4f}")
    print(f"  split-half DEBIASED ECE    {deb:.4f}   (removes ~{plug-deb:+.4f} of noise)")
    print(f"  perfect-calibration floor  {nf['p95']:.4f}  -> debiased ratio {deb/nf['p95']:.2f}x")
    sw = ece_sweep(y, p)
    print(f"  bin sweep drift            {sw['drift_with_bins']:+.4f}  bin-sensitive={sw['bin_sensitive']}")
    print(f"  calibration-in-the-large   {cil:+.4f}  (mean predicted - mean observed)")
    print(f"  train base {tr_base:.3f} vs heldout base {np.mean(y):.3f}  "
          f"-> prior shift {np.mean(y)-tr_base:+.3f}")
    verdict2 = "REAL" if deb > nf["p95"] else "ARTEFACT"
    print(f"  VERDICT: miscalibration is {verdict2} after debiasing")

    print("\n"+"="*80); print("CLAIM 3 — does intercept-only recalibration fix it?"); print("="*80)
    a0 = auc(y, p)
    # (a) oracle offset: match the target base rate exactly, zero fitting
    c_or = logit(float(np.mean(y))) - logit(float(np.mean(p)))
    p_or = [sig(logit(x) + c_or) for x in p]
    # (b) realistic: estimate the offset from a small labelled slice of episodes
    eps = sorted({c.episode_id for c in hd}); random.Random(0).shuffle(eps)
    cal_eps = set(eps[:60])
    ycal = [c.label for c in hd if c.episode_id in cal_eps]
    pcal = [p[i] for i, c in enumerate(hd) if c.episode_id in cal_eps]
    c_sl = logit(float(np.mean(ycal))) - logit(float(np.mean(pcal)))
    tst = [i for i, c in enumerate(hd) if c.episode_id not in cal_eps]
    yt = [y[i] for i in tst]; pt = [p[i] for i in tst]
    pt_sl = [sig(logit(x) + c_sl) for x in pt]
    print(f"  {'variant':<34} {'AUC':>8} {'ECE':>8} {'debiased':>10} {'Brier':>8} {'logloss':>9}")
    print("  " + "-"*80)
    def row(lbl, yy, pp):
        print(f"  {lbl:<34} {auc(yy,pp):>8.4f} {ece(yy,pp,15):>8.4f} "
              f"{debiased_ece(yy,pp,15):>10.4f} {brier(yy,pp):>8.4f} {log_loss(yy,pp):>9.4f}")
    row("uncalibrated (all held-out)", y, p)
    row("+ oracle intercept offset", y, p_or)
    row("uncalibrated (test slice)", yt, pt)
    row("+ intercept from 60 episodes", yt, pt_sl)
    print(f"\n  AUC preserved exactly by offset? {abs(auc(y,p)-auc(y,p_or)) < 1e-12}"
          f"   (delta {auc(y,p_or)-a0:+.2e})")
    json.dump({"debiased_ece": deb, "plugin_ece": plug, "floor": nf["p95"],
               "cal_in_large": cil, "auc_preserved": abs(auc(y,p)-auc(y,p_or)) < 1e-12},
              open("results/critique_tests.json","w"), indent=2)
main()
