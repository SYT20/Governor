#!/usr/bin/env python3
"""Audit the Delta result against five objections, all on existing branch data.

1. RELIABILITY CEILING. Delta is a difference of two Bernoulli rates from 10
   replicates, so it carries Monte Carlo noise. R^2 cannot exceed the reliability
   of its own target. Var(observed) = Var(true) + Var(noise), so
   reliability = 1 - Var(noise)/Var(observed) bounds any achievable R^2.
2. SIMPLE BASELINES. Zero, per-action mean, per-regime mean, linear. If a
   two-variable rule reaches the GBM's R^2, the GBM is doing nothing.
3. ESTIMAND ISOLATION. Train on the SAME branch states/features with two targets:
   absolute realised outcome vs baseline-relative Delta. Within a state these
   induce identical orderings by construction, so any difference in achieved
   ranking is attributable to what each target lets the model LEARN.
4. PRECISION, NOT SIGN ACCURACY. With 6.9% positives, "always negative" scores
   93.1%. Sign accuracy is the wrong metric and I should not have quoted it.
5. POLICY-VALUE CURVE across thresholds, not one arbitrary 0.05.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

pts = json.loads(Path("results/delta_branches.json").read_text())
HELD = {"sharp_signal", "flaky_tests", "cheap_wide"}
for p in pts: p["split"] = "heldout" if p["family"] in HELD else "train"
REPS = 10

D = []
for p in pts:
    b = p["baseline"]
    if b not in p["realised"]: continue
    base = p["realised"][b]
    for a, v in p["realised"].items():
        if a == b: continue
        D.append({"split": p["split"], "family": p["family"], "action": a.split("->")[0],
                  "features": p["features"], "belief": p["belief"],
                  "delta": v - base, "abs": v, "base": base})
tr = [d for d in D if d["split"]=="train"]; hd = [d for d in D if d["split"]=="heldout"]
print(f"  samples: {len(tr)} train / {len(hd)} held-out\n")

# --- 1. reliability ceiling ---------------------------------------------------
print("  [1] RELIABILITY CEILING on R^2")
var_obs = np.var([d["delta"] for d in D])
var_noise = np.mean([d["abs"]*(1-d["abs"])/REPS + d["base"]*(1-d["base"])/REPS for d in D])
rel = max(0.0, 1 - var_noise/var_obs)
print(f"      Var(observed Delta) {var_obs:.5f}   Var(MC noise, {REPS} reps) {var_noise:.5f}")
print(f"      reliability = 1 - noise/observed = {rel:.3f}")
print(f"      -> no predictor can exceed R^2 ~= {rel:.3f} against this noisy target.")

# --- features ------------------------------------------------------------------
FE = sorted(tr[0]["features"]); ACTS = sorted({d["action"] for d in D})
def X(rows):
    M=[]
    for d in rows:
        v=[d["features"][k] for k in FE]
        v+=[1.0 if d["action"]==a else 0.0 for a in ACTS]
        b=d["belief"]; v+=[max(b) if b else 0.0, float(len(b))]
        M.append(v)
    return np.array(M)
ytr=np.array([d["delta"] for d in tr]); yhd=np.array([d["delta"] for d in hd])
def r2(y,p): return 1-np.sum((y-p)**2)/max(np.sum((y-y.mean())**2),1e-12)

# --- 2. simple baselines -------------------------------------------------------
print("\n  [2] SIMPLE BASELINES vs GBM (held-out R^2)")
preds={}
preds["zero"]=np.zeros(len(hd))
preds["train mean"]=np.full(len(hd), ytr.mean())
am={a:np.mean([d["delta"] for d in tr if d["action"]==a]) for a in ACTS}
preds["per-action mean"]=np.array([am[d["action"]] for d in hd])
def bud(d): return round(d["features"]["frac_budget_remaining"],1)
bm={}
for d in tr: bm.setdefault((d["action"],bud(d)),[]).append(d["delta"])
bm={k:np.mean(v) for k,v in bm.items()}
preds["action x budget mean"]=np.array([bm.get((d["action"],bud(d)), am[d["action"]]) for d in hd])
ridge=Ridge(alpha=1.0).fit(X(tr),ytr); preds["ridge (linear)"]=ridge.predict(X(hd))
gbm=HistGradientBoostingRegressor(max_depth=3,max_iter=200,learning_rate=0.05,
                                  min_samples_leaf=20,random_state=0).fit(X(tr),ytr)
preds["GBM"]=gbm.predict(X(hd))
for k,v in preds.items():
    print(f"      {k:<22} R^2 {r2(yhd,v):>+7.3f}   (ceiling {rel:.3f})")

# --- 3. estimand isolation -----------------------------------------------------
print("\n  [3] ESTIMAND ISOLATION — same states/features, two targets")
y_abs_tr=np.array([d["abs"] for d in tr])
gbm_abs=HistGradientBoostingRegressor(max_depth=3,max_iter=200,learning_rate=0.05,
                                      min_samples_leaf=20,random_state=0).fit(X(tr),y_abs_tr)
p_abs=gbm_abs.predict(X(hd)); p_del=preds["GBM"]
by=({},{})
for i,d in enumerate(hd):
    key=(d["family"],d["features"]["step"],round(d["base"],4))
    by[0].setdefault(key,[]).append((p_abs[i],d["delta"]))
    by[1].setdefault(key,[]).append((p_del[i],d["delta"]))
def pairacc(g):
    c=t=0
    for v in g.values():
        for i in range(len(v)):
            for j in range(i+1,len(v)):
                if abs(v[i][1]-v[j][1])<1e-9: continue
                t+=1; c+=int((v[i][0]>v[j][0])==(v[i][1]>v[j][1]))
    return c/max(t,1), t
a_abs,n1=pairacc(by[0]); a_del,n2=pairacc(by[1])
print(f"      target = absolute realised outcome : within-state pairwise {a_abs:.3f} (n={n1})")
print(f"      target = baseline-relative Delta   : within-state pairwise {a_del:.3f} (n={n2})")
print(f"      difference {a_del-a_abs:+.3f}  <- attributable to the TARGET, since")
print(f"      both orderings are identical by construction within a state.")

# --- 4/5. precision and policy-value curve ------------------------------------
print(f"\n  [4/5] PRECISION and POLICY-VALUE CURVE (base rate positive = "
      f"{np.mean(yhd>0):.1%}; 'always negative' scores {np.mean(yhd<=0):.1%})")
print(f"      {'thr':>6} {'n sel':>6} {'coverage':>9} {'precision':>10} "
      f"{'mean Delta':>11} {'policy gain':>12}")
for t in (-0.10,-0.05,0.0,0.02,0.05,0.10):
    sel=p_del>t
    if sel.sum()==0: continue
    print(f"      {t:>6.2f} {int(sel.sum()):>6} {sel.mean():>9.1%} "
          f"{np.mean(yhd[sel]>0):>10.1%} {yhd[sel].mean():>+11.4f} "
          f"{(yhd[sel].sum()/len(yhd)):>+12.4f}")
print(f"\n      policy gain = mean Delta over ALL decisions if we override only when")
print(f"      predicted > thr. The maximum over this curve is the observed")
print(f"      opportunity for THIS predictor -- not an upper bound on what is")
print(f"      achievable, which my earlier '~1pp ceiling' wrongly implied.")
