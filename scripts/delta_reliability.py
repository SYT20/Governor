#!/usr/bin/env python3
"""Three demanded corrections, tested rather than argued.

1. RELIABILITY WITH CRN COVARIANCE. My earlier ceiling assumed Y_a and Y_b were
   independent. Common random numbers deliberately correlate them, so
   Var(Y_a - Y_b) = Var(Y_a) + Var(Y_b) - 2Cov, and my noise estimate was too
   large -- meaning the ceiling was too LOW and R^2 0.353 is a smaller fraction of
   it than the 80% I claimed. Measured here two ways: empirical covariance, and a
   split-half (5+5 replicate) test-retest reliability with Spearman-Brown.

2. THRESHOLD POST-SELECTION. I evaluated several thresholds on held-out data and
   quoted the best. That is optimistic. Now a strict three-way split BY REGIME:
   fit on train families, choose the threshold on VALIDATION families, evaluate
   once on TEST families never touched.

3. TOP-1 AND REGRET. R^2 does not tell the controller which action to take.
   Reported here: within-state top-1 accuracy and decision regret.
"""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from governor.accounting.meter import Accountant, Envelope
from governor.arms.baselines import HeuristicArm
from governor.envs.families import BY_NAME, FAMILIES, heldout_families, train_families
from governor.envs.synthbug import Tier
from governor.experiments.branch import _fresh_context, _rollout
from governor.models.cost import Reserve
from governor.policy.runner import (_apply_observation, _charge_or_truncate,
                                    deterministic_candidates)
from scripts.build_corpus import fit_profile_over_families
from scripts.run_experiment import fit_channels
from sklearn.ensemble import HistGradientBoostingRegressor

CACHE = Path("results/delta_replicates.json")
REPS = 12

def collect(fams, seeds, profile, channels, envelope):
    out=[]
    for split, fam in fams:
        n0=len(out)
        for seed in seeds:
            task=fam.task(seed); acc=Accountant(envelope=envelope)
            ctx=_fresh_context(task,acc,profile); arm=HeuristicArm(); arm.reset(task)
            reserve=Reserve(profile); snaps=[]
            while not task.terminated and ctx.step<40:
                cands=deterministic_candidates(ctx,tuple(Tier))
                need=reserve.required(verified=ctx.verified)
                adm=[a for a in cands if acc.admissible(profile.vector(a.action_class),
                     reserve={} if a.mode.is_terminal else need)]
                if not adm: break
                chosen,_=arm.act(ctx,adm)
                nt=[a for a in adm if not a.mode.is_terminal]
                if len(nt)>=2 and not chosen.mode.is_terminal:
                    snaps.append((ctx.step,copy.deepcopy(task),copy.deepcopy(ctx),list(nt),chosen))
                rl=task.cost_of(chosen); _,tr=_charge_or_truncate(acc,chosen.action_class,rl)
                ob=task.step(chosen); _apply_observation(ctx,chosen,ob,channels,tr); ctx.step+=1
            if not snaps: continue
            for frac in (0.3,0.7):
                i=min(len(snaps)-1,int(frac*len(snaps)))
                step_i,st,sc,cands,baseline=snaps[i]
                cands=cands[:4]
                if baseline not in cands: cands=[baseline]+cands[:3]
                rec={"family":fam.name,"split":split,"seed":seed,"decision_id":step_i,
                     "features":dict(sc.features()),"belief":list(sc.belief),
                     "baseline":str(baseline),"reps":{}}
                for cand in cands:
                    ys=[]
                    for r in range(REPS):
                        t2=copy.deepcopy(st); c2=copy.deepcopy(sc)
                        c2.accountant=copy.deepcopy(sc.accountant)
                        t2._rng.seed(seed*100_003+step_i*997+r*31)   # CRN across actions
                        rl=t2.cost_of(cand)
                        _,tr=_charge_or_truncate(c2.accountant,cand.action_class,rl)
                        ob=t2.step(cand); _apply_observation(c2,cand,ob,channels,tr); c2.step+=1
                        ys.append(int(_rollout(t2,c2,profile,channels,tuple(Tier),40)))
                    rec["reps"][str(cand)]=ys      # PER-REPLICATE, not just the mean
                out.append(rec)
        print(f"    {split:<6} {fam.name:<18} +{len(out)-n0:>3} (total {len(out)})",flush=True)
    return out

def main()->int:
    ch,_=fit_channels(__import__("governor.envs.synthbug",fromlist=["SynthConfig"]).SynthConfig(),120)
    prof=fit_profile_over_families(train_families(),40)
    env=Envelope(tokens=60_000,cost=0.30,wall_s=200.0,tool_calls=25)
    TR=train_families()[:7]; VA=train_families()[7:10]; TE=heldout_families()
    print(f"  THREE-WAY REGIME SPLIT (no post-selection)")
    print(f"    train      {[f.name for f in TR]}")
    print(f"    validation {[f.name for f in VA]}   <- threshold chosen here")
    print(f"    test       {[f.name for f in TE]}   <- evaluated ONCE\n")
    if CACHE.exists():
        pts=json.loads(CACHE.read_text()); print(f"  loaded {len(pts)} cached")
    else:
        fams=[("train",f) for f in TR]+[("val",f) for f in VA]+[("test",f) for f in TE]
        pts=collect(fams,range(90_000,90_016),prof,ch,env)
        CACHE.write_text(json.dumps(pts)); print(f"  collected {len(pts)}")

    # ---- 1. reliability with CRN covariance -------------------------------
    print("\n  [1] RELIABILITY — CRN covariance and split-half")
    dif,cov_terms,sh_a,sh_b=[],[],[],[]
    for p in pts:
        b=p["baseline"]
        if b not in p["reps"]: continue
        yb=np.array(p["reps"][b],dtype=float)
        for a,ya in p["reps"].items():
            if a==b: continue
            ya=np.array(ya,dtype=float)
            dif.append(ya.mean()-yb.mean())
            cov_terms.append(np.cov(ya,yb,ddof=1)[0,1] if ya.std()>0 and yb.std()>0 else 0.0)
            h=REPS//2
            sh_a.append(ya[:h].mean()-yb[:h].mean()); sh_b.append(ya[h:].mean()-yb[h:].mean())
    dif=np.array(dif); sh_a=np.array(sh_a); sh_b=np.array(sh_b)
    var_obs=dif.var()
    naive=np.mean([np.mean(p["reps"][a])*(1-np.mean(p["reps"][a]))/REPS
                   + np.mean(p["reps"][p["baseline"]])*(1-np.mean(p["reps"][p["baseline"]]))/REPS
                   for p in pts if p["baseline"] in p["reps"] for a in p["reps"] if a!=p["baseline"]])
    mean_cov=float(np.mean(cov_terms))
    corrected=naive-2*mean_cov/REPS
    r_half=float(np.corrcoef(sh_a,sh_b)[0,1])
    sb=2*r_half/(1+r_half) if r_half>-1 else 0.0
    print(f"      Var(observed Delta)                 {var_obs:.5f}")
    print(f"      naive noise (independence assumed)  {naive:.5f}  -> reliability {1-naive/var_obs:.3f}")
    print(f"      mean Cov(Y_a,Y_b) from CRN          {mean_cov:+.5f}")
    print(f"      covariance-corrected noise          {corrected:.5f}  -> reliability {1-corrected/var_obs:.3f}")
    print(f"      SPLIT-HALF r ({REPS//2}+{REPS//2} reps)              {r_half:+.3f}  "
          f"-> Spearman-Brown reliability {sb:.3f}")
    print(f"      => empirical ceiling on R^2 is the split-half value, {sb:.3f}")

    # ---- model + 3-way split -----------------------------------------------
    D=[]
    for p in pts:
        b=p["baseline"]
        if b not in p["reps"]: continue
        base=float(np.mean(p["reps"][b]))
        for a,ys in p["reps"].items():
            if a==b: continue
            D.append({"split":p["split"],"key":(p["family"],p["seed"],p["decision_id"]),
                      "action":a.split("->")[0],"features":p["features"],
                      "belief":p["belief"],"delta":float(np.mean(ys))-base})
    FE=sorted(D[0]["features"]); ACTS=sorted({d["action"] for d in D})
    def X(rows):
        return np.array([[d["features"][k] for k in FE]
                         +[1.0 if d["action"]==a else 0.0 for a in ACTS]
                         +[max(d["belief"]) if d["belief"] else 0.0] for d in rows])
    tr=[d for d in D if d["split"]=="train"]; va=[d for d in D if d["split"]=="val"]
    te=[d for d in D if d["split"]=="test"]
    print(f"\n  samples: {len(tr)} train / {len(va)} val / {len(te)} test")
    m=HistGradientBoostingRegressor(max_depth=3,max_iter=200,learning_rate=0.05,
                                    min_samples_leaf=20,random_state=0).fit(X(tr),
                                    np.array([d["delta"] for d in tr]))
    pv=m.predict(X(va)); pt=m.predict(X(te))
    yv=np.array([d["delta"] for d in va]); yt=np.array([d["delta"] for d in te])
    r2=lambda y,p:1-np.sum((y-p)**2)/max(np.sum((y-y.mean())**2),1e-12)
    print(f"      held-out R^2: val {r2(yv,pv):+.3f}   test {r2(yt,pt):+.3f}   "
          f"(ceiling {sb:.3f})")

    # ---- 2. threshold on VAL, evaluated once on TEST -----------------------
    print("\n  [2] THRESHOLD chosen on VALIDATION, evaluated ONCE on TEST")
    grid=np.arange(-0.05,0.12,0.01); best,bestg=None,-9
    for t in grid:
        s=pv>t
        g=yv[s].sum()/len(yv) if s.sum() else -9
        if g>bestg: bestg,best=g,t
    s=pt>best
    print(f"      chosen threshold {best:+.2f}  (validation gain {bestg:+.4f})")
    if s.sum():
        print(f"      TEST: n={int(s.sum())}/{len(yt)}  coverage {s.mean():.1%}  "
              f"precision {np.mean(yt[s]>0):.1%}  mean Delta {yt[s].mean():+.4f}  "
              f"policy gain {yt[s].sum()/len(yt):+.4f}")
    else:
        print(f"      TEST: threshold selected nothing.")

    # ---- 3. top-1 and regret ------------------------------------------------
    print("\n  [3] TOP-1 SELECTION and REGRET on TEST states")
    by={}
    for i,d in enumerate(te): by.setdefault(d["key"],[]).append((pt[i],d["delta"]))
    ok=[v for v in by.values() if len(v)>=2]
    top1=np.mean([max(v,key=lambda x:x[0])[1]==max(x[1] for x in v) for v in ok])
    reg=np.mean([max(x[1] for x in v)-max(v,key=lambda x:x[0])[1] for v in ok])
    rnd=np.mean([max(x[1] for x in v)-np.mean([x[1] for x in v]) for v in ok])
    print(f"      states {len(ok)}   top-1 accuracy {top1:.1%}   mean regret {reg:+.4f}")
    print(f"      random-choice regret {rnd:+.4f}  -> model reduces regret by "
          f"{(1-reg/max(rnd,1e-9)):.1%}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
