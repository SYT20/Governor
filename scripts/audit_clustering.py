import sys, json, random
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fit_value_model import load
ck=load("results/corpus_af.db")
tr=[c for c in ck if c.split=="train" and c.was_random and not c.mode.startswith("STOP")]
rules=json.load(open("results/discovered_rules.json"))
print(f"  rules under audit: {len(rules)}\n")
rng=np.random.default_rng(0); B=1200
rows=[]
for r in rules:
    f,sense,cut,act=r["f"],r["sense"],r["cut"],r["action"]
    sel=[c for c in tr if (c.features[f]>cut if sense==">" else c.features[f]<=cut)]
    # pre-aggregate per episode: (sumA, nA, sumB, nB)
    agg=defaultdict(lambda:[0,0,0,0])
    for c in sel:
        g=agg[c.episode_id]
        if f"{c.mode}@{c.tier}"==act: g[0]+=c.label; g[1]+=1
        else: g[2]+=c.label; g[3]+=1
    M=np.array(list(agg.values()),dtype=float)
    if len(M)<10: continue
    nA=int(M[:,1].sum()); epsA=int((M[:,1]>0).sum())
    idx=rng.integers(0,len(M),size=(B,len(M)))
    S=M[idx]                                  # (B, n_eps, 4)
    sA,cA,sB,cB=S[:,:,0].sum(1),S[:,:,1].sum(1),S[:,:,2].sum(1),S[:,:,3].sum(1)
    ok=(cA>=5)&(cB>=5)
    d=np.where(ok,(sA/np.maximum(cA,1))-(sB/np.maximum(cB,1)),np.nan)
    d=d[~np.isnan(d)]
    if len(d)<200: continue
    lo,hi=np.percentile(d,[2.5,97.5])
    rows.append((r,nA,epsA,lo,hi,lo*hi>0))
rows.sort(key=lambda x:-abs(x[0]["effect"]))
print(f"  {'rule':<42} {'n_ck':>5} {'n_ep':>5} {'naive CI':>17} {'CLUSTERED CI':>19} {'ok':>4}")
print("  "+"-"*100)
for r,nA,epsA,lo,hi,ok in rows[:14]:
    cond=f"{r['f']} {r['sense']} {r['cut']:.2f} -> {r['action']}"
    print(f"  {cond:<42} {nA:>5} {epsA:>5} [{r['lo']:>+6.3f},{r['hi']:>+6.3f}] "
          f"[{lo:>+7.3f},{hi:>+7.3f}] {'YES' if ok else 'no':>4}")
surv=[x for x in rows if x[5]]
infl=[(hi-lo)/max(r['hi']-r['lo'],1e-9) for r,_,_,lo,hi,_ in rows]
print(f"\n  === AUDIT RESULT ===")
print(f"  naive checkpoint-level 'significant'   : {len(rows)}")
print(f"  survives EPISODE-CLUSTERED bootstrap   : {len(surv)}")
print(f"  median CI width inflation              : {np.median(infl):.2f}x")
print(f"  median checkpoints per episode per arm : "
      f"{np.median([n/max(e,1) for _,n,e,_,_,_ in rows]):.2f}")
