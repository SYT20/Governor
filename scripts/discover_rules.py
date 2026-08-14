"""Corrected rule discovery. Four defects fixed from the first pass.

1. TERMINAL ACTIONS EXCLUDED. Choosing STOP_* mechanically ends the episode and
   fixes the outcome, so those "rules" restate the grading function rather than
   describing a decision. They dominated every top slot in pass 1.
2. WILSON INTERVALS instead of Welch. On 0/1 data where all treated units share a
   label the sample variance is zero and Welch's SE collapses to the control
   group's, producing a +-0.03 interval on n=8. That is not a valid interval for a
   Bernoulli mean from 8 draws.
3. MINIMUM SUPPORT IN BOTH ARMS (>=40 each).
4. BH DEPENDENCE ACKNOWLEDGED. Conditions are nested and overlapping, so BH is
   anti-conservative. Bonferroni reported alongside as the conservative bound.
"""
import sys, math, json
import numpy as np
sys.path.insert(0,"/Users/keshavgautam/Desktop/Suyash/Atlan Proj")
from scripts.fit_value_model import load

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0.0, c-h), min(1.0, c+h))

def diff_ci(ka, na, kb, nb):
    """Newcombe hybrid-score interval for a difference of proportions.
    Correct when a group is degenerate (all 0s or all 1s), unlike Welch."""
    l1,u1 = wilson(ka,na); l2,u2 = wilson(kb,nb)
    d = ka/na - kb/nb
    return d, d - math.sqrt((ka/na-l1)**2 + (u2-kb/nb)**2), \
              d + math.sqrt((u1-ka/na)**2 + (kb/nb-l2)**2)

ck=load("results/corpus_af.db")
tr=[c for c in ck if c.split=="train" and c.was_random and not c.mode.startswith("STOP")]
print(f"  randomised NON-TERMINAL decisions: {len(tr)} from "
      f"{len({c.episode_id for c in tr})} episodes")

FEATS=["max_belief","belief_margin","belief_entropy","frac_budget_remaining",
       "n_explore","n_exploit","n_verify","steps_since_new_evidence","step"]
acts=sorted({f"{c.mode}@{c.tier}" for c in tr})
MIN=40
tests=[]
for f in FEATS:
    cuts=sorted({round(np.percentile([c.features[f] for c in tr],q),4)
                 for q in (20,35,50,65,80)})
    for cut in cuts:
        for sense in (">","<="):
            sel=[c for c in tr if (c.features[f]>cut if sense==">" else c.features[f]<=cut)]
            for a in acts:
                A=[c.label for c in sel if f"{c.mode}@{c.tier}"==a]
                B=[c.label for c in sel if f"{c.mode}@{c.tier}"!=a]
                if len(A)<MIN or len(B)<MIN: continue
                d,lo,hi = diff_ci(sum(A),len(A),sum(B),len(B))
                # two-sided p from the normal approx on the pooled SE
                pa,pb=sum(A)/len(A),sum(B)/len(B)
                pp=(sum(A)+sum(B))/(len(A)+len(B))
                se=math.sqrt(max(pp*(1-pp)*(1/len(A)+1/len(B)),1e-18))
                z=abs(pa-pb)/se
                p=2*(1-0.5*(1+math.erf(z/math.sqrt(2))))
                tests.append((f,sense,cut,a,d,p,len(A),lo,hi))

m=len(tests); tests.sort(key=lambda x:x[5])
bh=[]; 
for i,t in enumerate(tests,1):
    if t[5] <= 0.05*i/m: bh=tests[:i]
bonf=[t for t in tests if t[5] <= 0.05/m]
print(f"  tests: {m}   surviving BH-FDR(0.05): {len(bh)}   surviving Bonferroni: {len(bonf)}")
print(f"  (conditions are nested/overlapping, so BH is anti-conservative here;")
print(f"   Bonferroni is the defensible bound)\n")

sig=[t for t in bonf if t[7]*t[8] > 0]     # CI excludes zero
print(f"  {'condition':<40} {'action':<14} {'effect':>8} {'95% CI (Newcombe)':>20} {'n':>5}")
print("  "+"-"*94)
for f,sense,cut,a,d,p,n,lo,hi in sorted(sig,key=lambda x:-abs(x[4]))[:14]:
    print(f"  {f+' '+sense+f' {cut:.3f}':<40} {a:<14} {d:>+8.3f} "
          f"[{lo:>+7.3f},{hi:>+7.3f}] {n:>5}")

print("\n  === Non-circularity test ===")
be=[t for t in sig if t[0] in ("max_belief","belief_margin") and t[1]==">"
    and t[3].startswith("EXPLOIT") and t[4]>0]
if be:
    b=max(be,key=lambda x:x[4])
    print(f"  Belief-threshold -> EXPLOIT recovered independently:")
    print(f"    {b[0]} > {b[2]:.3f} -> {b[3]}   effect {b[4]:+.3f} "
          f"[{b[7]:+.3f},{b[8]:+.3f}]  n={b[6]}")
    print(f"  Stage 6 authored max_belief >= 0.55 by hand. Discovery reached the")
    print(f"  same rule FAMILY from randomised data alone, under Bonferroni.")
else:
    print("  NOT recovered under Bonferroni with CI excluding zero.")
    print("  Stage 6's structure was hand-chosen and is NOT independently validated.")
json.dump([{"f":t[0],"sense":t[1],"cut":t[2],"action":t[3],"effect":t[4],
            "n":t[6],"lo":t[7],"hi":t[8]} for t in sig],
          open("results/discovered_rules.json","w"), indent=1)
