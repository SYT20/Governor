"""GATEWAY TEST: can a policy built ONLY from discovered rules beat the baseline?

No hand-authored thresholds. Rules were discovered on TRAINING regimes from
randomised decisions only. Evaluation is on HELD-OUT regimes the discovery never
saw, and at budget levels not used for discovery.

Terminal handling is harness mechanics, not policy: stop when verified, stop when
nothing is affordable. Terminals were excluded from discovery because choosing
STOP_* mechanically fixes the outcome and merely restates the grading function.

If this loses, rule extraction is not the breakthrough and the Ruflo-inspired
architecture should not be built.
"""
import sys, json
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from governor.accounting.meter import Envelope
from governor.arms.adaptive import AdaptiveArm
from governor.arms.baselines import FixedArm, HeuristicArm, _terminal
from governor.envs.families import heldout_families, train_families
from governor.envs.synthbug import Action, SynthBug, SynthConfig
from governor.policy.runner import EpisodeContext, run_episode
from scripts.build_corpus import fit_profile_over_families
from scripts.run_experiment import fit_channels

RULES=json.load(open("results/discovered_rules.json"))
print(f"  discovered rules in bank: {len(RULES)}  (train regimes, randomised decisions only)")

@dataclass(slots=True)
class RuleBankArm:
    name: str = "R_rulebank"
    rules: list = field(default_factory=lambda: RULES)
    _fired: int = 0
    _fallback: int = 0
    def reset(self, task: SynthBug) -> None:
        self._fired=0; self._fallback=0
    def act(self, ctx: EpisodeContext, admissible):
        if ctx.verified:
            return _terminal(admissible, ctx), "STOP_VERIFIED"
        acting=[a for a in admissible if not a.mode.is_terminal]
        if not acting:
            return _terminal(admissible, ctx), "NO_ADMISSIBLE"
        f=ctx.features()
        score={}
        for a in acting:
            key=f"{a.mode}@{a.tier}"
            tot=0.0
            for r in self.rules:
                if r["action"]!=key: continue
                v=f.get(r["f"])
                if v is None: continue
                if (v>r["cut"]) if r["sense"]==">" else (v<=r["cut"]):
                    tot+=r["effect"]
            score[a]=tot
        best=max(score,key=score.get)
        if abs(score[best])<1e-9:
            self._fallback+=1
            return min(acting,key=lambda a: ctx.profile.estimate(a.action_class,"cost").value), "NO_RULE_FIRED"
        self._fired+=1
        return best, "RULE"

ch,_=fit_channels(SynthConfig(),120)
prof=fit_profile_over_families(train_families(),40)
ref=Envelope(tokens=60_000,cost=0.30,wall_s=200.0,tool_calls=25)

def ev(arm,scale,n=60):
    w=[];c=0.0;env=ref.scaled(scale)
    for fam in heldout_families():
        for sd in range(70_000,70_000+n):
            r=run_episode(task=fam.task(sd),arm=arm,envelope=env,profile=prof,
                          channel_for=ch,store=None,budget_scale=scale)
            w.append(int(r.succeeded)); c+=r.consumed["cost"]
    return np.array(w), c/max(sum(w),1)

def mcnemar(a,b):
    n01=int(((a==0)&(b==1)).sum()); n10=int(((a==1)&(b==0)).sum()); n=n01+n10
    if n==0: return 1.0
    from math import comb
    k=min(n01,n10)
    return min(1.0, 2*sum(comb(n,i) for i in range(k+1))/2**n)

print(f"\n  HELD-OUT regimes. * = budget level never used in discovery or tuning.\n")
print(f"  {'budget':>8} {'A_fixed':>9} {'C_hand':>9} {'P_adapt':>9} {'R_rules':>9} "
      f"{'R vs C':>8} {'p':>8}")
print("  "+"-"*68)
for scale,star in ((1.00,""),(0.60,"*"),(0.35,"*"),(0.25,"")):
    wa,_=ev(FixedArm(),scale); wc,cc=ev(HeuristicArm(),scale)
    wp,_=ev(AdaptiveArm(scale=scale),scale); wr,cr=ev(RuleBankArm(),scale)
    p=mcnemar(wc,wr)
    print(f"  {f'{scale:.0%}{star}':>8} {wa.mean():>9.1%} {wc.mean():>9.1%} "
          f"{wp.mean():>9.1%} {wr.mean():>9.1%} {wr.mean()-wc.mean():>+8.1%} {p:>8.4f}")
