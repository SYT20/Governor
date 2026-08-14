"""Parametric heuristic: the same rule skeleton, with the constants exposed.

Stage 6 asks whether the heuristic's advantage comes from *structure* (a threshold
policy over a few state variables) or from the specific constants a human chose.
Distilling the heuristic by imitation would be circular -- its rules are source
code, so any fitted tree recovers them by construction. Ablating the rules and
LEARNING the thresholds is not circular, and answers the question that matters:

    can a policy with this structure and fitted parameters beat both the
    hand-tuned heuristic and the value-function controller?

If yes, the finding is that the representation was wrong all along -- the task is
to learn a small number of switching thresholds, not P(success | s, a).
"""

from __future__ import annotations

from dataclasses import dataclass

from governor.arms.baselines import _pick, _terminal
from governor.envs.synthbug import Action, Mode, SynthBug, Tier
from governor.policy.runner import EpisodeContext


@dataclass(slots=True)
class ParametricHeuristic:
    """Every constant in the hand-tuned heuristic, exposed and ablatable."""

    name: str = "P_parametric"

    # Rule 3: commit to a repair once the leading hypothesis is this strong.
    confident: float = 0.55
    max_exploits: int = 2
    # Rule 2: below this fraction of budget, verify cheaply and get out.
    low_budget: float = 0.25
    # Tier escalation points: spend more while budget is plentiful.
    exploit_hi_frac: float = 0.60
    explore_hi_frac: float = 0.70

    # Ablation switches. Disabling a rule tells us whether it is load-bearing.
    use_verified_stop: bool = True
    use_budget_floor: bool = True
    use_confident_exploit: bool = True
    use_verify_after_edit: bool = True
    use_tier_escalation: bool = True

    def reset(self, task: SynthBug) -> None:
        pass

    def act(self, ctx: EpisodeContext, admissible: list[Action]) -> tuple[Action, str]:
        f = ctx.features()
        verified = f["last_test_pass"] == 1.0
        frac = f["frac_budget_remaining"]
        top = f["max_belief"]
        n_exploit, n_verify = f["n_exploit"], f["n_verify"]

        if self.use_verified_stop and verified:
            return _terminal(admissible, ctx), "STOP_VERIFIED"

        if self.use_budget_floor and frac < self.low_budget:
            if n_exploit > 0 and n_verify == 0:
                a = _pick(admissible, Mode.VERIFY, Tier.T0)
                if a:
                    return a, "BUDGET_FLOOR"
            return _terminal(admissible, ctx), "BUDGET_FLOOR"

        if self.use_confident_exploit and top >= self.confident and n_exploit < self.max_exploits:
            tier = (Tier.T2 if (self.use_tier_escalation and frac > self.exploit_hi_frac)
                    else Tier.T1)
            a = _pick(admissible, Mode.EXPLOIT, tier)
            if a:
                return a, "CONFIDENT_EXPLOIT"

        if self.use_verify_after_edit and n_exploit > n_verify:
            a = _pick(admissible, Mode.VERIFY, Tier.T1)
            if a:
                return a, "VERIFY_AFTER_EDIT"

        tier = (Tier.T2 if (self.use_tier_escalation and frac > self.explore_hi_frac)
                else Tier.T1)
        a = _pick(admissible, Mode.EXPLORE, tier)
        if a:
            return a, "EXPLORE_LEADER"
        return _terminal(admissible, ctx), "NO_PROGRESS"
