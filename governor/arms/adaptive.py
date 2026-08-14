"""Arm P — budget-conditional thresholds. The policy the project was looking for.

The route here matters, because every negative result along the way is evidence
for this design rather than wasted effort:

  Stage 3   a calibrated value model has real state-discrimination signal
            (AUC 0.84, Brier +34%) but its probabilities do not transfer across
            regimes.
  Stage 4   that AUC was measuring state difficulty, not action choice. Within-
            state action ranking is 0.61, and 0.53 on training regimes -- chance.
            Neither an advantage decomposition nor action-conditioned features
            nor 2,000 episodes moved it much.
  Stage 5   the value-function controller loses to a hand-tuned heuristic by 34pp.
  Stage 6A  that heuristic is essentially ONE rule: commit to a repair once the
            leading hypothesis crosses a confidence threshold. Three other rules
            change task success by exactly zero.
  Stage 6C  fitting the thresholds beats the value function 93.3% to 48.3% at full
            budget -- but a FIXED threshold set optimised at one budget collapses
            at another (10.0% at 25%).
  Stage 6D  the optimal thresholds move monotonically with the envelope.
  Stage 6E  interpolating them generalises to budget levels never tuned on.

So the estimation problem was wrong the whole time. Learning P(success | s, a)
over sixteen features, to approximate a decision that one comparison against one
constant makes correctly, is a far harder problem than learning the constant --
and the constant depends on how much budget there is.

Conditioning on the envelope is not leakage. The resource envelope is an INPUT,
specified by the user at episode start (brief section 11). The policy uses
something it genuinely has, unlike OracleArm which reads the hidden true cause.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from governor.arms.parametric import ParametricHeuristic
from governor.envs.synthbug import Action, SynthBug
from governor.policy.runner import EpisodeContext

# Thresholds fitted independently at three envelope scales on TRAINING regimes
# (Stage 6D), then held fixed. Nothing below is re-tuned per task.
#
# Read down the columns and the strategy is legible: with money, wait for near
# certainty before committing and hold little back; without money, commit on a
# bare majority, reserve four times as much, and stop escalating tiers.
FITTED: dict[float, dict] = {
    1.00: dict(confident=0.947, max_exploits=4, low_budget=0.108,
               exploit_hi_frac=0.629, explore_hi_frac=0.517, use_tier_escalation=True),
    0.50: dict(confident=0.639, max_exploits=2, low_budget=0.141,
               exploit_hi_frac=0.629, explore_hi_frac=0.517, use_tier_escalation=True),
    0.25: dict(confident=0.503, max_exploits=4, low_budget=0.430,
               exploit_hi_frac=0.629, explore_hi_frac=0.517, use_tier_escalation=False),
}


def thresholds_for(scale: float) -> dict:
    """Linearly interpolate the fitted thresholds at an arbitrary envelope scale.

    Stage 6E verified this generalises: the interpolated policy beats the
    hand-tuned heuristic at 75% and 35% budget, neither of which was tuned on.
    """
    xs = sorted(FITTED)
    if scale <= xs[0]:
        lo = hi = xs[0]
    elif scale >= xs[-1]:
        lo = hi = xs[-1]
    else:
        lo = max(x for x in xs if x <= scale)
        hi = min(x for x in xs if x >= scale)
    t = 0.0 if hi == lo else (scale - lo) / (hi - lo)
    a, b = FITTED[lo], FITTED[hi]
    return dict(
        confident=a["confident"] + t * (b["confident"] - a["confident"]),
        max_exploits=int(round(a["max_exploits"] + t * (b["max_exploits"] - a["max_exploits"]))),
        low_budget=a["low_budget"] + t * (b["low_budget"] - a["low_budget"]),
        exploit_hi_frac=a["exploit_hi_frac"],
        explore_hi_frac=a["explore_hi_frac"],
        use_tier_escalation=(a if t < 0.5 else b)["use_tier_escalation"],
    )


@dataclass(slots=True)
class AdaptiveArm:
    """Selects its switching thresholds from the size of the envelope it is given."""

    scale: float = 1.0
    name: str = "P_adaptive"
    _inner: ParametricHeuristic | None = field(default=None, init=False)

    def reset(self, task: SynthBug) -> None:
        self._inner = ParametricHeuristic(**thresholds_for(self.scale))
        self._inner.reset(task)

    def act(self, ctx: EpisodeContext, admissible: list[Action]) -> tuple[Action, str]:
        assert self._inner is not None, "reset() must run before act()"
        return self._inner.act(ctx, admissible)
