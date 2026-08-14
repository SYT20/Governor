"""Arm E — the Governor policy, built to the posture the experiments actually support.

Nothing here is aspirational. Every rule traces to a measured result:

  Stage 4A/4C   within-state action ranking is 0.62 overall, 0.74 on pairs whose
                realised values differ clearly. So the model is trusted only when
                its margin is wide; near-ties are treated as ties.
  Stage 4A      the median best-worst spread across actions is 0.000 -- at ~54% of
                states no action beats any other. A policy that agonises over those
                is spending compute to no effect, so ties are broken on COST.
  Stage 3       absolute probabilities are miscalibrated across regimes (2.8x the
                noise floor). Therefore probability never decides STOP on its own;
                verification does.
  critique_tests  a one-parameter logit offset fixes ~39% of that miscalibration
                from 60 episodes while preserving AUC exactly. So the probability
                channel is offset-corrected, not refit.
  H.1           hard constraints are deterministic and belong to the accountant,
                never to a probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from governor.corpus.build import Checkpoint
from governor.envs.synthbug import Action, Mode, SynthBug, Tier
from governor.policy.runner import EpisodeContext


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(slots=True)
class GovernorArm:
    """Ranking-first controller with cost tie-breaks and verified stopping."""

    model: object
    channels: dict = field(default_factory=dict)
    name: str = "E_governor"

    # Margin below which two actions are treated as indistinguishable. Set from
    # the measured ranking resolution: below roughly this gap the model's ordering
    # was not better than chance, so acting on it is superstition.
    tau: float = 0.05
    # Cost weight for the tie-break, in units of predicted success per unit cost.
    lambda_cost: float = 0.35
    # One-parameter recalibration offset (logit scale). 0.0 = uncorrected.
    intercept_offset: float = 0.0
    # Extra margin a terminal action must clear before the policy will stop.
    stop_margin: float = 0.02
    _n_hyp: int = 4

    def reset(self, task: SynthBug) -> None:
        self._n_hyp = task.config.n_hypotheses

    # -- scoring ---------------------------------------------------------------

    def _checkpoint(self, ctx: EpisodeContext, a: Action) -> Checkpoint:
        return Checkpoint(
            episode_id="live", decision_id=ctx.step, family="live", split="live",
            seed=0, action=str(a), mode=str(a.mode), tier=str(a.tier),
            was_random=False, n_admissible=1, features=ctx.features(),
            belief=list(ctx.belief), label=0,
        )

    def _score(self, ctx: EpisodeContext, actions: list[Action]) -> dict[Action, float]:
        cks = [self._checkpoint(ctx, a) for a in actions]
        raw = self.model.predict(cks)
        return {
            a: _sig(_logit(float(p)) + self.intercept_offset)
            for a, p in zip(actions, raw)
        }

    # -- policy ----------------------------------------------------------------

    def act(self, ctx: EpisodeContext, admissible: list[Action]) -> tuple[Action, str]:
        # 1. Verified success terminates immediately. This is deliberately NOT a
        #    probability decision -- Stage 3 showed absolute probabilities do not
        #    transfer across regimes, and stopping is the consumer least able to
        #    tolerate that.
        if ctx.verified:
            for a in admissible:
                if a.mode is Mode.STOP_VERIFIED:
                    return a, "STOP_VERIFIED"

        # 2. Hard budget floor: deterministic, owned by the accountant.
        frac = ctx.accountant.fraction_remaining()
        acting = [a for a in admissible if not a.mode.is_terminal]
        if frac < 0.20 or not acting:
            if ctx.n_by_mode.get("EXPLOIT", 0) > ctx.n_by_mode.get("VERIFY", 0):
                for a in admissible:
                    if a.mode is Mode.VERIFY and a.tier == Tier.T0:
                        return a, "BUDGET_FLOOR"
            for a in admissible:
                if a.mode.is_terminal:
                    return a, "BUDGET_FLOOR"
            return admissible[0], "BUDGET_FLOOR"

        # 3. Expected utility, with cost as a first-class term.
        #
        #    The first version of this arm used cost only as a tie-break, so above
        #    the margin it maximised P(success) and ignored price entirely. It then
        #    reached for the most expensive tier whenever the model liked it, and
        #    spent 40% more per success than the hand-tuned heuristic while winning
        #    18 points less. A budget-aware controller cannot have cost enter only
        #    when it happens to be indifferent.
        #
        #    Cost is expressed as a FRACTION OF WHAT REMAINS rather than in absolute
        #    units. The same action is cheap early and ruinous when the envelope is
        #    nearly spent, and the fraction captures that automatically -- no
        #    schedule, no hand-tuned decay.
        # Terminal actions must COMPETE, not be excluded. The first version scored
        # only non-terminal actions, which left exactly two ways to stop: tests
        # pass, or the budget floor forces it. There was no "further action is not
        # worth it" path at all -- section I's EU(STOP) = V(s), the entire stopping
        # mechanism. The measured consequence was severe: 17.8 decisions per
        # episode against the heuristic's 7.2, dithering until the envelope ran
        # out, which is why cost per success was 40% worse and success 22-38 points
        # lower.
        scorable = acting + [a for a in admissible if a.mode.is_terminal]
        scores = self._score(ctx, scorable)

        rem = ctx.accountant.remaining()
        eu: dict[Action, float] = {}
        for a in scorable:
            c = ctx.profile.estimate(a.action_class, "cost").value
            frac_cost = c / max(rem.get("cost", 0.0), 1e-9)
            eu[a] = scores[a] - self.lambda_cost * min(frac_cost, 1.0)

        ranked = sorted(scorable, key=lambda a: -eu[a])
        best = ranked[0]

        # Stopping is a decision the model is least equipped to make well
        # (Stage 3: absolute probabilities do not transfer across regimes), so it
        # must clear a margin rather than merely tie. Continuing is the safer error.
        if best.mode.is_terminal and eu[best] - max(
            (eu[a] for a in scorable if not a.mode.is_terminal), default=-1.0
        ) < self.stop_margin:
            ranked = [a for a in ranked if not a.mode.is_terminal] or ranked
            best = ranked[0]

        # 4. Wide margin on EU -> trust it. Narrow margin -> the model has not
        #    demonstrated it can resolve this ordering, so take the cheaper action.
        if len(ranked) == 1 or eu[best] - eu[ranked[1]] > self.tau:
            return best, "HIGHEST_EU"

        near = [a for a in ranked if eu[best] - eu[a] <= self.tau]
        cheapest = min(near, key=lambda a: ctx.profile.estimate(a.action_class, "cost").value)
        return cheapest, "TIE_BROKEN_ON_COST"
