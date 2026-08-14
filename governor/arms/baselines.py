"""Baseline arms A, B, C, the epsilon-greedy corpus collector, and the oracle.

Decision Record section J.3. Governor itself (arm E) needs the value model, which
lands at Stage 3; these arms exist first so that when it arrives there is already
something honest to beat.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from governor.envs.synthbug import Action, Mode, SynthBug, Tier
from governor.policy.runner import Arm, EpisodeContext


def _pick(admissible: list[Action], mode: Mode, tier: Tier, target: int | None = None):
    """First admissible action matching a spec, else None."""
    for a in admissible:
        if a.mode is mode and a.tier == tier and (target is None or a.target == target):
            return a
    for a in admissible:  # relax the tier
        if a.mode is mode and (target is None or a.target == target):
            return a
    return None


def _terminal(admissible: list[Action], ctx: EpisodeContext) -> Action:
    want = Mode.STOP_VERIFIED if ctx.verified else Mode.STOP_UNVERIFIED
    for a in admissible:
        if a.mode is want:
            return a
    for a in admissible:
        if a.mode.is_terminal:
            return a
    return admissible[0]


@dataclass(slots=True)
class FixedArm:
    """Arm A. One tier, one rigid script: explore k times, exploit, verify, stop.

    The floor. Represents an agent whose scaffold spends the same effort on every
    task regardless of how the task is going.
    """

    tier: Tier = Tier.T1
    n_explore: int = 3
    name: str = "A_fixed"

    def reset(self, task: SynthBug) -> None:
        pass

    def act(self, ctx: EpisodeContext, admissible: list[Action]) -> tuple[Action, str]:
        done_explore = ctx.n_by_mode.get("EXPLORE", 0)
        if done_explore < self.n_explore:
            a = _pick(admissible, Mode.EXPLORE, self.tier)
            if a:
                return a, "FIXED_SCRIPT"
        if ctx.n_by_mode.get("EXPLOIT", 0) == 0:
            a = _pick(admissible, Mode.EXPLOIT, self.tier)
            if a:
                return a, "FIXED_SCRIPT"
        if ctx.n_by_mode.get("VERIFY", 0) == 0:
            a = _pick(admissible, Mode.VERIFY, self.tier)
            if a:
                return a, "FIXED_SCRIPT"
        return _terminal(admissible, ctx), "FIXED_SCRIPT"


@dataclass(slots=True)
class StaticRoutedArm:
    """Arm B. Picks a tier once from task features, then never switches.

    Isolates *routing* from *switching*: if Governor beats A but not B, the win came
    from choosing a good tier up front, not from adapting mid-episode.
    """

    name: str = "B_static"
    n_explore: int = 3
    _tier: Tier = field(default=Tier.T1, init=False)

    def reset(self, task: SynthBug) -> None:
        # More candidate causes -> harder task -> route to a stronger tier.
        k = task.config.n_hypotheses
        self._tier = Tier.T0 if k <= 2 else (Tier.T1 if k <= 4 else Tier.T2)

    def act(self, ctx: EpisodeContext, admissible: list[Action]) -> tuple[Action, str]:
        inner = FixedArm(tier=self._tier, n_explore=self.n_explore)
        a, _ = inner.act(ctx, admissible)
        return a, "STATIC_ROUTE"


@dataclass(slots=True)
class HeuristicArm:
    """Arm C. The honest strong baseline.

    A handful of sensible rules. Decision Record section J.3: if this ties Governor,
    that is a finding to report, not a failure to hide. Everything here is the kind
    of rule a competent engineer writes in an afternoon.
    """

    name: str = "C_heuristic"
    confident: float = 0.55
    low_budget: float = 0.25
    max_exploits: int = 2

    def reset(self, task: SynthBug) -> None:
        pass

    def act(self, ctx: EpisodeContext, admissible: list[Action]) -> tuple[Action, str]:
        frac = ctx.accountant.fraction_remaining()
        top = max(ctx.belief)
        best_h = ctx.belief.index(top)

        # Rule 1: tests passed -> stop. Never spend after success.
        if ctx.verified:
            return _terminal(admissible, ctx), "STOP_VERIFIED"

        # Rule 2: budget nearly gone -> cheapest verify, then out.
        if frac < self.low_budget:
            if ctx.n_by_mode.get("EXPLOIT", 0) > 0 and ctx.n_by_mode.get("VERIFY", 0) == 0:
                a = _pick(admissible, Mode.VERIFY, Tier.T0)
                if a:
                    return a, "BUDGET_FLOOR"
            return _terminal(admissible, ctx), "BUDGET_FLOOR"

        # Rule 3: confident enough -> attempt the fix.
        if top >= self.confident and ctx.n_by_mode.get("EXPLOIT", 0) < self.max_exploits:
            tier = Tier.T2 if frac > 0.6 else Tier.T1
            a = _pick(admissible, Mode.EXPLOIT, tier, best_h)
            if a:
                return a, "CONFIDENT_EXPLOIT"

        # Rule 4: a repair was attempted but not checked -> check it.
        if ctx.n_by_mode.get("EXPLOIT", 0) > ctx.n_by_mode.get("VERIFY", 0):
            a = _pick(admissible, Mode.VERIFY, Tier.T1)
            if a:
                return a, "VERIFY_AFTER_EDIT"

        # Rule 5: otherwise reduce uncertainty, escalating when budget is healthy.
        tier = Tier.T2 if frac > 0.7 else Tier.T1
        a = _pick(admissible, Mode.EXPLORE, tier, best_h)
        if a:
            return a, "EXPLORE_LEADER"

        return _terminal(admissible, ctx), "NO_ADMISSIBLE_PROGRESS"


@dataclass(slots=True)
class EpsilonGreedyCollector:
    """Corpus collector. Decision Record section G.1 -- the key addition of rev 2.

    A corpus generated by a single fixed policy has no action overlap: in any given
    state you only ever observe the action that policy took, so Q(s,a) for every
    other action is unidentifiable and no amount of regularisation recovers it.

    This wraps a base policy and takes a uniformly random admissible action with
    probability epsilon, which restores overlap and makes the randomised subset
    unconfounded by construction.
    """

    base: Arm = field(default_factory=HeuristicArm)
    epsilon: float = 0.30
    seed: int = 0
    name: str = "collector"
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def reset(self, task: SynthBug) -> None:
        self._rng = random.Random(self.seed * 100_003 + task.seed)
        self.base.reset(task)

    def act(self, ctx: EpisodeContext, admissible: list[Action]) -> tuple[Action, str]:
        if self._rng.random() < self.epsilon:
            return self._rng.choice(admissible), "EXPLORATION_RANDOM"
        return self.base.act(ctx, admissible)


@dataclass(slots=True)
class OracleArm:
    """Arm F. Knows the true cause; measures the achievable ceiling.

    Not a policy -- it cheats. Its purpose is to bound how much of the remaining gap
    is attributable to *decision quality* versus irreducible environment stochasticity
    (a repair aimed correctly still fails with probability 1 - p_fix).
    """

    name: str = "F_oracle"
    _truth: int = field(default=0, init=False)

    def reset(self, task: SynthBug) -> None:
        self._truth = task.true_cause

    def propose(self, ctx: EpisodeContext) -> list[Action]:
        """Contribute the truth-targeted repair at every tier.

        Without this the oracle is capped by whatever the belief-ordered
        deterministic set happens to offer, which made it score *below* the
        heuristic -- an oracle that cannot name the answer is not a ceiling.
        """
        return [Action(Mode.EXPLOIT, t, self._truth) for t in Tier]

    def act(self, ctx: EpisodeContext, admissible: list[Action]) -> tuple[Action, str]:
        if ctx.verified:
            return _terminal(admissible, ctx), "STOP_VERIFIED"
        if ctx.n_by_mode.get("EXPLOIT", 0) > ctx.n_by_mode.get("VERIFY", 0):
            a = _pick(admissible, Mode.VERIFY, Tier.T1)
            if a:
                return a, "ORACLE_VERIFY"
        for tier in (Tier.T2, Tier.T1, Tier.T0):
            a = _pick(admissible, Mode.EXPLOIT, tier, self._truth)
            if a:
                return a, "ORACLE_EXPLOIT"
        return _terminal(admissible, ctx), "ORACLE_OUT_OF_BUDGET"


ARMS = {
    "A_fixed": FixedArm,
    "B_static": StaticRoutedArm,
    "C_heuristic": HeuristicArm,
    "F_oracle": OracleArm,
}
