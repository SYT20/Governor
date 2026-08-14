"""SynthBug: a fast synthetic debugging environment with known ground truth.

Why this exists
---------------
The Governor policy layer needs no GPU and no Docker; only the *executor* does.
SynthBug is a stand-in executor that preserves the decision structure of agentic
debugging -- uncertainty over a root cause, costly evidence, costly repair attempts,
noisy verification, a hard budget -- while running thousands of episodes per second
on a laptop.

It buys three things that a real benchmark cannot:

1. **Ground truth.** The true root cause is known, so a hindsight oracle is exact
   rather than approximated by branching.
2. **Known channel parameters.** The evidence channel's true (alpha, beta) are set by
   config, so the corrected two-parameter Bayes update of section G.3 can be tested
   against arithmetic truth, and the *fitted* channel reliability can be checked for
   convergence to the values that generated the data.
3. **Volume.** The value model needs thousands of labelled checkpoints. Here they
   cost microseconds.

What it does NOT buy: any evidence about real code-fixing skill. Results here
validate the *machinery* -- accounting, belief update, calibration, policy, metrics --
not task performance. Every claim derived from SynthBug must say so.

Environment semantics
---------------------
One hidden root cause `true_cause` among `n_hypotheses`. The agent may:

  EXPLORE(tier, target)  draw binary evidence about whether `target` is the cause.
                         P(o=1 | target is cause)     = alpha[tier]   (sensitivity)
                         P(o=1 | target is not cause) = beta[tier]    (false alarm)
  EXPLOIT(tier, target)  attempt a repair aimed at `target`.
                         correct target -> CORRECT with prob p_fix[tier], else WRONG.
  VERIFY(tier)           noisy test of the current patch; accuracy = verify_acc[tier].
  STOP_*                 terminate.

The episode succeeds iff the agent submits (STOP_VERIFIED or STOP_UNVERIFIED) while
the patch is genuinely CORRECT. STOP_FAILURE never succeeds.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable


class Mode(StrEnum):
    EXPLORE = "EXPLORE"
    EXPLOIT = "EXPLOIT"
    VERIFY = "VERIFY"
    STOP_VERIFIED = "STOP_VERIFIED"
    STOP_UNVERIFIED = "STOP_UNVERIFIED"
    STOP_FAILURE = "STOP_FAILURE"

    @property
    def is_terminal(self) -> bool:
        return self.name.startswith("STOP")


class Tier(StrEnum):
    T0 = "T0"
    T1 = "T1"
    T2 = "T2"


class PatchState(StrEnum):
    NONE = "NONE"
    WRONG = "WRONG"
    CORRECT = "CORRECT"


@dataclass(frozen=True, slots=True)
class Action:
    mode: Mode
    tier: Tier = Tier.T1
    target: int | None = None

    def __str__(self) -> str:
        t = "" if self.target is None else f"->h{self.target}"
        return f"{self.mode}@{self.tier}{t}"

    @property
    def action_class(self) -> str:
        """Key into the cost/latency profile tables."""
        return f"{self.mode}@{self.tier}"


@dataclass(frozen=True, slots=True)
class Observation:
    """What the agent learns from one action. Never exposes hidden state."""

    action: Action
    kind: str  # "evidence" | "repair" | "test" | "terminal"
    value: int | None = None  # binary for evidence/test; None for repair
    target: int | None = None
    note: str = ""


@dataclass(frozen=True, slots=True)
class SynthConfig:
    """Generative parameters. Everything the agent must not see lives here."""

    n_hypotheses: int = 4

    # Evidence channel per tier: (alpha=sensitivity, beta=false-alarm rate).
    # Higher tiers are more discriminative *and* more expensive. Note beta is NOT
    # 1-alpha -- that asymmetry is exactly what the corrected likelihood handles.
    alpha: dict[Tier, float] = field(
        default_factory=lambda: {Tier.T0: 0.62, Tier.T1: 0.78, Tier.T2: 0.91}
    )
    beta: dict[Tier, float] = field(
        default_factory=lambda: {Tier.T0: 0.34, Tier.T1: 0.20, Tier.T2: 0.08}
    )

    # Probability a repair aimed at the *correct* cause actually lands.
    p_fix: dict[Tier, float] = field(
        default_factory=lambda: {Tier.T0: 0.45, Tier.T1: 0.70, Tier.T2: 0.88}
    )
    # Probability a repair aimed at the wrong cause silently breaks something else.
    p_regress: float = 0.15

    # Probability VERIFY reports the patch's true status.
    verify_acc: dict[Tier, float] = field(
        default_factory=lambda: {Tier.T0: 0.80, Tier.T1: 0.93, Tier.T2: 0.99}
    )

    # Cost model: lognormal work units per (mode, tier), mapped to dimensions.
    work_mu: dict[str, float] = field(
        default_factory=lambda: {
            "EXPLORE@T0": math.log(1.0),
            "EXPLORE@T1": math.log(2.2),
            "EXPLORE@T2": math.log(5.0),
            "EXPLOIT@T0": math.log(2.0),
            "EXPLOIT@T1": math.log(4.5),
            "EXPLOIT@T2": math.log(9.0),
            "VERIFY@T0": math.log(1.5),
            "VERIFY@T1": math.log(3.0),
            "VERIFY@T2": math.log(6.5),
        }
    )
    work_sigma: float = 0.45
    tokens_per_work: float = 900.0
    cost_per_work: float = 0.004
    wall_s_per_work: float = 1.8

    # Task difficulty: prior mass concentration. 1.0 = uniform prior over causes.
    prior_concentration: float = 1.0


@dataclass(slots=True)
class SynthBug:
    """One episode. Stateful, seeded, and cheap.

    The agent interacts only through `step`. `true_cause` and `patch` are readable
    for *scoring and oracle construction only* -- the policy must never touch them.
    """

    config: SynthConfig
    seed: int
    true_cause: int = field(init=False)
    patch: PatchState = field(init=False, default=PatchState.NONE)
    terminated: bool = field(init=False, default=False)
    terminal_mode: Mode | None = field(init=False, default=None)
    n_steps: int = field(init=False, default=0)
    _rng: random.Random = field(init=False, repr=False)
    _prior: list[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        k = self.config.n_hypotheses
        if self.config.prior_concentration == 1.0:
            self._prior = [1.0 / k] * k
        else:
            w = [self._rng.gammavariate(self.config.prior_concentration, 1.0) for _ in range(k)]
            s = sum(w)
            self._prior = [x / s for x in w]
        # Sample the true cause from the prior the agent is entitled to know.
        r, acc = self._rng.random(), 0.0
        self.true_cause = k - 1
        for i, p in enumerate(self._prior):
            acc += p
            if r <= acc:
                self.true_cause = i
                break

    # -- public, agent-visible -------------------------------------------------

    @property
    def prior(self) -> list[float]:
        """The prior over causes. The agent may legitimately start here."""
        return list(self._prior)

    def cost_of(self, action: Action) -> dict[str, float]:
        """Sample the realised cost of an action. Consumes randomness."""
        if action.mode.is_terminal:
            return {"tokens": 120.0, "cost": 0.0005, "wall_s": 0.2, "tool_calls": 0.0}
        mu = self.config.work_mu[action.action_class]
        work = math.exp(self._rng.gauss(mu, self.config.work_sigma))
        c = self.config
        return {
            "tokens": work * c.tokens_per_work,
            "cost": work * c.cost_per_work,
            "wall_s": work * c.wall_s_per_work,
            "tool_calls": 1.0,
        }

    def step(self, action: Action) -> Observation:
        """Apply an action and return what the agent is allowed to see."""
        if self.terminated:
            raise RuntimeError("episode already terminated")
        self.n_steps += 1
        c = self.config

        if action.mode.is_terminal:
            self.terminated = True
            self.terminal_mode = action.mode
            return Observation(action, kind="terminal", note=str(action.mode))

        if action.mode is Mode.EXPLORE:
            if action.target is None:
                raise ValueError("EXPLORE requires a target hypothesis")
            p = c.alpha[action.tier] if action.target == self.true_cause else c.beta[action.tier]
            o = 1 if self._rng.random() < p else 0
            return Observation(action, kind="evidence", value=o, target=action.target)

        if action.mode is Mode.EXPLOIT:
            if action.target is None:
                raise ValueError("EXPLOIT requires a target hypothesis")
            if action.target == self.true_cause:
                if self._rng.random() < c.p_fix[action.tier]:
                    self.patch = PatchState.CORRECT
                else:
                    # Aimed right, missed. Leaves whatever was there before.
                    self.patch = self.patch if self.patch is PatchState.CORRECT else PatchState.WRONG
            else:
                if self._rng.random() < c.p_regress or self.patch is not PatchState.CORRECT:
                    self.patch = PatchState.WRONG
            return Observation(action, kind="repair", target=action.target)

        if action.mode is Mode.VERIFY:
            truth = 1 if self.patch is PatchState.CORRECT else 0
            acc = c.verify_acc[action.tier]
            reported = truth if self._rng.random() < acc else 1 - truth
            return Observation(action, kind="test", value=reported)

        raise ValueError(f"unhandled mode {action.mode}")

    # -- scoring (never visible to the policy) --------------------------------

    def succeeded(self) -> bool:
        """Ground-truth grading, analogous to FAIL_TO_PASS/PASS_TO_PASS."""
        submitted = self.terminal_mode in (Mode.STOP_VERIFIED, Mode.STOP_UNVERIFIED)
        return bool(submitted and self.patch is PatchState.CORRECT)

    def ground_truth(self) -> dict[str, object]:
        return {
            "true_cause": self.true_cause,
            "patch": str(self.patch),
            "terminal": str(self.terminal_mode) if self.terminal_mode else None,
            "succeeded": self.succeeded(),
            "n_steps": self.n_steps,
        }


# -- helpers -------------------------------------------------------------------


def all_actions(n_hypotheses: int, tiers: Iterable[Tier] = tuple(Tier)) -> list[Action]:
    """The complete action space. Used by the deterministic candidate generator
    (section F.3) and by the exhaustive oracle."""
    out: list[Action] = []
    for tier in tiers:
        for h in range(n_hypotheses):
            out.append(Action(Mode.EXPLORE, tier, h))
            out.append(Action(Mode.EXPLOIT, tier, h))
        out.append(Action(Mode.VERIFY, tier))
    for m in (Mode.STOP_VERIFIED, Mode.STOP_UNVERIFIED, Mode.STOP_FAILURE):
        out.append(Action(m, Tier.T0))
    return out


def make_task(seed: int, config: SynthConfig | None = None) -> SynthBug:
    return SynthBug(config=config or SynthConfig(), seed=seed)
