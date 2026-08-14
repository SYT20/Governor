"""Belief state over root-cause hypotheses, and the evidence channel.

Implements the corrected likelihood of Decision Record section G.3.

Revision 1 of the design used a single "hit rate" per channel::

    P(o | H) = hit_rate      if o matched the prediction
               1 - hit_rate  otherwise

That is only valid for a *symmetric* channel, i.e. one where the false-alarm rate
happens to equal one minus the sensitivity. Real channels are not symmetric: a
targeted test can be very likely to fail when the hypothesis is right *and*
moderately likely to fail when it is wrong. One parameter cannot express that.

The corrected form uses two independently estimated parameters per channel::

    alpha = P(o = 1 | H_i true,  prediction for H_i was 1)   "fires when it should"
    beta  = P(o = 1 | H_i true,  prediction for H_i was 0)   "fires when it should not"

which yields the standard diagnostic likelihood ratios LR+ = alpha/beta and
LR- = (1-alpha)/(1-beta). Revision 1's formula is the special case beta = 1 - alpha.

The LLM supplies only the *sign* of the prediction. The magnitude always comes from
here, and is either fitted from logged data or held at a conservative fixed value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

Belief = list[float]

_EPS = 1e-12


def uniform(n: int) -> Belief:
    return [1.0 / n] * n


def normalise(weights: list[float]) -> Belief:
    total = sum(weights)
    if total <= _EPS:
        # Every hypothesis was ruled out. Rather than divide by zero, fall back to
        # uniform: total refutation means we have learned nothing usable, not that
        # the world is empty.
        return uniform(len(weights))
    return [w / total for w in weights]


def entropy(belief: Belief, *, normalised: bool = True) -> float:
    """Shannon entropy in nats; optionally scaled to [0, 1].

    This is the `belief_entropy` feature the value model consumes, and the trigger
    for the contradiction override. Normalising by log(k) keeps it comparable across
    tasks with different hypothesis counts.
    """
    h = -sum(p * math.log(p) for p in belief if p > _EPS)
    if normalised and len(belief) > 1:
        return h / math.log(len(belief))
    return h


@dataclass(frozen=True, slots=True)
class Channel:
    """An evidence source with measured asymmetric reliability.

    Attributes:
        name: Keys the reliability table, e.g. "explore@T1".
        alpha: P(observation = 1 | targeted hypothesis is true).
        beta: P(observation = 1 | targeted hypothesis is false).
        n_alpha, n_beta: Observations behind each parameter. Drives shrinkage and
            the decision of whether this channel is fitted or held at the fallback.
    """

    name: str
    alpha: float
    beta: float
    n_alpha: int = 0
    n_beta: int = 0

    def __post_init__(self) -> None:
        for v, n in ((self.alpha, "alpha"), (self.beta, "beta")):
            if not 0.0 < v < 1.0:
                raise ValueError(f"{n} must be in (0,1), got {v}")
        if self.alpha <= self.beta:
            # A channel whose false-alarm rate meets or exceeds its sensitivity
            # carries no usable signal, and would move belief the wrong way.
            raise ValueError(
                f"channel {self.name!r} is uninformative or inverted: "
                f"alpha={self.alpha} <= beta={self.beta}"
            )

    @property
    def lr_positive(self) -> float:
        """Likelihood ratio for a positive observation: alpha / beta."""
        return self.alpha / self.beta

    @property
    def lr_negative(self) -> float:
        """Likelihood ratio for a negative observation: (1-alpha) / (1-beta)."""
        return (1.0 - self.alpha) / (1.0 - self.beta)

    def likelihoods(self, n_hypotheses: int, target: int, observation: int) -> list[float]:
        """P(observation | H_i) for every hypothesis i.

        The action targeted hypothesis `target`, which is equivalent to predicting
        "1" for that hypothesis and "0" for all others.
        """
        if observation not in (0, 1):
            raise ValueError(f"observation must be binary, got {observation!r}")
        if not 0 <= target < n_hypotheses:
            raise IndexError(f"target {target} out of range for {n_hypotheses}")
        if observation == 1:
            return [self.alpha if i == target else self.beta for i in range(n_hypotheses)]
        return [
            (1.0 - self.alpha) if i == target else (1.0 - self.beta)
            for i in range(n_hypotheses)
        ]


def update_belief(
    belief: Belief, channel: Channel, target: int, observation: int
) -> Belief:
    """One Bayesian update. The only function permitted to write a belief vector."""
    lik = channel.likelihoods(len(belief), target, observation)
    return normalise([p * l for p, l in zip(belief, lik)])


# -- fallback for cold start (Decision D5) --------------------------------------


def conservative_channel(name: str = "fallback", *, cap: float = 3.0) -> Channel:
    """A channel with a bounded likelihood ratio and no fitted parameters.

    This is the v0 that ships before enough resolved episodes exist to estimate
    (alpha, beta). Evidence still moves belief in the right direction, but a single
    observation can never move it by more than `cap`:1, so the policy cannot become
    overconfident on the strength of an unvalidated channel.

    With alpha = cap/(1+cap) and beta = 1/(1+cap), LR+ is exactly `cap`.
    """
    if cap <= 1.0:
        raise ValueError("cap must exceed 1 for the channel to be informative")
    alpha = cap / (1.0 + cap)
    beta = 1.0 / (1.0 + cap)
    return Channel(name=name, alpha=alpha, beta=beta)


# -- fitting channel reliability from logged data -------------------------------


@dataclass(slots=True)
class ChannelEstimator:
    """Beta-Binomial posteriors for a channel's alpha and beta.

    Counts come from resolved episodes only -- those where the true hypothesis is
    ultimately known. That is a real constraint worth stating loudly: if few episodes
    resolve, these posteriors stay too wide to use, and the policy should keep the
    conservative fallback instead.

    Partial pooling (section G.3): `prior_alpha`/`prior_beta` can be set to a global
    channel prior so a thin per-task-class cell shrinks toward the pooled estimate
    rather than toward the uninformative Beta(1,1).
    """

    name: str
    # Beta(a, b) prior for alpha: successes = observation 1 when target was true.
    a_alpha: float = 1.0
    b_alpha: float = 1.0
    # Beta(a, b) prior for beta: successes = observation 1 when target was false.
    a_beta: float = 1.0
    b_beta: float = 1.0

    def observe(self, *, target_was_true: bool, observation: int) -> None:
        if observation not in (0, 1):
            raise ValueError("observation must be binary")
        if target_was_true:
            self.a_alpha += observation
            self.b_alpha += 1 - observation
        else:
            self.a_beta += observation
            self.b_beta += 1 - observation

    @property
    def n_alpha(self) -> int:
        return int(self.a_alpha + self.b_alpha - 2)

    @property
    def n_beta(self) -> int:
        return int(self.a_beta + self.b_beta - 2)

    def mean_alpha(self) -> float:
        return self.a_alpha / (self.a_alpha + self.b_alpha)

    def mean_beta(self) -> float:
        return self.a_beta / (self.a_beta + self.b_beta)

    def ci(self, which: str, mass: float = 0.90) -> tuple[float, float]:
        """Normal approximation to the Beta credible interval.

        Adequate for a gate ("is this tight enough to use?"); not for reporting.
        """
        a, b = (self.a_alpha, self.b_alpha) if which == "alpha" else (self.a_beta, self.b_beta)
        n = a + b
        mean = a / n
        var = (a * b) / (n * n * (n + 1.0))
        z = 1.6449 if abs(mass - 0.90) < 1e-9 else 1.9600
        half = z * math.sqrt(var)
        return (max(0.0, mean - half), min(1.0, mean + half))

    def is_usable(self, *, min_n: int = 30, max_ci_width: float = 0.30) -> bool:
        """Whether to trust this fitted channel over the conservative fallback."""
        if self.n_alpha < min_n or self.n_beta < min_n:
            return False
        wa = self.ci("alpha")[1] - self.ci("alpha")[0]
        wb = self.ci("beta")[1] - self.ci("beta")[0]
        if wa > max_ci_width or wb > max_ci_width:
            return False
        return self.mean_alpha() > self.mean_beta()

    def to_channel(self) -> Channel:
        return Channel(
            name=self.name,
            alpha=self.mean_alpha(),
            beta=self.mean_beta(),
            n_alpha=self.n_alpha,
            n_beta=self.n_beta,
        )

    def channel_or_fallback(self, **gate: object) -> Channel:
        """Fitted channel when the data supports it, conservative one otherwise."""
        if self.is_usable(**gate):  # type: ignore[arg-type]
            return self.to_channel()
        return conservative_channel(name=f"{self.name}:fallback")


@dataclass(slots=True)
class ChannelBook:
    """Reliability table keyed by channel name."""

    estimators: dict[str, ChannelEstimator] = field(default_factory=dict)

    def estimator(self, name: str) -> ChannelEstimator:
        if name not in self.estimators:
            self.estimators[name] = ChannelEstimator(name=name)
        return self.estimators[name]

    def observe(self, name: str, *, target_was_true: bool, observation: int) -> None:
        self.estimator(name).observe(
            target_was_true=target_was_true, observation=observation
        )

    def channel(self, name: str) -> Channel:
        return self.estimator(name).channel_or_fallback()

    def report(self) -> list[dict[str, object]]:
        rows = []
        for name, e in sorted(self.estimators.items()):
            rows.append(
                {
                    "channel": name,
                    "alpha": round(e.mean_alpha(), 4),
                    "beta": round(e.mean_beta(), 4),
                    "n_alpha": e.n_alpha,
                    "n_beta": e.n_beta,
                    "usable": e.is_usable(),
                    "lr_plus": round(e.mean_alpha() / max(e.mean_beta(), _EPS), 3),
                }
            )
        return rows
