"""Environment 5 — instrumented reasoning modes. Implements the rev5 spec.

Design lives in PREREGISTRATION-env5-two-currency.md rev5. This module does not
decide anything; it implements what is fixed there.

    M0  direct action            0 compute   act on the current posterior
    M1  diagnostic deliberation  c1 compute  "what regime am I in?"  (probe)
    M2  strategic planning       c2 compute  "which acquisition?"    (k=2 lookahead)

TWO RESOURCES, structurally separate:

    B_tool     acquisitions          -> Envelope.tool_calls
    B_compute  deliberation          -> Envelope.cost

`Envelope.cost` carries compute deliberately, leaving `Envelope.tokens` free to
mean actual LLM tokens later. Compute units are instrumented primitive
operations; the mapping from them to tokens is a question to be measured, not an
identity. Rev3 of the spec claimed that identity and it was wrong.

FOUR PRIMITIVE COUNTERS, frozen before any policy evaluation:

    likelihood_evals    calls to loglik_cols
    posterior_updates   normalisations of the H-vector
    candidate_evals     per-candidate expected-entropy evaluations
    branch_expansions   lookahead nodes expanded

Which one becomes `B_compute` is decided by the preregistered selection rule
(dominance + coverage + linearity), not here and not by preference.

THE INVARIANT THAT MATTERS. `M2.plan()` must not acquire anything: it reasons
over the posterior predictive, never over real observations. If it read actual
feature values while planning, its advantage would be extra information rather
than better planning and the resource comparison would be void. Asserted at
runtime by `ModeRunner`, not claimed in prose -- Env 4a's observability leak
entered through exactly that kind of unasserted claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from governor.envs.gated_family import N_LABELS

LOOKAHEAD_DEPTH = 2       # PREREGISTERED, frozen. Variable depth is Env 6.
LOOKAHEAD_POOL = 6        # candidates retained per level, by myopic gain


@dataclass(slots=True)
class Counters:
    """The four preregistered primitives. No others are counted."""

    likelihood_evals: int = 0
    posterior_updates: int = 0
    candidate_evals: int = 0
    branch_expansions: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "likelihood_evals": self.likelihood_evals,
            "posterior_updates": self.posterior_updates,
            "candidate_evals": self.candidate_evals,
            "branch_expansions": self.branch_expansions,
        }

    def __sub__(self, other: "Counters") -> "Counters":
        return Counters(**{k: getattr(self, k) - getattr(other, k)
                           for k in self.as_dict()})

    def copy(self) -> "Counters":
        return Counters(**self.as_dict())


class InstrumentedBayes:
    """Wraps a ProbeBayes and counts every primitive it performs.

    Composition rather than subclassing: the underlying scorer stays exactly the
    object that passed Env 4a's G1/G2a/G2b/G2c, so instrumentation cannot change
    the inference it was validated on.
    """

    def __init__(self, bayes) -> None:  # noqa: ANN001
        self.b = bayes
        self.c = Counters()

    # -- counted primitives ----------------------------------------------------

    def loglik_cols(self, x, cols):
        self.c.likelihood_evals += 1
        return self.b.loglik_cols(x, cols)

    def label_posterior(self, logL):
        self.c.posterior_updates += 1
        return self.b.label_posterior(logL)

    def regime_posterior(self, logL):
        self.c.posterior_updates += 1
        return self.b.regime_posterior(logL)

    def gains(self, logL, available, target="label"):
        self.c.candidate_evals += len(available)
        return self.b.gains(logL, available, target)

    def gains_under_regime(self, logL, available, r):
        self.c.candidate_evals += len(available)
        self.c.posterior_updates += 1
        return self.b.gains_under_regime(logL, available, r)

    # -- passthrough -----------------------------------------------------------

    def __getattr__(self, name):
        return getattr(self.b, name)

    def reset_counters(self) -> None:
        self.c = Counters()


# -- the three modes ------------------------------------------------------------
#
# REDESIGNED after the construction gate measured C(M1):C(M2) = 1:129 with no
# shared primitive. Two changes, both forced by measurement rather than chosen:
#
# 1. M0 IS A REFLEX, NOT THE MYOPIC OPTIMUM. The spec asserted C(M0)=0 while the
#    implementation computed exact myopic gains over all 51 candidates. Those are
#    inconsistent: choosing the Bayes-optimal acquisition IS deliberation. Acting
#    without deliberation means using a heuristic, so M0 now selects by prior
#    feature dispersion with zero posterior computation. C(M0)=0 is now true
#    rather than asserted.
#
# 2. M1 IS INTERNAL COMPUTATION, NOT AN OBSERVATION. The old M1 read the probe
#    scalar -- information acquisition wearing deliberation's label. The new M1
#    obtains NO new observation. It asks, over evidence already held: "would
#    knowing the task regime change what I do?" by computing the myopic argmax
#    under each regime hypothesis and measuring agreement.
#
# The review's test for whether M1 is meaningful: remove it, and the system loses
# the information needed to choose between M0 and M2. Agreement across regimes
# says planning cannot exploit regime uncertainty; disagreement says it might.
# Nothing else in the state carries that.


def m0_reflex(ib: InstrumentedBayes, logL, available, remaining_tool):
    """M0: act with no deliberation at all.

    A reflex, not the myopic optimum. Picks the unobserved feature with the
    largest prior dispersion across hypotheses -- a fixed property of the
    model, requiring no posterior evaluation, so this mode is genuinely free.
    """
    afford = [g for g in available
              if ib.cost[g] <= remaining_tool + 1e-9 and g != ib.probe_group]
    if not afford:
        return None
    return max(afford, key=lambda a: ib.b._prior_spread[a] / ib.cost[a])


def m1_assess(ib: InstrumentedBayes, logL, available, remaining_tool,
              pool: int = 8):
    """M1: metacognitive assessment. NO new observation is obtained.

    Computes, for each regime hypothesis weighted by P(regime | evidence), the
    myopic argmax restricted to a small candidate pool, and returns the
    probability mass on the modal recommendation. Low agreement means the right
    action depends on which task this is, so planning may pay; high agreement
    means it does not.

    Returns (assessment, recommended_action). The assessment is the quantity a
    Governor would condition on; it is unavailable without this computation.
    """
    afford = [g for g in available
              if ib.cost[g] <= remaining_tool + 1e-9 and g != ib.probe_group]
    if not afford:
        return None, None
    pr = ib.regime_posterior(logL)
    base = ib.gains(logL, afford[:pool * 2])
    cand = sorted(afford[:pool * 2], key=lambda a: -base[a] / ib.cost[a])[:pool]

    votes: dict[int, float] = {}
    for r, w in enumerate(pr):
        if w < 0.02:                       # negligible regimes are not scored
            continue
        gr = ib.gains_under_regime(logL, cand, r)
        a = max(cand, key=lambda q: gr[q] / ib.cost[q])
        votes[a] = votes.get(a, 0.0) + float(w)
    if not votes:
        return None, cand[0]
    top = max(votes, key=lambda a: votes[a])
    return votes[top], top


def m2_plan(ib: InstrumentedBayes, logL, available, remaining_tool,
            depth: int = LOOKAHEAD_DEPTH, pool: int = LOOKAHEAD_POOL):
    """M2: strategic planning -- k-step lookahead over acquisition sequences.

    Reasons entirely over the posterior predictive. `x` is not a parameter, so
    real observations are structurally unavailable to it; ModeRunner asserts
    tool_calls is unchanged across the call as a second, independent guard.
    """
    afford = [g for g in available
              if ib.cost[g] <= remaining_tool + 1e-9 and g != ib.probe_group]
    if not afford:
        return None
    first = ib.gains(logL, afford)
    cand = sorted(afford, key=lambda a: -first[a] / ib.cost[a])[:pool]

    best, best_v = None, -np.inf
    for a in cand:
        ib.c.branch_expansions += 1
        rem = remaining_tool - float(ib.cost[a])
        v = first[a]
        if depth > 1 and rem > 0:
            nxt = [g for g in cand if g != a and ib.cost[g] <= rem + 1e-9]
            if nxt:
                sub = ib.gains(logL, nxt)
                ib.c.branch_expansions += len(nxt)
                v += max(sub[g] / ib.cost[g] for g in nxt) * ib.cost[a]
        if v / ib.cost[a] > best_v:
            best_v, best = v / ib.cost[a], a
    return best


@dataclass(slots=True)
class ModeRunner:
    """Executes modes, charges the right resource, and enforces the invariants."""

    ib: InstrumentedBayes
    b_tool: float
    b_compute: float
    tool_spent: float = 0.0
    compute_spent: float = 0.0
    trace: list[dict] = field(default_factory=list)

    def _compute_of(self, delta: Counters, unit: str) -> float:
        return float(delta.as_dict()[unit])

    def invoke(self, mode: str, logL, available, x, unit: str):
        """Run one mode. Returns (new_logL, chosen_action, delta_counters).

        `unit` names the primitive currently serving as B_compute -- passed in
        rather than hardcoded, because the preregistered selection rule fixes it
        only after profiling.
        """
        before = self.ib.c.copy()
        tool_before = self.tool_spent
        action, new_logL = None, logL

        assess = None
        if mode == "M0":
            action = m0_reflex(self.ib, logL, available,
                               self.b_tool - self.tool_spent)
        elif mode == "M1":
            assess, action = m1_assess(self.ib, logL, available,
                                       self.b_tool - self.tool_spent)
        elif mode == "M2":
            action = m2_plan(self.ib, logL, available,
                             self.b_tool - self.tool_spent)
        else:
            raise ValueError(f"unknown mode {mode}")

        delta = self.ib.c - before
        # PREREGISTERED INVARIANT: deliberation consumes no tool budget.
        assert self.tool_spent == tool_before, (
            f"{mode} changed tool_calls during deliberation -- "
            "an acquisition leaked into a reasoning mode")
        cost = 0.0 if mode == "M0" else self._compute_of(delta, unit)
        self.compute_spent += cost
        self.trace.append({"mode": mode, "compute": cost,
                           "assessment": assess, **delta.as_dict()})
        return new_logL, action, delta

    def acquire(self, logL, action, x):
        """Execute an acquisition. Charges tool_calls, never compute."""
        before = self.ib.c.copy()
        self.tool_spent += float(self.ib.cost[action])
        new_logL = logL + self.ib.b.loglik_cols(x, self.ib.group_cols[action])
        # deliberately uses the RAW scorer: an execution is not deliberation and
        # must not inflate the compute meter.
        assert (self.ib.c - before).as_dict() == Counters().as_dict()
        return new_logL

    def can_afford_compute(self, cost: float) -> bool:
        return self.compute_spent + cost <= self.b_compute + 1e-9

    def can_afford_tool(self, cost: float) -> bool:
        return self.tool_spent + cost <= self.b_tool + 1e-9
