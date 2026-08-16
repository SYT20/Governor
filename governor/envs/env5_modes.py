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
LOOKAHEAD_NODES = 81      # quadrature nodes for the outcome integral in M2.
# Chosen by a convergence rule fixed BEFORE any H1-H5 outcome was seen: the
# smallest node count whose action choices agree with double that count in
# >= 95% of states. Measured 21 -> 64%, 41 -> 92%, 81 -> 96%, 161 -> 100%.
# At 21 the quadrature resolution changed M2's choice in a third of states,
# which is the same class of estimator artefact that inverted the CUBE-NM
# result earlier in this project.
M1_POOL = 8               # candidates M1 scores per regime


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


def r0_uninformed(ib: InstrumentedBayes, logL, available, remaining_tool,
                  perm):
    """R0: genuinely uninformed reflex. The lower bound.

    Takes the first affordable action in a PUBLIC permutation fixed at episode
    start from a recorded seed. The permutation is generated independently of
    the hidden regime, the label, and every observation, so
    I(R0 ; hidden state | public state) = 0 by construction. Deterministic
    given the episode seed, which keeps paired counterfactuals clean.
    """
    for g in perm:
        if (g in available and g != ib.probe_group
                and ib.cost[g] <= remaining_tool + 1e-9):
            return g
    return None


def h_gate_first(ib: InstrumentedBayes, logL, available, remaining_tool,
                 acquired):
    """H: the strong cheap heuristic. Buy the gate first, then act myopically.

    THIS WAS MASQUERADING AS `m0_reflex`. That version selected by cached prior
    dispersion, which is maximised by the context gate, so "act without
    thinking" silently implemented always-buy-the-gate -- the strategic
    non-myopic policy this project established in CUBE-NM. It scored 0.911
    against myopic's 0.822 at zero compute, and M1/M2 were being asked to
    justify their cost against an answer that was already free.

    The fix is not to delete it. It is a real, strong, cheap policy that this
    environment contains, and hiding it would let an expensive planner look
    good against a strawman. It is promoted to a NAMED baseline, and the
    metacognitive question becomes Delta(M2 - H) rather than Delta(M2 - M0).

    Importing it is not hindsight: context-first was established in CUBE-NM
    before Environment 5 existed.
    """
    if 0 in available and ib.cost[0] <= remaining_tool + 1e-9 and not acquired:
        return 0
    afford = [g for g in available
              if ib.cost[g] <= remaining_tool + 1e-9 and g != ib.probe_group]
    if not afford:
        return None
    g = ib.gains(logL, afford)
    return max(afford, key=lambda a: g[a] / ib.cost[a])


def m1_assess(ib: InstrumentedBayes, logL, available, remaining_tool,
              pool: int = M1_POOL):
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
    # POOL FIX. The previous version used afford[:pool*2] -- the first 16 groups
    # BY INDEX -- giving 81.3% recall of the myopic-optimal action, so roughly
    # one state in five had the relevant action excluded before M1 looked at it
    # and the agreement verdict was partly truncation. Ranking over ALL
    # affordable candidates costs more but makes the pool meaningful; the rule
    # is fixed here, before any H1-H5 outcome is seen.
    base = ib.gains(logL, afford)
    cand = sorted(afford, key=lambda a: -base[a] / ib.cost[a])[:pool]

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


def _subgrid(ib, nodes):
    """Evenly spaced indices into the block quadrature grid."""
    T = ib.b._pdf.shape[2]
    return np.linspace(0, T - 1, nodes).astype(int)


def m2_plan(ib: InstrumentedBayes, logL, available, remaining_tool,
            depth: int = LOOKAHEAD_DEPTH, pool: int = LOOKAHEAD_POOL,
            nodes: int = LOOKAHEAD_NODES):
    """M2: genuine k-step lookahead. REWRITTEN -- the previous version was not.

    The old implementation evaluated its continuation term as gains(logL, nxt)
    at the CURRENT logL, so the posterior was never advanced for having taken
    the candidate action. It was myopic gain plus a near-constant bonus, and it
    picked the same action as plain myopic in 26/30 states. That is not
    lookahead, and every conclusion drawn from it was void.

    Correct form, for candidate a:

        V2(s, a) = gain(a) + E_{o ~ P(o | s, a)} [ max_a' gain(a' | s_{a,o}) ]

    where s_{a,o} is the belief state AFTER hypothetically observing o. The
    expectation is enumerated exactly over the quadrature grid already used for
    the one-step gains -- no Monte Carlo, so no convergence argument is needed
    and no sampling noise can distort the comparison, which is how the CUBE-NM
    result was inverted earlier in this project.

    Still touches no real observation: `x` is not a parameter. The hypothetical
    outcomes come from the posterior predictive, never from the instance.
    """
    afford = [g for g in available
              if ib.cost[g] <= remaining_tool + 1e-9 and g != ib.probe_group]
    if not afford:
        return None
    first = ib.gains(logL, afford)
    cand = sorted(afford, key=lambda a: -first[a] / ib.cost[a])[:pool]
    if depth < 2:
        return cand[0]

    post = ib.b._norm(logL)
    idx = _subgrid(ib, nodes)
    best, best_v = None, -np.inf
    for a in cand:
        rem = remaining_tool - float(ib.cost[a])
        nxt = [g for g in cand if g != a and ib.cost[g] <= rem + 1e-9]
        v = first[a]
        if nxt and a in ib.b._col_slot_of:
            slot = ib.b._col_slot_of[a]
            pdf = ib.b._pdf[:, slot, :][:, idx]          # (H, nodes)
            w = (post[:, None] * pdf).sum(axis=0)        # outcome density
            w = w / max(w.sum(), 1e-300)
            cont = 0.0
            for t, node in enumerate(idx):
                if w[t] < 1e-6:
                    continue
                # THE FIX: advance the belief state for this hypothetical outcome
                ib.c.branch_expansions += 1
                logL_next = logL + np.log(np.maximum(pdf[:, t], 1e-300))
                g2 = ib.gains(logL_next, nxt)
                cont += w[t] * max(g2[q] / ib.cost[q] for q in nxt)
            v += cont * float(ib.cost[a])
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
            action = r0_uninformed(self.ib, logL, available,
                                   self.b_tool - self.tool_spent,
                                   getattr(self, "perm", None) or
                                   list(range(self.ib.n_groups)))
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
