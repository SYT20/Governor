"""How much adaptive headroom an allocation problem CONTAINS, before you build
a controller for it.

Two families were rejected in this project after their ceilings were measured at
+0.046 and a non-significant +0.086. Both were designed by intuition and
measured afterwards. This module derives the ceiling instead, so a candidate
configuration can be screened in microseconds rather than in a day of API quota.

THE LAW. An episode has `n` items; the budget affords `k` upgrades; item i has
realised gain g_i from the deep budget. A clairvoyant allocator takes the k
largest gains. Any allocator that cannot see gains -- greedy, a fixed schedule,
a random subset -- takes k gains at random. So per-item utility differs by

    ceiling(n, k) = ( E[sum of the top k gains]  -  k * E[g] ) / n

For BINARY gains (g in {0,1}, P(g=1) = p), the number of useful items is
X ~ Binomial(n, p) and the clairvoyant captures min(k, X) of them:

    ceiling(n, k, p) = ( E[min(k, X)] - k*p ) / n                        (exact)

Two consequences follow immediately, and they are the two things the earlier
designs got wrong.

1. THE CEILING IS MAXIMISED AT k ~ n*p AND GROWS WITH n TOWARD p*(1-p).
   Using E[min(k,X)] = np - E[(X-k)+] and the half-normal approximation
   E[(X-k)+] ~ 0.399*sigma at k = np, with sigma = sqrt(n p (1-p)):

       ceiling ~ p(1-p) - 0.399 * sqrt(p(1-p)/n)

   The first term is the ideal; the second is a finite-episode penalty that
   decays like 1/sqrt(n). At p = 0.5 the ideal is 0.25, but n = 4 gives only
   0.156 -- a 38% loss purely from having four items. SHORT EPISODES CANNOT
   CONTAIN MUCH HEADROOM, whatever else is true.

2. IT VANISHES AT BOTH ENDS OF p. p(1-p) -> 0 as p -> 0 (nothing benefits) and
   as p -> 1 (everything benefits, so which items you pick stops mattering).
   A task where the deep budget almost always helps is as useless for measuring
   allocation as one where it never does.

This is the IDEAL ceiling: what the problem contains. What a real environment
DELIVERS is lower, because a hard budget must reserve the worst case and the
engine may use far less. `realisation_ratio` measures that gap, and it is the
quantity that killed Phase 4 -- an ideal ceiling of 0.154 delivered 0.046.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb, sqrt

import numpy as np


def binom_pmf(n: int, p: float) -> np.ndarray:
    k = np.arange(n + 1)
    return np.array([comb(n, int(i)) * p ** int(i) * (1 - p) ** (n - int(i))
                     for i in k], float)


def ceiling_binary(n: int, k: int, p: float) -> float:
    """Exact ideal ceiling for binary gains. No simulation, no approximation."""
    if k <= 0 or k >= n:
        return 0.0 if k <= 0 else max(0.0, 0.0)
    x = np.arange(n + 1)
    pmf = binom_pmf(n, p)
    return float((np.minimum(k, x) @ pmf - k * p) / n)


def ceiling_asymptote(p: float) -> float:
    """The ceiling an infinitely long episode could contain: p*(1-p)."""
    return p * (1.0 - p)


def ceiling_approx(n: int, p: float) -> float:
    """p(1-p) minus the finite-episode penalty, at the optimal k."""
    return p * (1 - p) - 0.3989 * sqrt(p * (1 - p) / n)


def best_k(n: int, p: float) -> tuple[int, float]:
    vals = [(k, ceiling_binary(n, k, p)) for k in range(1, n)]
    return max(vals, key=lambda t: t[1])


def ceiling_general(gains: np.ndarray, n: int, k: int, trials: int = 20000,
                    seed: int = 0) -> float:
    """Ideal ceiling for an ARBITRARY empirical gain distribution.

    Draws episodes by sampling n gains with replacement from the observed pool,
    which is the right null: it holds the marginal distribution fixed and
    assumes nothing about within-episode structure.
    """
    rng = np.random.default_rng(seed)
    g = np.asarray(gains, float)
    if k <= 0 or k >= n:
        return 0.0
    draws = rng.choice(g, size=(trials, n), replace=True)
    top = np.sort(draws, axis=1)[:, -k:].sum(axis=1)
    return float((top.mean() - k * g.mean()) / n)


@dataclass(slots=True)
class Design:
    n_items: int
    k_upgrades: int
    p_useful: float
    ideal_ceiling: float
    asymptote: float
    finite_episode_loss: float

    def __str__(self) -> str:
        return (f"n={self.n_items} k={self.k_upgrades} p={self.p_useful:.3f} "
                f"ideal={self.ideal_ceiling:+.4f} "
                f"(asymptote {self.asymptote:+.4f}, "
                f"finite-episode loss {self.finite_episode_loss:.1%})")


def design(n: int, p: float, k: int | None = None) -> Design:
    kk = k if k is not None else best_k(n, p)[0]
    c = ceiling_binary(n, kk, p)
    a = ceiling_asymptote(p)
    return Design(n, kk, p, c, a, 1.0 - c / a if a > 0 else 0.0)


def n_for_target(p: float, target: float, n_max: int = 200) -> int | None:
    """Smallest episode length whose IDEAL ceiling reaches `target`.

    Answers the design question directly: given how often the deep budget helps,
    how many items must compete for the budget before adaptive allocation can
    possibly be worth more than `target`?
    """
    if target >= ceiling_asymptote(p):
        return None                     # unreachable at ANY episode length
    for n in range(2, n_max + 1):
        if best_k(n, p)[1] >= target:
            return n
    return None


def realisation_ratio(measured: float, ideal: float) -> float:
    """Fraction of the available headroom an environment actually delivers.

    Below ~0.5 the environment's own mechanics -- worst-case reservation,
    infeasible positions -- are destroying more than the controller could ever
    recover, and the environment should be fixed before any controller is built.
    """
    return measured / ideal if ideal > 1e-12 else float("nan")
