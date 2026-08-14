"""Empirical cost and latency profiles.

Decision Record section F: every action class carries a machine-readable profile of
what it actually costs, fitted from logged episodes rather than authored by hand.

The admissibility filter consumes the *upper* quantile (section H.1), so this module
exposes p90 as the `ucb` of a fitted Estimate. Thin cells shrink toward the pooled
per-mode profile rather than reporting a confident number from three observations.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from governor.accounting.meter import DIMENSIONS
from governor.core.estimate import Estimate


def _quantile(sorted_xs: list[float], q: float) -> float:
    """Linear-interpolated quantile. Empty -> 0.0."""
    if not sorted_xs:
        return 0.0
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = q * (len(sorted_xs) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_xs[int(pos)]
    frac = pos - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


@dataclass(slots=True)
class CostProfile:
    """Observed cost distribution per (action_class, dimension).

    `data_version` is the corpus tag stamped onto every Estimate this profile emits,
    so a decision record can always be traced back to the data that produced it.
    """

    data_version: str = "unfitted"
    min_observations: int = 8
    frozen: bool = False
    _obs: dict[tuple[str, str], list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _pooled: dict[tuple[str, str], list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _table: dict[tuple[str, str], Estimate] = field(default_factory=dict)

    # -- fitting ---------------------------------------------------------------

    def observe(self, action_class: str, costs: dict[str, float]) -> None:
        """Record one realised cost vector.

        Refuses once frozen. Decision Record section L: models are frozen before
        evaluation, because a profile that keeps learning while the arms run makes
        the comparison uninterpretable -- later episodes would be scored by a
        different model than earlier ones.
        """
        if self.frozen:
            raise RuntimeError(
                "cost profile is frozen; observing during evaluation would make "
                "arm comparisons uninterpretable (see Decision Record section L)"
            )
        mode = action_class.split("@")[0]
        for dim, val in costs.items():
            if dim not in DIMENSIONS:
                raise KeyError(f"unknown dimension {dim!r}")
            self._obs[(action_class, dim)].append(val)
            self._pooled[(mode, dim)].append(val)

    def n(self, action_class: str, dim: str) -> int:
        return len(self._obs[(action_class, dim)])

    def freeze(self, data_version: str) -> None:
        """Precompute every quantile and lock the profile.

        Sorting the observation history on each lookup made scoring O(n log n) per
        candidate per dimension, which dominated the whole run. Quantiles are fixed
        once the corpus is, so they belong in a table.
        """
        self.data_version = data_version
        self._table = {}
        for key in set(self._obs) | {
            (ac, d) for (ac, d) in self._obs
        }:
            self._table[key] = self._compute(key[0], key[1])
        self.frozen = True

    def _compute(self, action_class: str, dim: str) -> Estimate:
        xs = sorted(self._obs[(action_class, dim)])
        pooled = sorted(self._pooled[(action_class.split("@")[0], dim)])
        source = xs if len(xs) >= self.min_observations else pooled or xs

        if not source:
            # Nothing observed anywhere. Return a wide, obviously-unfitted estimate
            # so the cold-start rule fires rather than the policy trusting a zero.
            return Estimate.fitted(
                0.0,
                ci=(0.0, float("inf")),
                model_id="cost_profile:empty",
                data_version=self.data_version,
                n_effective=0,
                unit=dim,
            )

        shrunk = len(xs) < self.min_observations
        return Estimate.fitted(
            _quantile(source, 0.50),
            ci=(_quantile(source, 0.10), _quantile(source, 0.90)),
            model_id=f"cost_profile{':pooled' if shrunk else ''}",
            data_version=self.data_version,
            n_effective=len(source),
            unit=dim,
        )

    # -- querying --------------------------------------------------------------

    def estimate(self, action_class: str, dim: str) -> Estimate:
        """Cost estimate with p50 as the value and (p10, p90) as the interval.

        The policy scores on `.value` and checks admissibility on `.ucb`, which is
        the p90. That asymmetry is deliberate: plan on the median, budget for the
        tail.
        """
        key = (action_class, dim)
        if self.frozen:
            hit = self._table.get(key)
            return hit if hit is not None else self._compute(*key)
        return self._compute(*key)

    def vector(self, action_class: str) -> dict[str, Estimate]:
        """All dimensions at once, for the admissibility filter."""
        return {d: self.estimate(action_class, d) for d in DIMENSIONS}

    def report(self) -> list[dict[str, object]]:
        classes = sorted({k[0] for k in self._obs})
        rows = []
        for c in classes:
            row: dict[str, object] = {"action_class": c, "n": self.n(c, "tokens")}
            for d in ("tokens", "cost", "wall_s"):
                e = self.estimate(c, d)
                row[f"{d}_p50"] = round(e.value, 4)
                row[f"{d}_p90"] = round(e.ucb, 4)
            rows.append(row)
        return rows


@dataclass(slots=True)
class Reserve:
    """State-dependent cost of terminating safely (Decision Record section H.1).

    Rev 1 used a fixed reserve. That over-reserves once the patch is already
    verified, because the only remaining obligation is to report.
    """

    profile: CostProfile
    verify_class: str = "VERIFY@T0"
    stop_class: str = "STOP_VERIFIED@T0"

    def required(self, *, verified: bool) -> dict[str, float]:
        stop = {d: self.profile.estimate(self.stop_class, d).ucb for d in DIMENSIONS}
        if verified:
            return stop
        verify = {d: self.profile.estimate(self.verify_class, d).ucb for d in DIMENSIONS}
        return {d: verify[d] + stop[d] for d in DIMENSIONS}
