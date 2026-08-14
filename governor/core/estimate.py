"""Provenance-carrying numeric estimates.

The load-bearing invariant of Governor (Decision Record section D): every number the
policy consumes must be traceable to a measurement, a fitted model, or an arithmetic
derivation. This module makes that a type error rather than a code-review convention.

The policy never accepts a bare float. It accepts an `Estimate`, and an `Estimate`
cannot be constructed without declaring where its value came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

Source = Literal["measured", "fitted", "derived"]
_VALID_SOURCES = ("measured", "fitted", "derived")


class ProvenanceError(ValueError):
    """Raised when a number reaches the policy without usable provenance."""


@dataclass(frozen=True, slots=True)
class Estimate:
    """A number the policy is allowed to use.

    Attributes:
        value: The point estimate.
        source: How the value was obtained.
            ``measured``  - read from an instrument (token counter, clock, exit code).
            ``fitted``    - produced by a model fit on recorded data.
            ``derived``   - computed arithmetically from other Estimates.
        ci: Optional (low, high) interval. Required for ``fitted`` values, because a
            fitted number without an interval cannot support the cold-start rule.
        model_id: Identifier of the fitting model. Required for ``fitted``.
        data_version: Hash/tag of the corpus the model was fit on. Required for ``fitted``.
        n_effective: Clustering-adjusted sample size behind the estimate (see J.8).
        unit: Free-form unit tag, carried for reporting and sanity checks.
    """

    value: float
    source: Source
    ci: tuple[float, float] | None = None
    model_id: str | None = None
    data_version: str | None = None
    n_effective: int | None = None
    unit: str | None = None
    inputs: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.source not in _VALID_SOURCES:
            raise ProvenanceError(
                f"source must be one of {_VALID_SOURCES}, got {self.source!r}"
            )
        if not isinstance(self.value, (int, float)) or self.value != self.value:
            raise ProvenanceError(f"value must be a real number, got {self.value!r}")

        if self.source == "fitted":
            # A fitted number without an interval cannot drive the cold-start
            # fallback, so the policy would silently trust an unfalsifiable guess.
            if self.ci is None:
                raise ProvenanceError("fitted estimates require a ci=(low, high)")
            if self.model_id is None:
                raise ProvenanceError("fitted estimates require a model_id")
            if self.data_version is None:
                raise ProvenanceError("fitted estimates require a data_version")

        if self.ci is not None:
            lo, hi = self.ci
            if lo > hi:
                raise ProvenanceError(f"ci low {lo} exceeds high {hi}")
            if not (lo <= self.value <= hi):
                raise ProvenanceError(
                    f"value {self.value} outside its own ci {self.ci}"
                )

    # -- constructors ----------------------------------------------------------

    @classmethod
    def measured(cls, value: float, *, unit: str | None = None) -> "Estimate":
        """An instrument reading. No interval: it is what happened."""
        return cls(value=value, source="measured", unit=unit)

    @classmethod
    def fitted(
        cls,
        value: float,
        *,
        ci: tuple[float, float],
        model_id: str,
        data_version: str,
        n_effective: int | None = None,
        unit: str | None = None,
    ) -> "Estimate":
        """A model output. Must carry an interval and its lineage."""
        return cls(
            value=value,
            source="fitted",
            ci=ci,
            model_id=model_id,
            data_version=data_version,
            n_effective=n_effective,
            unit=unit,
        )

    @classmethod
    def derived(
        cls, value: float, *, inputs: tuple[str, ...] = (), unit: str | None = None
    ) -> "Estimate":
        """Arithmetic over other Estimates. ``inputs`` names what it was built from."""
        return cls(value=value, source="derived", inputs=inputs, unit=unit)

    # -- properties ------------------------------------------------------------

    @property
    def ci_width(self) -> float:
        """Width of the credible interval; ``inf`` when there is none."""
        if self.ci is None:
            return float("inf")
        return self.ci[1] - self.ci[0]

    @property
    def lcb(self) -> float:
        """Lower confidence bound. Falls back to the point estimate."""
        return self.ci[0] if self.ci is not None else self.value

    @property
    def ucb(self) -> float:
        """Upper confidence bound. Falls back to the point estimate.

        This is what the admissibility filter uses for pessimistic cost checks.
        """
        return self.ci[1] if self.ci is not None else self.value

    def is_confident(self, max_ci_width: float) -> bool:
        """Whether this estimate is tight enough for the policy to act on.

        Drives the cold-start rule (section G.4): when the argmax estimate is too
        wide, the controller defers to the hand-tuned heuristic instead of acting
        on noise.
        """
        return self.ci_width <= max_ci_width

    def as_record(self) -> dict:
        """Flatten for the decision record. Keeps lineage queryable in SQL."""
        return {
            "value": self.value,
            "source": self.source,
            "ci_low": self.ci[0] if self.ci else None,
            "ci_high": self.ci[1] if self.ci else None,
            "model_id": self.model_id,
            "data_version": self.data_version,
            "n_effective": self.n_effective,
            "unit": self.unit,
        }

    def with_value(self, value: float) -> "Estimate":
        """Same provenance, new value. Used when rescaling units."""
        if self.ci is not None:
            shift = value - self.value
            return replace(self, value=value, ci=(self.ci[0] + shift, self.ci[1] + shift))
        return replace(self, value=value)


def require_provenance(x: object, what: str = "value") -> Estimate:
    """Gate for policy entry points.

    Call this on anything crossing into scoring or admissibility. A bare float is
    rejected here rather than silently becoming a decision.
    """
    if not isinstance(x, Estimate):
        raise ProvenanceError(
            f"{what} must be an Estimate with declared provenance, got "
            f"{type(x).__name__}. Bare numbers are not admissible in the policy."
        )
    return x
