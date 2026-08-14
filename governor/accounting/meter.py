"""Resource metering and hard enforcement.

Decision Record section H.1: estimation and enforcement are separate systems with
different jobs.

  - Statistical admissibility (p90 cost + reserve) reduces the *rate* of overruns.
    It guarantees nothing. Its metric is the truncation rate.
  - Runtime enforcement, implemented here, guarantees BVR = 0 *structurally*: the
    executor is handed a hard cap equal to what remains, and a charge that would
    exceed the envelope is a bug that raises rather than a number that gets logged.

Everything in this module is `measured`. It never estimates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, Mapping

from governor.core.estimate import Estimate

DIMENSIONS = ("tokens", "cost", "wall_s", "tool_calls")


class BudgetViolation(RuntimeError):
    """A charge exceeded the envelope.

    This is always a bug in the executor or the enforcement wiring, never an
    expected outcome. The CI invariant asserts this is never raised.
    """


class BudgetExhausted(RuntimeError):
    """Dispatch refused: no admissible action fits in what remains.

    Expected control flow, not an error condition. The policy catches this and
    terminates the episode.
    """


@dataclass(frozen=True, slots=True)
class Envelope:
    """A hard multi-dimensional resource limit for one episode."""

    tokens: float = float("inf")
    cost: float = float("inf")
    wall_s: float = float("inf")
    tool_calls: float = float("inf")

    def as_dict(self) -> dict[str, float]:
        return {d: getattr(self, d) for d in DIMENSIONS}

    def scaled(self, fraction: float) -> "Envelope":
        """Return this envelope at a fraction of its size.

        This is how the degradation curve (100/75/50/25/10%) is produced without
        touching any other part of the system.
        """
        if fraction <= 0:
            raise ValueError("fraction must be positive")
        return Envelope(
            **{
                d: (v * fraction if v != float("inf") else v)
                for d, v in self.as_dict().items()
            }
        )


@dataclass(slots=True)
class LedgerEntry:
    """One metered event. Append-only."""

    seq: int
    label: str
    tokens: float = 0.0
    cost: float = 0.0
    wall_s: float = 0.0
    tool_calls: float = 0.0
    truncated: bool = False

    def as_dict(self) -> dict[str, float]:
        return {d: getattr(self, d) for d in DIMENSIONS}


@dataclass(slots=True)
class Accountant:
    """Meters actual spend and enforces the envelope.

    The accountant is the only component permitted to say what something cost.
    Every other component either estimates (and says so) or asks the accountant.
    """

    envelope: Envelope
    entries: list[LedgerEntry] = field(default_factory=list)
    _consumed: dict[str, float] = field(
        default_factory=lambda: {d: 0.0 for d in DIMENSIONS}
    )
    truncations: int = 0

    # -- reading ---------------------------------------------------------------

    def consumed(self) -> dict[str, float]:
        return dict(self._consumed)

    def remaining(self) -> dict[str, float]:
        env = self.envelope.as_dict()
        return {d: env[d] - self._consumed[d] for d in DIMENSIONS}

    def consumed_estimate(self, dim: str) -> Estimate:
        """Consumption as a provenance-carrying value, for the decision record."""
        return Estimate.measured(self._consumed[dim], unit=dim)

    def fraction_remaining(self) -> float:
        """Tightest remaining fraction across all bounded dimensions.

        This is the `frac_budget_remaining` feature the value model consumes. It is
        `derived`, and deliberately the *minimum* across dimensions: an episode with
        plenty of tokens but no wall-clock left is a constrained episode.
        """
        env = self.envelope.as_dict()
        fracs = [
            (env[d] - self._consumed[d]) / env[d]
            for d in DIMENSIONS
            if env[d] not in (0, float("inf"))
        ]
        return max(0.0, min(fracs)) if fracs else 1.0

    # -- statistical admissibility (section H.1, upper layer) ------------------

    def admissible(
        self,
        projected: Mapping[str, Estimate],
        reserve: Mapping[str, float] | None = None,
    ) -> bool:
        """Would this action still leave room to terminate safely?

        Uses the *upper* bound of each projected cost, not the point estimate.
        Guarantees nothing on its own -- that is what `charge` is for -- but this
        is the lever that keeps the truncation rate low.
        """
        reserve = reserve or {}
        rem = self.remaining()
        for dim, est in projected.items():
            if dim not in rem:
                raise KeyError(f"unknown budget dimension {dim!r}")
            if est.ucb + reserve.get(dim, 0.0) > rem[dim]:
                return False
        return True

    # -- hard enforcement (section H.1, lower layer) ---------------------------

    def hard_cap(self) -> dict[str, float]:
        """The cap handed to the executor so it physically cannot overspend.

        The executor must treat these as kill thresholds: stop generating at the
        token cap, abort at the wall-clock cap.
        """
        return {d: max(0.0, v) for d, v in self.remaining().items()}

    def charge(self, label: str, *, truncated: bool = False, **amounts: float) -> LedgerEntry:
        """Record actual spend. Raises if it would break the envelope.

        Reaching `BudgetViolation` means the executor ignored its hard cap. It is a
        bug detector, not a control-flow mechanism -- the policy should have stopped
        dispatching long before this.
        """
        unknown = set(amounts) - set(DIMENSIONS)
        if unknown:
            raise KeyError(f"unknown budget dimension(s) {sorted(unknown)}")
        for dim, amt in amounts.items():
            if amt < 0:
                raise ValueError(f"negative charge {amt} on {dim}")

        env = self.envelope.as_dict()
        for dim, amt in amounts.items():
            if self._consumed[dim] + amt > env[dim] + 1e-9:
                raise BudgetViolation(
                    f"charge {amt} on {dim!r} would take consumed from "
                    f"{self._consumed[dim]} past envelope {env[dim]}. "
                    "The executor overran its hard cap."
                )

        entry = LedgerEntry(seq=len(self.entries), label=label, truncated=truncated, **amounts)
        for dim, amt in amounts.items():
            self._consumed[dim] += amt
        self.entries.append(entry)
        if truncated:
            self.truncations += 1
        return entry

    def dispatch_or_refuse(self, floor: Mapping[str, float]) -> None:
        """Refuse to start an action that cannot possibly fit.

        `floor` is the cheapest conceivable cost of the cheapest action. If even
        that does not fit, the episode is over.
        """
        rem = self.remaining()
        for dim, amt in floor.items():
            if rem.get(dim, 0.0) < amt:
                raise BudgetExhausted(
                    f"{dim!r} has {rem.get(dim, 0.0):.4g} remaining, "
                    f"cheapest action needs {amt:.4g}"
                )

    # -- invariants ------------------------------------------------------------

    def reconcile(self, tol: float = 1e-6) -> None:
        """Drift check #2: the ledger must sum to the running totals.

        Runs in CI on every episode, forever. Silent accounting drift is the failure
        mode that invalidates results months after it starts.
        """
        for dim in DIMENSIONS:
            total = sum(getattr(e, dim) for e in self.entries)
            if abs(total - self._consumed[dim]) > tol:
                raise AssertionError(
                    f"ledger drift on {dim!r}: entries sum to {total}, "
                    f"consumed says {self._consumed[dim]}"
                )

    def violated(self) -> bool:
        """True if any dimension exceeded its envelope. Must always be False."""
        env = self.envelope.as_dict()
        return any(self._consumed[d] > env[d] + 1e-9 for d in DIMENSIONS)

    def summary(self) -> dict[str, object]:
        return {
            "consumed": self.consumed(),
            "remaining": self.remaining(),
            "envelope": self.envelope.as_dict(),
            "n_entries": len(self.entries),
            "truncations": self.truncations,
            "violated": self.violated(),
            "utilisation": {
                d: (self._consumed[d] / v if v not in (0, float("inf")) else 0.0)
                for d, v in self.envelope.as_dict().items()
            },
        }


class WallClock:
    """Monotonic wall-clock charging. Separated so tests can inject a fake."""

    def __init__(self, accountant: Accountant, label: str) -> None:
        self.accountant = accountant
        self.label = label
        self._t0: float | None = None

    def __enter__(self) -> "WallClock":
        self._t0 = time.monotonic()
        return self

    def __exit__(self, *exc: object) -> None:
        assert self._t0 is not None
        self.accountant.charge(self.label, wall_s=time.monotonic() - self._t0)


def iter_entries(accountant: Accountant) -> Iterator[dict]:
    """Flatten the ledger for persistence."""
    for e in accountant.entries:
        yield {"seq": e.seq, "label": e.label, "truncated": e.truncated, **e.as_dict()}
