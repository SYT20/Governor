"""Task families — SynthBug's stand-in for repositories.

Decision Record section J.1 requires a *repo-level* disjoint split, not an
instance-level one: splitting within a repo leaks file structure, test conventions
and dependency shape straight into the features.

SynthBug has no repos, so families play that role. Each family is a distinct
generative regime (how many candidate causes, how discriminative the evidence, how
reliable verification, how expensive actions are). Holding out whole families is the
same guard as holding out whole repositories, and it is what makes the Stage 3
generalisation claim mean anything: a value model that only works on the regimes it
was fitted to has learned the simulator, not the decision problem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from governor.envs.synthbug import SynthBug, SynthConfig, Tier


def _cost(scale: float) -> dict[str, float]:
    """Per-action-class work means, scaled uniformly."""
    base = {
        "EXPLORE@T0": 1.0, "EXPLORE@T1": 2.2, "EXPLORE@T2": 5.0,
        "EXPLOIT@T0": 2.0, "EXPLOIT@T1": 4.5, "EXPLOIT@T2": 9.0,
        "VERIFY@T0": 1.5, "VERIFY@T1": 3.0, "VERIFY@T2": 6.5,
    }
    return {k: math.log(v * scale) for k, v in base.items()}


@dataclass(frozen=True, slots=True)
class Family:
    """A named generative regime."""

    name: str
    config: SynthConfig
    note: str

    def task(self, seed: int) -> SynthBug:
        return SynthBug(config=self.config, seed=seed)


def _cfg(
    *,
    k: int = 4,
    alpha: tuple[float, float, float] = (0.62, 0.78, 0.91),
    beta: tuple[float, float, float] = (0.34, 0.20, 0.08),
    p_fix: tuple[float, float, float] = (0.45, 0.70, 0.88),
    verify: tuple[float, float, float] = (0.80, 0.93, 0.99),
    cost_scale: float = 1.0,
    p_regress: float = 0.15,
) -> SynthConfig:
    t = (Tier.T0, Tier.T1, Tier.T2)
    return SynthConfig(
        n_hypotheses=k,
        alpha=dict(zip(t, alpha)),
        beta=dict(zip(t, beta)),
        p_fix=dict(zip(t, p_fix)),
        verify_acc=dict(zip(t, verify)),
        work_mu=_cost(cost_scale),
        p_regress=p_regress,
    )


# Ten regimes. The split below holds out three of them entirely.
FAMILIES: tuple[Family, ...] = (
    Family("baseline", _cfg(), "the reference regime"),
    Family("shallow", _cfg(k=2), "few candidate causes; cheap to disambiguate"),
    Family("wide", _cfg(k=7), "many candidate causes; exploration matters more"),
    Family(
        "faint_signal",
        _cfg(alpha=(0.55, 0.64, 0.74), beta=(0.40, 0.32, 0.22)),
        "weakly discriminative evidence; belief moves slowly",
    ),
    Family(
        "sharp_signal",
        _cfg(alpha=(0.75, 0.90, 0.97), beta=(0.22, 0.09, 0.02)),
        "highly discriminative evidence; one probe often suffices",
    ),
    Family(
        "brittle_repair",
        _cfg(p_fix=(0.25, 0.45, 0.65), p_regress=0.30),
        "repairs often miss and frequently regress; verification matters",
    ),
    Family(
        "reliable_repair",
        _cfg(p_fix=(0.70, 0.88, 0.96)),
        "repairs usually land; exploration is the bottleneck",
    ),
    Family(
        "flaky_tests",
        _cfg(verify=(0.62, 0.75, 0.88)),
        "verification is noisy; a single passing test is weak evidence",
    ),
    Family(
        "costly",
        _cfg(cost_scale=2.4),
        "every action is expensive; the envelope binds early",
    ),
    Family(
        "cheap_wide",
        _cfg(k=6, cost_scale=0.5, alpha=(0.60, 0.72, 0.85)),
        "many causes but cheap probes; favours long exploration",
    ),
    Family(
        "deep_uncertainty",
        _cfg(k=8, alpha=(0.52, 0.61, 0.70), beta=(0.42, 0.35, 0.26)),
        "many causes and faint evidence; difficulty compounds",
    ),
    Family(
        "trivial",
        _cfg(k=2, alpha=(0.80, 0.92, 0.98), p_fix=(0.80, 0.92, 0.97), cost_scale=0.7),
        "easy on every axis; a fixed script should already do well",
    ),
    Family(
        "verify_cheap",
        _cfg(verify=(0.94, 0.98, 0.995), cost_scale=1.3, p_fix=(0.35, 0.55, 0.75)),
        "verification is accurate and relatively cheap; favours guess-then-check",
    ),
)

BY_NAME: dict[str, Family] = {f.name: f for f in FAMILIES}

# Held out entirely from corpus collection. Stage 3 reports in-corpus vs
# out-of-corpus calibration on this split; a large gap means the model memorised
# the regimes rather than learning the decision problem.
HELDOUT: tuple[str, ...] = ("sharp_signal", "flaky_tests", "cheap_wide")
TRAIN: tuple[str, ...] = tuple(f.name for f in FAMILIES if f.name not in HELDOUT)


def train_families() -> list[Family]:
    return [BY_NAME[n] for n in TRAIN]


def heldout_families() -> list[Family]:
    return [BY_NAME[n] for n in HELDOUT]


def describe() -> list[dict[str, object]]:
    rows = []
    for f in FAMILIES:
        c = f.config
        rows.append(
            {
                "family": f.name,
                "split": "heldout" if f.name in HELDOUT else "train",
                "k": c.n_hypotheses,
                "alpha_T1": c.alpha[Tier.T1],
                "beta_T1": c.beta[Tier.T1],
                "p_fix_T1": c.p_fix[Tier.T1],
                "verify_T1": c.verify_acc[Tier.T1],
                "note": f.note,
            }
        )
    return rows
