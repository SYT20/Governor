"""Calibration metrics, always reported against a base-rate predictor.

Decision Record section G.4. The gate for Stage 3 is not "the model has good Brier
score" -- a constant predictor that always outputs the training base rate scores a
respectable Brier on a balanced dataset. The gate is that the model *beats that
constant*, on families it was never fitted to.

Reporting skill scores rather than raw scores makes that impossible to fudge:
a skill score of 0 means "no better than predicting the base rate", and negative
means actively worse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_EPS = 1e-12


def brier(y: list[int], p: list[float]) -> float:
    return sum((pi - yi) ** 2 for yi, pi in zip(y, p)) / max(len(y), 1)


def log_loss(y: list[int], p: list[float], eps: float = 1e-6) -> float:
    tot = 0.0
    for yi, pi in zip(y, p):
        q = min(max(pi, eps), 1 - eps)
        tot -= yi * math.log(q) + (1 - yi) * math.log(1 - q)
    return tot / max(len(y), 1)


def reliability_bins(
    y: list[int], p: list[float], n_bins: int = 10
) -> list[dict[str, float]]:
    """Equal-width bins with observed frequency vs mean predicted probability."""
    out = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, pi in enumerate(p) if (pi >= lo and (pi < hi or (b == n_bins - 1 and pi <= hi)))]
        if not idx:
            out.append({"lo": lo, "hi": hi, "n": 0, "conf": 0.0, "acc": 0.0, "gap": 0.0})
            continue
        conf = sum(p[i] for i in idx) / len(idx)
        acc = sum(y[i] for i in idx) / len(idx)
        out.append({"lo": lo, "hi": hi, "n": len(idx), "conf": conf, "acc": acc,
                    "gap": abs(acc - conf)})
    return out


def ece(y: list[int], p: list[float], n_bins: int = 10) -> float:
    """Expected Calibration Error: sample-weighted |accuracy - confidence|."""
    n = max(len(y), 1)
    return sum(b["gap"] * b["n"] for b in reliability_bins(y, p, n_bins)) / n


def auc(y: list[int], p: list[float]) -> float:
    """Rank-based AUC via the Mann-Whitney relation. Ties get half credit."""
    pos = [pi for yi, pi in zip(y, p) if yi == 1]
    neg = [pi for yi, pi in zip(y, p) if yi == 0]
    if not pos or not neg:
        return 0.5
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg], key=lambda t: t[0])
    # assign average ranks
    ranks: dict[int, float] = {}
    i = 0
    r = 1.0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg = (r + (r + (j - i))) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg
        r += (j - i + 1)
        i = j + 1
    sum_pos_ranks = sum(ranks[k] for k, (_, lab) in enumerate(allv) if lab == 1)
    n1, n0 = len(pos), len(neg)
    return (sum_pos_ranks - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def calibration_noise_floor(
    p: list[float], *, n_sim: int = 400, n_bins: int = 10, seed: int = 0
) -> dict[str, float]:
    """ECE distribution a *perfectly calibrated* model would show on this sample.

    ECE is not a proper scoring rule: it penalises resolution. A constant predictor
    puts every sample in one bin and so cannot be miscalibrated across bins, while a
    model that spreads n samples over k bins pays sampling noise of roughly
    sqrt(p(1-p)/n_bin) in every bin. Comparing the two directly is a comparison a
    perfectly calibrated model loses by construction.

    The honest null is therefore not "the base rate" but "a model whose predicted
    probabilities are exactly right". Simulate labels from the model's own
    predictions, recompute ECE, and read off the distribution. An observed ECE
    inside that distribution means the model is as calibrated as this much data can
    demonstrate.
    """
    import random as _random

    rng = _random.Random(seed)
    sims = []
    for _ in range(n_sim):
        y_sim = [1 if rng.random() < pi else 0 for pi in p]
        sims.append(ece(y_sim, p, n_bins))
    sims.sort()

    def q(f: float) -> float:
        return sims[min(len(sims) - 1, max(0, int(f * len(sims))))]

    return {"p50": q(0.50), "p90": q(0.90), "p95": q(0.95), "mean": sum(sims) / len(sims)}


@dataclass(slots=True)
class CalibrationReport:
    """Everything Stage 3's gate needs, plus what a reviewer will ask for."""

    n: int
    base_rate: float
    model: dict[str, float] = field(default_factory=dict)
    baseline: dict[str, float] = field(default_factory=dict)
    skill: dict[str, float] = field(default_factory=dict)
    bins: list[dict[str, float]] = field(default_factory=list)
    noise_floor: dict[str, float] = field(default_factory=dict)

    @property
    def calibrated_within_noise(self) -> bool:
        """Observed ECE is no worse than perfect calibration would look here."""
        p95 = self.noise_floor.get("p95")
        return p95 is not None and self.model.get("ece", 1.0) <= p95

    @property
    def beats_base_rate(self) -> bool:
        """The Stage 3 gate.

        Both components must hold:
          * strictly positive skill on Brier AND log loss -- these are proper
            scoring rules, so a model cannot win by declining to discriminate;
          * ECE within the noise floor for perfect calibration on this sample.

        Revision 1 of this gate required beating the base rate's *ECE*, which a
        zero-resolution constant wins almost automatically. That was a
        misspecification, corrected after Stage 3's first run and recorded in the
        decision log rather than silently relaxed.
        """
        proper = self.skill.get("brier", 0.0) > 0.0 and self.skill.get("log_loss", 0.0) > 0.0
        return proper and self.calibrated_within_noise

    def render(self, title: str = "") -> str:
        h = f"    {title}" if title else ""
        rows = [
            h,
            f"      {'metric':<12} {'model':>10} {'base rate':>11} {'skill':>9}",
            f"      {'-'*12} {'-'*10} {'-'*11} {'-'*9}",
        ]
        for k in ("brier", "log_loss", "ece"):
            s = self.skill.get(k, 0.0)
            rows.append(
                f"      {k:<12} {self.model.get(k, 0):>10.4f} "
                f"{self.baseline.get(k, 0):>11.4f} {s:>+9.1%}"
            )
        rows.append(f"      {'auc':<12} {self.model.get('auc', 0):>10.4f} "
                    f"{0.5:>11.4f} {'':>9}")
        nf = self.noise_floor
        if nf:
            mark = "within" if self.calibrated_within_noise else "ABOVE"
            rows.append(
                f"      ECE noise floor (perfect calibration): "
                f"p50={nf['p50']:.4f} p95={nf['p95']:.4f}  -> observed is {mark}"
            )
        rows.append(f"      n={self.n}  base_rate={self.base_rate:.3f}")
        return "\n".join(rows)


def evaluate(
    y: list[int], p: list[float], *, train_base_rate: float, n_bins: int = 10
) -> CalibrationReport:
    """Score predictions against a constant train-base-rate predictor.

    `train_base_rate` must come from the TRAINING split. Using the test split's own
    base rate would hand the baseline information the model never had, which makes
    the comparison meaningless in the baseline's favour.
    """
    const = [train_base_rate] * len(y)
    m = {"brier": brier(y, p), "log_loss": log_loss(y, p), "ece": ece(y, p, n_bins),
         "auc": auc(y, p)}
    b = {"brier": brier(y, const), "log_loss": log_loss(y, const),
         "ece": ece(y, const, n_bins), "auc": 0.5}

    def skill(mk: float, bk: float) -> float:
        # Skill score: fraction of the baseline's error removed. Lower-is-better
        # metrics only, so positive means improvement.
        return 0.0 if bk <= _EPS else (bk - mk) / bk

    return CalibrationReport(
        n=len(y),
        base_rate=sum(y) / max(len(y), 1),
        model=m,
        baseline=b,
        skill={k: skill(m[k], b[k]) for k in ("brier", "log_loss", "ece")},
        bins=reliability_bins(y, p, n_bins),
        noise_floor=calibration_noise_floor(p, n_bins=n_bins),
    )
