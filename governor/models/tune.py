"""Hyperparameter search for the value model — Decision Record D2.

Two rules make this legitimate rather than a way to launder the gate:

1. **The held-out families are never touched.** Tuning against them would turn the
   Stage 3 gate into a training objective, and a gate you optimised against
   measures nothing. The search sees only training families.

2. **Cross-validation is leave-one-family-out.** The gate asks "does this
   generalise to a regime it has never seen", so the search's inner loop asks the
   same question. Tuning under a random split would optimise for interpolation and
   then be surprised by the extrapolation the gate demands.

The objective is Brier (a proper scoring rule, so it cannot be won by declining to
discriminate) plus a penalty for calibration error beyond what sampling noise
explains. Optimising raw ECE alone would reward a model that predicts the base rate
for everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from governor.corpus.build import Checkpoint
from governor.models.calibration import brier, calibration_noise_floor, ece
from governor.models.value import fit_model


@dataclass(slots=True)
class TuneResult:
    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    history: list[dict[str, Any]] = field(default_factory=list)

    def render(self) -> str:
        rows = [f"    best objective {self.best_value:.5f} over {self.n_trials} trials",
                "    best params:"]
        for k, v in sorted(self.best_params.items()):
            rows.append(f"      {k:<22} {v}")
        return "\n".join(rows)


def _fold_score(
    train_ck: list[Checkpoint], test_ck: list[Checkpoint], params: dict[str, Any]
) -> tuple[float, float, float]:
    """Fit on one family-holdout fold and return (brier, ece, noise_p95)."""
    m = fit_model(
        train_ck,
        kind=params["kind"],
        uses_actions=True,
        data_version="tune",
        n_calib_folds=params["n_calib_folds"],
        estimator_kwargs={k: v for k, v in params.items()
                          if k not in ("kind", "n_calib_folds")},
    )
    p = [float(x) for x in m.predict(test_ck)]
    y = [c.label for c in test_ck]
    nf = calibration_noise_floor(p, n_sim=120)
    return brier(y, p), ece(y, p), nf["p95"]


def tune(
    train: list[Checkpoint],
    *,
    n_trials: int = 30,
    seed: int = 0,
    ece_penalty: float = 0.5,
    max_folds: int = 6,
) -> TuneResult:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    families = sorted({c.family for c in train})
    # Cap the fold count for runtime; the folds chosen are spread across the
    # family list rather than the first k, so the search is not tuned to one
    # corner of the regime space.
    step = max(1, len(families) // max_folds)
    fold_families = families[::step][:max_folds]

    def objective(trial: "optuna.Trial") -> float:
        kind = trial.suggest_categorical("kind", ["logistic", "gbm"])
        params: dict[str, Any] = {
            "kind": kind,
            "n_calib_folds": trial.suggest_int("n_calib_folds", 3, 5),
        }
        if kind == "logistic":
            params["C"] = trial.suggest_float("C", 0.02, 10.0, log=True)
        else:
            params["max_depth"] = trial.suggest_int("max_depth", 2, 5)
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.02, 0.25, log=True)
            params["max_iter"] = trial.suggest_int("max_iter", 60, 300, step=20)
            params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 20, 160, step=20)
            params["l2_regularization"] = trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True)

        losses = []
        for fam in fold_families:
            tr = [c for c in train if c.family != fam]
            te = [c for c in train if c.family == fam]
            if len(te) < 50 or len({c.label for c in te}) < 2:
                continue
            b, e, floor = _fold_score(tr, te, params)
            losses.append(b + ece_penalty * max(0.0, e - floor))
        if not losses:
            return float("inf")
        return float(np.mean(losses))

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return TuneResult(
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        n_trials=n_trials,
        history=[{"number": t.number, "value": t.value, "params": t.params}
                 for t in study.trials if t.value is not None],
    )
