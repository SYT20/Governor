"""Allocation under SOFT_EXPECTED_BUDGET:  E[ sum actual tokens ] <= B.

Each item may be given any of the available budget levels. A policy sees only
observable text features and must satisfy the constraint in expectation.

THE TWO POLICIES THAT MATTER, AND THE ONE DIFFERENCE BETWEEN THEM.

  MYOPIC   b_i = the cheapest level whose PREDICTED correctness clears tau.
           "How much does this item seem to need?" -- a difficulty predictor
           with a knob, tuned to hit the budget.

  GOVERNOR b_i = argmax_b [ qhat(i,b) - lambda * that(i,b) ].
           The same predictions, but tokens carry a PRICE. An item needing 4000
           tokens for +0.10 loses to one needing 800 for +0.08. lambda is the
           shadow price of the budget, tuned on calibration.

Both use identical predictors, identical level sets, and are tuned identically
to hit B. The ONLY difference is whether the resource is priced. That isolates
the question Phase 5 could not answer: does opportunity-cost reasoning add
anything beyond knowing which items are hard?

Phase 5 found the full dynamic program statistically indistinguishable from a
`q>0` rule. This is the same question asked where the ceiling is large enough
to answer it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class LevelPredictors:
    """Per-level predictors of correctness and cost, fitted on calibration only."""
    levels: list[int]
    q: dict = field(default_factory=dict)     # level -> correctness model
    t: dict = field(default_factory=dict)     # level -> token model
    cv_r2_q: dict = field(default_factory=dict)
    cv_r2_t: dict = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Q = np.column_stack([np.clip(self.q[b].predict(X), 0.0, 1.0)
                             for b in self.levels])
        T = np.column_stack([np.maximum(self.t[b].predict(X), 1.0)
                             for b in self.levels])
        return Q, T


def fit_predictors(X: np.ndarray, correct: np.ndarray, tokens: np.ndarray,
                   levels: list[int], seed: int = 0) -> LevelPredictors:
    """One correctness model and one cost model per level.

    Ridge rather than boosted trees: with a few hundred calibration items and a
    binary target per level, boosting overfits -- measured in Phase 4, where it
    reported cv_R2 = -0.062 on 40 items while a single feature correlated +0.72.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_predict
    lp = LevelPredictors(levels=list(levels))
    cv = KFold(5, shuffle=True, random_state=seed)
    for j, b in enumerate(levels):
        for name, y, store, r2 in (("q", correct[:, j], lp.q, lp.cv_r2_q),
                                   ("t", tokens[:, j], lp.t, lp.cv_r2_t)):
            m = Ridge(alpha=1.0)
            oof = cross_val_predict(Ridge(alpha=1.0), X, y, cv=cv)
            ss = float(np.sum((y - y.mean()) ** 2)) or 1e-12
            r2[b] = 1.0 - float(np.sum((y - oof) ** 2)) / ss
            store[b] = m.fit(X, y)
    return lp


# -- policies ------------------------------------------------------------------

def governor_alloc(Q: np.ndarray, T: np.ndarray, lam: float) -> np.ndarray:
    """Lagrangian: maximise predicted correctness minus the priced token cost."""
    return np.argmax(Q - lam * T, axis=1)


def myopic_alloc(Q: np.ndarray, tau: float) -> np.ndarray:
    """Cheapest level whose predicted correctness clears tau; else the dearest.

    Tokens are never priced. This is a difficulty predictor with a knob.
    """
    ok = Q >= tau
    idx = np.where(ok.any(axis=1), ok.argmax(axis=1), Q.shape[1] - 1)
    return idx


def tune(alloc_fn, knobs, T_true: np.ndarray, budget: float) -> float:
    """Pick the knob whose REALISED mean cost on calibration is closest to B
    without exceeding it; fall back to the closest if none fit."""
    best, best_gap = None, np.inf
    under = []
    for k in knobs:
        idx = alloc_fn(k)
        cost = float(T_true[np.arange(len(idx)), idx].mean())
        gap = abs(cost - budget)
        if cost <= budget + 1e-9:
            under.append((gap, k))
        if gap < best_gap:
            best, best_gap = k, gap
    return min(under)[1] if under else best


def realised(idx: np.ndarray, C_true: np.ndarray, T_true: np.ndarray) -> dict:
    r = np.arange(len(idx))
    tok = T_true[r, idx]
    return {"utility": float(C_true[r, idx].mean()),
            "mean_tokens": float(tok.mean()), "sd_tokens": float(tok.std()),
            "utility_per_ktoken": 1000.0 * float(C_true[r, idx].mean())
                                  / max(float(tok.mean()), 1e-9),
            "levels_used": {int(u): int(c) for u, c in
                            zip(*np.unique(idx, return_counts=True))}}
