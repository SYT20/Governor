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


class _Const:
    """Fallback when a level's target has one class (e.g. accuracy 1.0)."""
    def __init__(self, v):
        self.v = float(v)
    def predict(self, X):
        return np.full(len(X), self.v)
    def predict_proba(self, X):
        p = np.full(len(X), self.v)
        return np.column_stack([1 - p, p])


class _ProbaAdapter:
    """Expose predict() as P(correct) so the allocator sees a probability."""
    def __init__(self, clf):
        self.clf = clf
    def predict(self, X):
        return self.clf.predict_proba(X)[:, 1]


def fit_predictors(X: np.ndarray, correct: np.ndarray, tokens: np.ndarray,
                   levels: list[int], seed: int = 0, kind: str = "logistic",
                   calibrate: str | None = None) -> LevelPredictors:
    """One correctness model and one cost model per level.

    CORRECTNESS IS BINARY, SO THE LOSS MUST BE. E0017 fitted RIDGE to it,
    reported R^2 near zero, and that was read as "no signal"; on the same data a
    logistic model scores AUC 0.613-0.741 (E0018). Squared error on a sparse
    binary target says almost nothing about discriminability.

    RANKING IS NOT ENOUGH. The Lagrangian computes qhat - lambda*that, so the
    SCALE of qhat prices the tokens, not just its order. `calibrate` applies
    Platt ("sigmoid") or isotonic calibration, fitted on calibration data only.

    Cost stays on ridge: tokens are continuous and squared error is the right
    loss for them.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    lp = LevelPredictors(levels=list(levels))
    cv = KFold(5, shuffle=True, random_state=seed)
    for j, b in enumerate(levels):
        y = correct[:, j]
        if kind == "ridge" or len(np.unique(y)) < 2:
            if len(np.unique(y)) < 2:
                lp.q[b] = _Const(y.mean())
                lp.cv_r2_q[b] = 0.0
            else:
                oof = cross_val_predict(Ridge(alpha=1.0), X, y, cv=cv)
                ss = float(np.sum((y - y.mean()) ** 2)) or 1e-12
                lp.cv_r2_q[b] = 1.0 - float(np.sum((y - oof) ** 2)) / ss
                lp.q[b] = Ridge(alpha=1.0).fit(X, y)
        else:
            base = make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=2000))
            if calibrate:
                clf = CalibratedClassifierCV(base, method=calibrate, cv=5)
            else:
                clf = base
            clf.fit(X, y)
            lp.q[b] = _ProbaAdapter(clf)
            # Brier from out-of-fold probabilities, on calibration data only.
            oof = cross_val_predict(base, X, y, cv=cv, method="predict_proba")[:, 1]
            lp.cv_r2_q[b] = float(np.mean((oof - y) ** 2))     # Brier, lower better
        yt = tokens[:, j]
        oof = cross_val_predict(Ridge(alpha=1.0), X, yt, cv=cv)
        ss = float(np.sum((yt - yt.mean()) ** 2)) or 1e-12
        lp.cv_r2_t[b] = 1.0 - float(np.sum((yt - oof) ** 2)) / ss
        lp.t[b] = Ridge(alpha=1.0).fit(X, yt)
    return lp


# -- policies ------------------------------------------------------------------

def governor_alloc(Q: np.ndarray, T: np.ndarray, lam: float) -> np.ndarray:
    """Lagrangian: maximise predicted correctness minus the priced token cost."""
    return np.argmax(Q - lam * T, axis=1)


def enforced_alloc(order_idx, choose, T_actual: np.ndarray,
                   levels: list[int], total_budget: float,
                   prompt_tokens: int = 64) -> np.ndarray:
    """Sequential allocation with a HARD runtime budget.

    E0019 tuned lambda so the constraint held on the CALIBRATION half, reported
    the baseline at the nominal budget, and the Governor then spent 15% over on
    evaluation -- which is the entire reason its "+0.0282 BEATS" was withdrawn.
    Tuning cannot guarantee a constraint on data it has not seen. Enforcement
    can.

    Budget forcing truncates a level-b generation at b tokens, so b is a true
    upper bound on that call's cost. Reserving b is therefore safe, and because
    the call usually stops early the slack returns to the pool for later items.
    That gives a hard guarantee -- sum of actual <= n*B -- without throwing away
    the under-spend the way worst-case reservation did in E0013.
    """
    n = len(order_idx)
    remaining = float(total_budget) * n
    out = np.zeros(n, dtype=int)
    for pos, i in enumerate(order_idx):
        left = n - pos - 1
        wanted = int(choose(i))
        # walk down until this level plus a cheapest-level reserve for every
        # remaining item fits in what is left
        j = wanted
        while j > 0 and (levels[j] + prompt_tokens
                         + left * (levels[0] + prompt_tokens)) > remaining:
            j -= 1
        out[i] = j
        remaining -= float(T_actual[i, j])       # charge what it ACTUALLY cost
    return out


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
