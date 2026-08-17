"""The value predictor and the allocation rule it feeds.

WHAT IS PREDICTED. Not P(correct) — the GAIN from spending the larger budget on
this item rather than the smaller one:

    q(text) = E[ correct(HIGH) - correct(LOW) | observable text ]

This is the estimand fix that made Env 6 work, carried over. P(success) is
dominated by how hard the item is, which is largely shared across both modes and
therefore irrelevant to the decision. The DIFFERENCE is what the allocation
question actually asks about, and it is a much smaller, much more learnable
signal.

WHAT REPLACES THE ENV 6 HEURISTIC. Env 6 compared this item's predicted gain
against a one-step estimate of the next slot. That was a reasonable heuristic and
it is not optimal. Sequential arrival with a known gain distribution and k
upgrades left over m remaining items has an exact dynamic program:

    V(0, k) = V(m, 0) = 0
    V(m, k) = E_q[ max( q + V(m-1, k-1),  V(m-1, k) ) ]
    upgrade iff  q >= V(m-1, k) - V(m-1, k-1)  =:  threshold(m, k)

The threshold is the opportunity cost of the upgrade: what that unit of budget
is worth if saved. It falls as items run out and rises as budget gets scarce,
which is the behaviour a scheduler is supposed to have and which no constant
rule can imitate.

E_q is taken over the CALIBRATION distribution of predicted gains. Nothing from
the test split enters, including its feature distribution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from governor.phase4.tasks import FEATURE_NAMES, Item, feature_vector


@dataclass(slots=True)
class FitReport:
    n: int
    mean_gain: float
    cv_r2: float
    cv_mae: float
    baseline_mae: float
    spread: float           # SD of predicted gain -- if ~0 there is no signal


class ValuePredictor:
    """q(text) -> expected gain. Trained on calibration items only."""

    def __init__(self, kind: str = "gbt", seed: int = 0, family=None):
        from governor.phase4.family import ARITHMETIC
        self.family = family or ARITHMETIC
        self.kind, self.seed = kind, seed
        self.model = None
        self.q_samples: np.ndarray = np.zeros(0)
        self.report: FitReport | None = None
        self.feature_names: tuple[str, ...] = self.family.feature_names

    def _new(self):
        if self.kind == "ridge":
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0)
        if self.kind == "mean":                      # deliberate null model
            from sklearn.dummy import DummyRegressor
            return DummyRegressor(strategy="mean")
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_depth=3, max_iter=200,
                                             learning_rate=0.06,
                                             random_state=self.seed)

    def fit(self, items: list[Item], gains: np.ndarray,
            feature_names=None) -> FitReport:
        from sklearn.model_selection import KFold, cross_val_predict
        self.feature_names = tuple(feature_names or self.family.feature_names)
        X = np.array([self.family.vector(i.prompt, self.feature_names)
                      for i in items], float)
        y = np.asarray(gains, float)
        cv = KFold(5, shuffle=True, random_state=self.seed)
        oof = cross_val_predict(self._new(), X, y, cv=cv)
        ss_res = float(np.sum((y - oof) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-12
        self.model = self._new().fit(X, y)
        self.q_samples = oof                # OUT-OF-FOLD, not in-sample fits:
        # the DP's expectation must reflect the accuracy the predictor actually
        # has on unseen items, or the thresholds are calibrated to a sharper
        # signal than exists and the Governor over-selects.
        self.report = FitReport(
            n=len(y), mean_gain=float(y.mean()),
            cv_r2=1.0 - ss_res / ss_tot,
            cv_mae=float(np.mean(np.abs(y - oof))),
            baseline_mae=float(np.mean(np.abs(y - y.mean()))),
            spread=float(np.std(oof)))
        return self.report

    def predict_one(self, feats: dict[str, float]) -> float:
        x = np.array([[feats[k] for k in self.feature_names]], float)
        return float(self.model.predict(x)[0])

    def predict_items(self, items: list[Item]) -> np.ndarray:
        X = np.array([self.family.vector(i.prompt, self.feature_names)
                      for i in items], float)
        return self.model.predict(X)


class OpportunityCostDP:
    """Thresholds from the calibration gain distribution. Frozen after fit."""

    def __init__(self, q_samples: np.ndarray, n_items: int = 4,
                 max_k: int = 4):
        self.q = np.asarray(q_samples, float)
        self.n_items, self.max_k = n_items, max_k
        self.V = self._solve()

    def _solve(self) -> np.ndarray:
        V = np.zeros((self.n_items + 1, self.max_k + 1))
        for m in range(1, self.n_items + 1):
            for k in range(1, self.max_k + 1):
                take = self.q + V[m - 1, k - 1]
                skip = V[m - 1, k]
                V[m, k] = float(np.mean(np.maximum(take, skip)))
        return V

    def threshold(self, items_left: int, k: int) -> float:
        """Opportunity cost of one upgrade with `items_left` items and k budget.

        With k >= items_left every remaining item can be upgraded, so the
        opportunity cost is zero and the rule reduces to 'upgrade if it helps'.
        """
        if k <= 0:
            return float("inf")
        m = min(items_left, self.n_items)
        k = min(k, self.max_k)
        if k >= m:
            return 0.0
        return float(self.V[m - 1, k] - self.V[m - 1, k - 1])

    def table(self) -> dict[str, float]:
        return {f"m{m}_k{k}": round(self.threshold(m, k), 5)
                for m in range(1, self.n_items + 1)
                for k in range(0, self.max_k + 1)}
