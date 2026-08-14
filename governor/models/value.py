"""Value / action-value models — Decision Record section G.2, deciding D1.

Two candidates, fitted on the same corpus and compared on held-out families:

  Option A (direct)    Q(s,a) = P(success | s, a), one model with action indicators
  Option B (composed)  V(s,b) = P(success | s, b)
                       T(s,a) = fitted feature deltas per action class
                       Q(s,a) = E[ V(s') ]

Neither escapes confounding on its own -- that is what the randomised corpus of
section G.1 is for. What this module decides is which *estimator* generalises
better once the data supports both.

Guards implemented here, because each is a way to pass the gate without deserving
to:

  * Grouping. All cross-validation is grouped by episode. Checkpoints from one
    episode share a label, so an ungrouped split leaks the answer across the fold
    boundary and inflates every metric.
  * Calibration isolation. The isotonic calibrator is fitted on a held-out fold of
    the *training* families, never on the evaluation families.
  * Honest baseline. The base rate handed to the scorer comes from the training
    split.
  * Terminal-state audit. A checkpoint whose state already says "tests passed" is
    nearly deterministic. Those are legitimate decision points, but if the model's
    skill comes only from them it has learned to read a thermometer, not to
    predict. `audit_terminal_dependence` reports the split.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from governor.corpus.build import Checkpoint

# Features deliberately excluded from the model.
#
# `n_hypotheses` is a family fingerprint: it takes a distinct value per regime, so
# a model given it can identify which family a checkpoint came from and learn
# per-family base rates. That inflates in-corpus scores and collapses on held-out
# families with unseen values. It is exactly the leak the repo-level split exists
# to prevent, so it is dropped rather than trusted.
EXCLUDED_FEATURES = frozenset({"n_hypotheses"})


def feature_matrix(
    checkpoints: list[Checkpoint], names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, groups) with groups = episode id, for grouped CV."""
    X = np.array([[c.features[n] for n in names] for c in checkpoints], dtype=float)
    y = np.array([c.label for c in checkpoints], dtype=int)
    g = np.array([c.episode_id for c in checkpoints])
    return X, y, g


def usable_feature_names(checkpoints: list[Checkpoint]) -> list[str]:
    if not checkpoints:
        return []
    return sorted(set(checkpoints[0].features) - EXCLUDED_FEATURES)


def action_columns(checkpoints: list[Checkpoint]) -> list[str]:
    return sorted({f"{c.mode}@{c.tier}" for c in checkpoints})


def with_action_indicators(
    checkpoints: list[Checkpoint], names: list[str], acts: list[str],
    channels: dict | None = None,
) -> np.ndarray:
    """Direct-Q design matrix: state features plus one-hot action, plus the
    interaction of the action with budget remaining.

    The interaction matters: the value of an expensive action depends on how much
    budget is left, and a purely additive action effect cannot express that.
    """
    X, _, _ = feature_matrix(checkpoints, names)
    idx = {a: i for i, a in enumerate(acts)}
    A = np.zeros((len(checkpoints), len(acts)))
    for r, c in enumerate(checkpoints):
        A[r, idx[f"{c.mode}@{c.tier}"]] = 1.0
    bcol = names.index("frac_budget_remaining")
    inter = A * X[:, [bcol]]
    blocks = [X, A, inter]

    if channels is not None:
        # Stage 4B showed the estimator was not the bottleneck: the model had no
        # feature describing what a given action would DO in a given state, only a
        # one-hot saying which action it was. These columns supply that, computed
        # as deterministic arithmetic over the belief and the measured channel.
        from governor.models.action_features import ACTION_FEATURE_NAMES, action_features
        AF = np.zeros((len(checkpoints), len(ACTION_FEATURE_NAMES)))
        for r, c in enumerate(checkpoints):
            tgt = int(c.action.split("->h")[1]) if "->h" in c.action else None
            f = action_features(
                belief=c.belief, mode=c.mode, tier=c.tier, target=tgt,
                channels=channels,
                n_exploit=c.features.get("n_exploit", 0.0),
                n_verify=c.features.get("n_verify", 0.0),
            )
            AF[r] = [f[k] for k in ACTION_FEATURE_NAMES]
        blocks.append(AF)
    return np.hstack(blocks)


@dataclass(slots=True)
class FittedModel:
    """A calibrated probability model plus the lineage its Estimates will carry."""

    name: str
    pipeline: object
    feature_names: list[str]
    action_names: list[str]
    train_base_rate: float
    data_version: str
    uses_actions: bool
    n_effective: int
    channels: dict | None = None

    def predict(self, checkpoints: list[Checkpoint]) -> np.ndarray:
        X = (
            with_action_indicators(checkpoints, self.feature_names, self.action_names,
                                   self.channels)
            if self.uses_actions
            else feature_matrix(checkpoints, self.feature_names)[0]
        )
        return self.pipeline.predict_proba(X)[:, 1]


def _make_estimator(kind: str, **kw):
    """Build an estimator; `kw` comes from the optuna search (D2)."""
    if kind == "logistic":
        return Pipeline(
            [("scale", StandardScaler()),
             ("clf", LogisticRegression(C=kw.get("C", 0.5), max_iter=2000, solver="lbfgs"))]
        )
    if kind == "gbm":
        return HistGradientBoostingClassifier(
            max_depth=kw.get("max_depth", 3),
            max_iter=kw.get("max_iter", 180),
            learning_rate=kw.get("learning_rate", 0.06),
            min_samples_leaf=kw.get("min_samples_leaf", 40),
            l2_regularization=kw.get("l2_regularization", 1.0),
            random_state=0,
        )
    raise ValueError(f"unknown estimator {kind!r}")


def fit_model(
    checkpoints: list[Checkpoint],
    *,
    kind: str = "logistic",
    uses_actions: bool = True,
    data_version: str = "corpus",
    n_calib_folds: int = 4,
    estimator_kwargs: dict | None = None,
    channels: dict | None = None,
) -> FittedModel:
    """Fit and calibrate, with every split grouped by episode."""
    names = usable_feature_names(checkpoints)
    acts = action_columns(checkpoints) if uses_actions else []
    X = (
        with_action_indicators(checkpoints, names, acts, channels)
        if uses_actions
        else feature_matrix(checkpoints, names)[0]
    )
    _, y, groups = feature_matrix(checkpoints, names)

    base = _make_estimator(kind, **(estimator_kwargs or {}))
    # GroupKFold keeps every checkpoint of an episode inside one fold, so the
    # calibrator never sees a label it could have memorised from a sibling row.
    #
    # CalibratedClassifierCV does not forward `groups` to its splitter (sklearn
    # 1.5), so passing a GroupKFold instance silently raises rather than grouping.
    # Precomputing the splits and handing over the index pairs is the reliable way
    # to keep the grouping guarantee -- getting this wrong would inflate every
    # metric downstream without any visible symptom.
    n_splits = max(2, min(n_calib_folds, len(set(groups))))
    splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=splits)
    calibrated.fit(X, y)

    return FittedModel(
        name=f"{'Q' if uses_actions else 'V'}_{kind}",
        pipeline=calibrated,
        feature_names=names,
        action_names=acts,
        train_base_rate=float(y.mean()),
        data_version=data_version,
        uses_actions=uses_actions,
        n_effective=len(set(groups)),  # episodes, not checkpoints (see J.8)
        channels=channels,
    )


# -- Option B: the composed estimator ------------------------------------------


@dataclass(slots=True)
class EffectModel:
    """T(s,a): mean feature delta per action class, from randomised decisions only.

    Restricting to `was_random` rows is the point of the epsilon-greedy corpus: on
    those decisions the action was chosen independently of the state, so the
    observed change is an unconfounded estimate of the action's effect. Fitting on
    all rows would re-introduce the behaviour policy's preferences.
    """

    deltas: dict[str, np.ndarray] = field(default_factory=dict)
    fallback: np.ndarray | None = None
    names: list[str] = field(default_factory=list)
    n_obs: dict[str, int] = field(default_factory=dict)

    @classmethod
    def fit(cls, checkpoints: list[Checkpoint], names: list[str]) -> "EffectModel":
        by_ep: dict[str, list[Checkpoint]] = {}
        for c in checkpoints:
            by_ep.setdefault(c.episode_id, []).append(c)
        for v in by_ep.values():
            v.sort(key=lambda c: c.decision_id)

        acc: dict[str, list[np.ndarray]] = {}
        for eps in by_ep.values():
            for a, b in zip(eps, eps[1:]):
                if not a.was_random:
                    continue  # unconfounded subset only
                key = f"{a.mode}@{a.tier}"
                d = np.array([b.features[n] - a.features[n] for n in names], dtype=float)
                acc.setdefault(key, []).append(d)

        deltas = {k: np.mean(np.vstack(v), axis=0) for k, v in acc.items() if v}
        n_obs = {k: len(v) for k, v in acc.items()}
        allv = [d for v in acc.values() for d in v]
        fallback = np.mean(np.vstack(allv), axis=0) if allv else np.zeros(len(names))
        return cls(deltas=deltas, fallback=fallback, names=names, n_obs=n_obs)

    def apply(self, c: Checkpoint) -> np.ndarray:
        base = np.array([c.features[n] for n in self.names], dtype=float)
        key = f"{c.mode}@{c.tier}"
        d = self.deltas.get(key)
        if d is None or self.n_obs.get(key, 0) < 10:
            d = self.fallback if self.fallback is not None else np.zeros(len(self.names))
        return base + d


@dataclass(slots=True)
class ComposedModel:
    """Q(s,a) = V(s') where s' is the state T(s,a) predicts the action leads to."""

    v: FittedModel
    effect: EffectModel
    name: str = "V_compose_T"

    @property
    def train_base_rate(self) -> float:
        return self.v.train_base_rate

    @property
    def n_effective(self) -> int:
        return self.v.n_effective

    def predict(self, checkpoints: list[Checkpoint]) -> np.ndarray:
        X = np.vstack([self.effect.apply(c) for c in checkpoints])
        return self.v.pipeline.predict_proba(X)[:, 1]


def fit_composed(
    checkpoints: list[Checkpoint], *, kind: str = "logistic", data_version: str = "corpus"
) -> ComposedModel:
    names = usable_feature_names(checkpoints)
    v = fit_model(checkpoints, kind=kind, uses_actions=False, data_version=data_version)
    return ComposedModel(v=v, effect=EffectModel.fit(checkpoints, names))


# -- leakage audit --------------------------------------------------------------


def audit_terminal_dependence(
    checkpoints: list[Checkpoint], preds: np.ndarray
) -> dict[str, object]:
    """Split performance by whether the state already knew the answer.

    A checkpoint taken after a passing test is nearly deterministic. Those are real
    decision points and excluding them would be dishonest -- but if all of the
    model's skill lives there, it is reading a thermometer rather than forecasting,
    and the policy will get no useful guidance early in an episode when it matters.
    """
    from governor.models.calibration import evaluate

    verified = [i for i, c in enumerate(checkpoints) if c.features.get("last_test_pass", -1.0) == 1.0]
    unknown = [i for i, c in enumerate(checkpoints) if c.features.get("last_test_pass", -1.0) != 1.0]
    br = float(np.mean([c.label for c in checkpoints]))

    def sub(idx: list[int]) -> dict[str, float]:
        if len(idx) < 20:
            return {}
        y = [checkpoints[i].label for i in idx]
        p = [float(preds[i]) for i in idx]
        r = evaluate(y, p, train_base_rate=br)
        return {"n": len(idx), "brier_skill": r.skill["brier"], "auc": r.model["auc"],
                "base_rate": r.base_rate}

    return {"post_pass": sub(verified), "pre_resolution": sub(unknown)}


# -- Option C: explicit advantage model (added after Stage 4A) ------------------


@dataclass(slots=True)
class AdvantageModel:
    """Q(s,a) = V(s) + A(s,a), with the action effect fitted on residuals.

    Stage 4A showed the pooled Q model ranks STATES (pooled AUC 0.75) far better
    than it ranks ACTIONS at a fixed state (0.58, and 0.52 on training regimes).
    The cause is structural: fitting P(success | state features, action indicator)
    against a target dominated by state difficulty lets the model spend all its
    capacity on "is this episode going well" and treat the action indicator as
    near-noise. Bellemare et al.'s action-gap result is the same observation from
    the RL side -- when action values at a state are close, estimation error swamps
    the induced ordering.

    A note on what this does and does not fix. Within a single state V(s) is a
    constant, so subtracting it cannot change the ordering of actions -- advantage
    ranking and Q ranking are identical *as an evaluation*. The gain is entirely in
    FITTING: stage two regresses on residuals, where the state-difficulty signal has
    already been removed, so the action effect is what remains to be explained
    rather than a rounding error on top of it.

    Stage two uses only randomised decisions. On those the action was chosen
    independently of the state, so the residual difference between actions is an
    unconfounded estimate of the effect rather than a record of what the behaviour
    policy preferred.
    """

    v: FittedModel
    effect: object
    feature_names: list[str]
    action_names: list[str]
    train_base_rate: float
    n_effective: int
    name: str = "V_plus_advantage"
    data_version: str = "corpus"

    def advantage(self, checkpoints: list[Checkpoint]) -> np.ndarray:
        X = with_action_indicators(checkpoints, self.feature_names, self.action_names)
        return self.effect.predict(X)

    def predict(self, checkpoints: list[Checkpoint]) -> np.ndarray:
        base = self.v.pipeline.predict_proba(
            feature_matrix(checkpoints, self.feature_names)[0]
        )[:, 1]
        return np.clip(base + self.advantage(checkpoints), 1e-6, 1 - 1e-6)


def fit_advantage(
    checkpoints: list[Checkpoint],
    *,
    data_version: str = "corpus",
    randomised_only: bool = True,
) -> AdvantageModel:
    from sklearn.ensemble import HistGradientBoostingRegressor

    names = usable_feature_names(checkpoints)
    acts = action_columns(checkpoints)

    # Stage 1: state value, no action information at all.
    v = fit_model(checkpoints, kind="gbm", uses_actions=False,
                  data_version=data_version, n_calib_folds=3,
                  estimator_kwargs=dict(max_depth=3, max_iter=150,
                                        learning_rate=0.08, min_samples_leaf=40))

    # Stage 2: what the action adds, on top of what the state already explains.
    rows = [c for c in checkpoints if c.was_random] if randomised_only else checkpoints
    if len(rows) < 200:
        rows = checkpoints
    base = v.pipeline.predict_proba(feature_matrix(rows, names)[0])[:, 1]
    resid = np.array([c.label for c in rows], dtype=float) - base
    X = with_action_indicators(rows, names, acts)
    eff = HistGradientBoostingRegressor(
        max_depth=3, max_iter=200, learning_rate=0.06,
        min_samples_leaf=30, l2_regularization=1.0, random_state=0,
    )
    eff.fit(X, resid)

    return AdvantageModel(
        v=v, effect=eff, feature_names=names, action_names=acts,
        train_base_rate=v.train_base_rate,
        n_effective=len({c.episode_id for c in rows}),
        data_version=data_version,
    )
