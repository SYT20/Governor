"""Step 9 — the cognitive/Graft layer, as an ablation rather than an assertion.

The directive lists a minimal state: current observations, uncertainty, reasoning
history, remaining budget, previous actions, previous outcomes. This module makes
each of those a switchable component and measures whether it earns its place.

A PREDICTION, RECORDED BEFORE RUNNING IT. Most of these should do nothing, and
the design says why:

  text          should matter -- it is the only thing that identifies the item
  progress      should NOT help the PREDICTOR: an item's gain does not depend on
                where in the episode it appears. It already enters the DECISION
                through the DP's `items_left`.
  budget        same: it is a property of the decision, not of the item
  history       should NOT help. Items are grouped into episodes at random, so
                what earlier calls cost says nothing about the next item's gain.
  uncertainty   may help: knowing the predictor is unsure about THIS item is a
                reason to treat its q as less trustworthy than the threshold.

Env 5 filed `n_blocks_touched` as cognitive state and manufactured a +0.035
effect that vanished when it was moved into the progress control. Predicting
"no effect" in advance and confirming it is the stronger result; adding a
component because held-out utility wobbled upward is how that happened.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from governor.phase4.env import CHEAP, DEEP, P4Env
from governor.phase4.tasks import FEATURE_NAMES, Item, feature_vector

# Components are named so `oracle_leakage` and `progress_as_cognition` can be run
# over the composed feature list. `t` and `items_left` are progress features and
# must be declared justified when used, never smuggled in as cognition.
COMPONENTS: dict[str, tuple[str, ...]] = {
    "text": FEATURE_NAMES,
    "progress": ("pos_t", "pos_items_left"),
    "budget": ("budget_frac_left", "k_affordable"),
    "history": ("hist_n_deep", "hist_n_starved", "hist_tokens_frac"),
    "uncertainty": ("q_sd",),
}
JUSTIFIED_PROGRESS = frozenset({"pos_t", "pos_items_left"})


def state_features(env: P4Env, obs: dict, budget_left: float,
                   components: Sequence[str], q_sd: float = 0.0) -> dict[str, float]:
    """Compose the cognitive state from the requested components only."""
    out: dict[str, float] = {}
    if "text" in components:
        out.update({k: obs["features"][k] for k in FEATURE_NAMES})
    if "progress" in components:
        out["pos_t"] = float(obs["t"])
        out["pos_items_left"] = float(obs["items_left"])
    if "budget" in components:
        out["budget_frac_left"] = float(budget_left / env.budget)
        slack = budget_left - obs["items_left"] * env.cap(CHEAP)
        step = env.cap(DEEP) - env.cap(CHEAP)
        out["k_affordable"] = float(max(0.0, min(obs["items_left"],
                                                 np.floor(slack / step + 1e-9))))
    if "history" in components:
        h = obs["history"]
        out["hist_n_deep"] = float(sum(x["mode"] == DEEP for x in h))
        out["hist_n_starved"] = float(sum(x["starved"] for x in h))
        out["hist_tokens_frac"] = float(
            sum(x["total_tokens"] for x in h) / env.budget)
    if "uncertainty" in components:
        out["q_sd"] = float(q_sd)
    return out


def component_names(components: Sequence[str]) -> list[str]:
    return [n for c in components for n in COMPONENTS[c]]


class StatePredictor:
    """A gain predictor over a COMPOSED cognitive state rather than text alone.

    Trained on per-DECISION rows, because progress, budget and history only
    exist at a decision. The rows are collected by rolling calibration episodes
    out under several reference policies, so the state distribution is not the
    single degenerate trajectory one policy would visit (under all-cheap,
    `hist_n_deep` is always zero and the component looks useless for the wrong
    reason).
    """

    def __init__(self, components: Sequence[str], kind: str = "gbt", seed: int = 0):
        self.components = tuple(components)
        self.names = tuple(component_names(self.components))
        self.kind, self.seed = kind, seed
        self.model = None
        self.q_samples = np.zeros(0)
        self.cv_r2 = float("nan")

    def _vec(self, feats: dict[str, float]) -> np.ndarray:
        return np.array([[feats[k] for k in self.names]], float)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StatePredictor":
        from sklearn.model_selection import KFold, cross_val_predict
        from governor.phase4.predictor import ValuePredictor
        mk = lambda: ValuePredictor(kind=self.kind, seed=self.seed)._new()  # noqa: E731
        oof = cross_val_predict(mk(), X, y,
                                cv=KFold(5, shuffle=True, random_state=self.seed))
        ss_res = float(np.sum((y - oof) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-12
        self.cv_r2 = 1.0 - ss_res / ss_tot
        self.q_samples = oof
        self.model = mk().fit(X, y)
        return self

    def predict_one(self, feats: dict[str, float]) -> float:
        return float(self.model.predict(self._vec(feats))[0])


def collect_state_training(env: P4Env, eps: Sequence[int],
                           components: Sequence[str],
                           policies: dict, ens: "EnsemblePredictor | None" = None
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Per-decision (state, realised gain) rows from calibration rollouts."""
    from governor.gate.executor import run_episode
    names = component_names(components)
    X, y = [], []
    for e in eps:
        for pol in policies.values():
            s = env.reset(e)
            spent = 0.0
            # Re-derive the trajectory rather than trusting a recorded one: the
            # state a policy SAW is the state it acted on.
            tr = run_episode(env, pol, e, env.budget)
            for m, cost in zip(tr.modes, tr.costs):
                obs = env.observe(s)
                sd = ens.spread(obs["features"]) if ens else 0.0
                f = state_features(env, obs, env.budget - spent, components, sd)
                X.append([f[k] for k in names])
                y.append(env.realised_gain(s.items[s.t]))
                s, c = env.step(s, m)
                spent += c
    return np.array(X, float), np.array(y, float)


class EnsemblePredictor:
    """Bootstrap ensemble over the base predictor, for the `uncertainty` component.

    `q_sd` is the spread of member predictions on one item: how much the answer
    depends on which calibration items happened to be drawn. It is a property of
    the predictor's knowledge, not of the item, and it is the only thing in this
    module that could not be read off the prompt.
    """

    def __init__(self, kind: str = "gbt", n_members: int = 12, seed: int = 0):
        self.kind, self.n_members, self.seed = kind, n_members, seed
        self.members: list = []
        self.names: tuple[str, ...] = FEATURE_NAMES

    def fit(self, items: list[Item], gains: np.ndarray,
            names: Sequence[str] = FEATURE_NAMES) -> "EnsemblePredictor":
        from governor.phase4.predictor import ValuePredictor
        self.names = tuple(names)
        X = np.array([feature_vector(i.prompt, self.names) for i in items], float)
        y = np.asarray(gains, float)
        rng = np.random.default_rng(self.seed)
        self.members = []
        for m in range(self.n_members):
            idx = rng.integers(0, len(y), len(y))
            mdl = ValuePredictor(kind=self.kind, seed=m)._new()
            self.members.append(mdl.fit(X[idx], y[idx]))
        return self

    def spread(self, feats: dict[str, float]) -> float:
        x = np.array([[feats[k] for k in self.names]], float)
        return float(np.std([m.predict(x)[0] for m in self.members]))
