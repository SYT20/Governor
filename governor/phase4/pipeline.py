"""The Phase 4 pipeline, as functions rather than as a script.

Split out so the whole calibrate -> freeze -> evaluate sequence can be exercised
against a synthetic cache in the test suite. A pipeline that only runs when the
network does is a pipeline whose logic is validated by the same run that
produces the result, which is how a scoring bug ships as a finding.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from governor.harness.ledger import file_commit
from governor.harness.traps import run_trap_checks, secret_scan
from governor.phase4.config import HEURISTIC_FEATURES, HEURISTIC_QUANTILES
from governor.phase4.env import DEEP, P4Env
from governor.phase4.evaluate import (
    PolicyResult, constant, execute, paired_ci, token_evidence,
)
from governor.phase4.policies import (
    all_cheap, clairvoyant, fixed_schedule, governor, greedy, text_heuristic,
)
from governor.phase4.predictor import FitReport, OpportunityCostDP, ValuePredictor
from governor.phase4.tasks import FEATURE_NAMES, Item, features

BASELINES = ("H_all_cheap", "fixed_best", "fixed_all_deep", "greedy", "heuristic")


@dataclass(slots=True)
class Calibration:
    """Everything chosen on the calibration split, frozen before the test split.

    Held as one object so it is obvious what was fitted and impossible to fit
    anything else later without adding a field here.
    """
    best_schedule: tuple[int, ...]
    schedule_utilities: dict
    heuristic_feature: str
    heuristic_threshold: float
    heuristic_utility: float
    predictor: ValuePredictor
    dp: OpportunityCostDP
    report: FitReport
    gains: np.ndarray
    cal_utilities: dict
    base: str
    predictor_kind: str = "auto"


def _feature_values(env: P4Env, eps: Sequence[int], name: str) -> np.ndarray:
    """Feature values over every item in every episode, not just position 0."""
    return np.array([features(it.prompt)[name]
                     for e in eps for it in env.episodes[e]], float)


def calibrate(env: P4Env, pool: list[Item], eps: Sequence[int],
              predictor_kind: str = "auto",
              feature_names: Sequence[str] = FEATURE_NAMES) -> Calibration:
    scheds = [s for k in range(env.n_decisions + 1)
              for s in itertools.combinations(range(env.n_decisions), k)]
    sched_u = {s: execute(env, "s", constant(fixed_schedule(env, set(s))), eps).mean
               for s in scheds}
    # Tie-break toward FEWER deep calls: if two schedules score the same, the
    # cheaper one is the stronger baseline, and picking the dearer one would
    # hand the Governor an easier target.
    best_s = max(sched_u, key=lambda s: (sched_u[s], -len(s)))

    best_h, best_hu = None, -np.inf
    for f in HEURISTIC_FEATURES:
        vals = _feature_values(env, eps, f)
        for q in HEURISTIC_QUANTILES:
            thr = float(np.quantile(vals, q))
            u = execute(env, "h", constant(text_heuristic(env, f, thr)), eps).mean
            if u > best_hu:
                best_h, best_hu = (f, thr), u

    gains = np.array([env.realised_gain(it) for it in pool], float)
    if predictor_kind == "auto":
        # Model class chosen by CROSS-VALIDATED R2 ON CALIBRATION. A 40-item
        # smoke test had gbt at cv_R2 = -0.062 while a single observable feature
        # (sum_numeral_log10) correlated +0.718 with the gain -- the boosted
        # trees were overfitting a tiny sample, not reporting an absent signal.
        # Selecting the class by calibration CV is the standard fix and touches
        # no test data; the ablation reports every class regardless.
        cands = {}
        for k in ("gbt", "ridge", "mean"):
            p = ValuePredictor(kind=k)
            cands[k] = (p, p.fit(pool, gains, feature_names=feature_names))
        predictor_kind = max(cands, key=lambda k: cands[k][1].cv_r2)
        vp, rep = cands[predictor_kind]
    else:
        vp = ValuePredictor(kind=predictor_kind)
        rep = vp.fit(pool, gains, feature_names=feature_names)
    dp = OpportunityCostDP(vp.q_samples, n_items=env.n_decisions,
                           max_k=env.n_decisions)

    cal_u = {
        "H_all_cheap": execute(env, "c", constant(all_cheap(env)), eps).mean,
        "greedy": execute(env, "g", constant(greedy(env)), eps).mean,
        "fixed_best": sched_u[best_s],
    }
    return Calibration(
        best_schedule=best_s, schedule_utilities={str(k): v for k, v in sched_u.items()},
        heuristic_feature=best_h[0], heuristic_threshold=best_h[1],
        heuristic_utility=best_hu, predictor=vp, dp=dp, report=rep, gains=gains,
        cal_utilities=cal_u, base=max(cal_u, key=cal_u.get),
        predictor_kind=predictor_kind)


def evaluate_heldout(env: P4Env, cal: Calibration,
                     eps: Sequence[int]) -> tuple[dict[str, PolicyResult], list]:
    trace: list = []
    factories = {
        "H_all_cheap": constant(all_cheap(env)),
        "fixed_best": constant(fixed_schedule(env, set(cal.best_schedule))),
        "fixed_all_deep": constant(fixed_schedule(env, set(range(env.n_decisions)))),
        "greedy": constant(greedy(env)),
        "heuristic": constant(text_heuristic(env, cal.heuristic_feature,
                                             cal.heuristic_threshold)),
        "GOVERNOR": constant(governor(env, cal.predictor, cal.dp, trace)),
        "oracle": lambda e: clairvoyant(env, e),
    }
    return {k: execute(env, k, f, eps) for k, f in factories.items()}, trace


PREREG = "PREREGISTRATION-phase4-nemotron.md"


def summarise(R: dict[str, PolicyResult], cal: Calibration, env: P4Env,
              trace: list, commit: str = "", froze_commit: str | None = None,
              selection_item_ids: Sequence[str] | None = None,
              evaluation_item_ids: Sequence[str] | None = None) -> dict:
    """Callers MUST name the items the model was fitted on and the items it is
    scored on. Leaving them out makes `split_leakage` red, which is the point:
    silence about a split is not evidence of a clean one."""
    M = {k: r.metrics(env.budget) for k, r in R.items()}
    deltas = {b: paired_ci(R["GOVERNOR"].U, R[b].U) for b in BASELINES}
    den = M["oracle"]["U"] - M[cal.base]["U"]
    headroom = ((M["GOVERNOR"]["U"] - M[cal.base]["U"]) / den
                if abs(den) > 1e-9 else float("nan"))
    ev = {
        "gov_utils": R["GOVERNOR"].U, "greedy_utils": R["greedy"].U,
        "gov_calls": [sum(m == DEEP for m in ms) for ms in R["GOVERNOR"].modes],
        "greedy_calls": [sum(m == DEEP for m in ms) for ms in R["greedy"].modes],
        "decisions_by_state": R["GOVERNOR"].modes,
        "feature_names": list(FEATURE_NAMES),
        "answered_rate": M["GOVERNOR"]["answered_rate"],
        "utility": M["GOVERNOR"]["U"],
        "scored_via_executor": True,
        "decisions": [r["mode"] for r in trace],
        "cell_ids": [f"m{env.n_decisions - r['t']}_k{r['k']}" for r in trace],
        # The real evidence: the commit that froze the selection rules must
        # predate the commit that produced these numbers.
        "froze_commit": froze_commit if froze_commit is not None else file_commit(PREREG),
        "heldout_commit": commit,
        **token_evidence(R["GOVERNOR"], env),
    }
    if selection_item_ids is not None and evaluation_item_ids is not None:
        ev["selection_item_ids"] = list(selection_item_ids)
        ev["evaluation_item_ids"] = list(evaluation_item_ids)
    else:
        ev["selection_item_ids"] = []
        ev["evaluation_item_ids"] = [it.item_id for e in env.episodes for it in e]
    traps = run_trap_checks(ev)
    traps["secret_scan"] = secret_scan()
    red = [n for n, (ok, _) in traps.items() if not ok]
    primary = deltas[cal.base]
    return {"metrics": M, "deltas": deltas, "headroom": headroom,
            "traps": traps, "red": red, "primary": primary,
            "passed": bool(primary["beats"] and not red)}
