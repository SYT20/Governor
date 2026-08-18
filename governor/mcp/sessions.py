"""Session layer behind the MCP tools.

ONE IMPLEMENTATION, NOT TWO. Every decision here comes from the same
`governor.phase4.policies.governor` the experiments use, and every action goes
through `governor.execution.executor.ActionExecutor`, which is itself asserted trace-identical
to the frozen `run_episode`. A plugin with its own copy of the control loop
would drift from the thing that was validated, and the drift would be invisible
because both would keep working.

THE GATEKEEPER APPLIES HERE TOO. Starting a Phase 4R session calls
`require_gate_passed()`. A tool call is not a loophole around the sequential
protocol.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from governor.execution.executor import ActionExecutor
from governor.harness.ledger import git_commit
from governor.phase4.collect import ResponseCache
from governor.phase4.config import CAL_POOL_SEED, ENGINES, PROMPT_CAP
from governor.phase4.env import CHEAP, DEEP, P4Env, make_episodes
from governor.phase4.pipeline import calibrate
from governor.phase4.policies import (
    affordable_upgrades, all_cheap, fixed_schedule, governor, greedy,
    text_heuristic,
)
from governor.phase4.split import filter_evaluation, filter_selection
from governor.phase4.tasks import make_pool


class FamilyUnavailable(RuntimeError):
    pass


@dataclass(slots=True)
class Session:
    session_id: str
    family: str
    env: Any
    execution: ActionExecutor
    state: Any
    budget: float
    spent: float = 0.0
    step: int = 0
    done: bool = False
    utility: float = 0.0
    calibration: Any = None
    policy: Any = None
    statemgr: dict = field(default_factory=dict)
    log: list = field(default_factory=list)
    started: float = field(default_factory=time.time)


_SESSIONS: dict[str, Session] = {}


# -- families -------------------------------------------------------------------

def _phase4r_env(n_episodes: int = 4, evaluation: bool = False) -> tuple:
    """Phase 4R, GATED. Raises unless the held-out ceiling gate has passed."""
    from governor.phase4.gatekeeper import require_gate_passed
    require_gate_passed()
    cfg = ENGINES["qwen"]
    cache = ResponseCache(Path(cfg["cache"]), model=cfg["model"],
                          provider=cfg["provider"])
    pool = [i for i in make_pool(CAL_POOL_SEED, 400)
            if cache.get(i, 300) and cache.get(i, 700)]
    items = (filter_evaluation if evaluation else filter_selection)(pool)
    n = min(n_episodes, len(items) // 6)
    if n < 1:
        raise FamilyUnavailable("phase4r has too few cached items")
    env = P4Env(cache, make_episodes(items, n, 7, n_items=6), 300, 700,
                2868.0, PROMPT_CAP)
    cal = calibrate(env, filter_selection(pool), list(range(n)))
    return env, cal


def _synthetic_env(n_episodes: int = 4, seed: int = 0) -> tuple:
    """Deterministic local family. No network, no quota, always available.

    This is what the MCP smoke test runs against, so the harness can be
    exercised end to end on a machine with no API key at all.
    """
    from governor.phase4.collect import CallRecord
    from governor.phase4.tasks import features
    import tempfile
    tmp = Path(tempfile.mkdtemp()) / "mcp.sqlite"
    cache = ResponseCache(tmp, model="synthetic")
    pool = make_pool(seed=4242, n=n_episodes * 4 + 40)
    for it in pool:
        needs = features(it.prompt)["numerals"] >= 3
        for mt in (300, 700):
            ok = (not needs) or mt == 700
            used = int(mt * (0.9 if mt == 300 else 0.75))
            cache.put(it, mt, CallRecord(
                it.item_id, mt, str(it.answer) if ok else "0",
                "stop" if ok else "length", 60, used, used - 10, 60 + used,
                0.01), attempts=1)
    cal_items, test_items = pool[40:], pool[:40]
    env = P4Env(cache, make_episodes(cal_items, n_episodes, 3), 300, 700,
                4 * (PROMPT_CAP + 300) + 1 * 400.0, PROMPT_CAP)
    cal_env = P4Env(cache, make_episodes(test_items, 10, 11), 300, 700,
                    env.budget, PROMPT_CAP)
    cal = calibrate(cal_env, test_items, list(range(10)))
    return env, cal


FAMILIES = {"synthetic": _synthetic_env, "phase4r": _phase4r_env}


# -- tools ----------------------------------------------------------------------

def governor_start(family: str = "synthetic", episode: int = 0,
                   **kw) -> dict:
    if family not in FAMILIES:
        raise FamilyUnavailable(f"unknown family {family!r}; "
                                f"available: {sorted(FAMILIES)}")
    env, cal = FAMILIES[family](**kw)
    sid = uuid.uuid4().hex[:12]
    s = Session(session_id=sid, family=family, env=env, execution=ActionExecutor(env),
                state=env.reset(episode), budget=env.budget, calibration=cal,
                policy=governor(env, cal.predictor, cal.dp))
    _SESSIONS[sid] = s
    obs = env.observe(s.state)
    return {"session_id": sid, "family": family, "episode": episode,
            "n_decisions": env.n_decisions, "budget": env.budget,
            "modes": env.modes(), "commit": git_commit(),
            "predictor": {"kind": cal.predictor_kind,
                          "cv_r2": round(cal.report.cv_r2, 4)},
            "observation": _obs(obs)}


def _obs(o: dict) -> dict:
    """Observable state only. The prompt is trimmed for transport, never the
    features -- and hidden axes are not present to begin with."""
    return {"t": o["t"], "items_left": o["items_left"],
            "prompt": o.get("prompt", "")[:400],
            "features": {k: round(v, 4) for k, v in o.get("features", {}).items()},
            "history": o.get("history", [])}


def _require(sid: str) -> Session:
    if sid not in _SESSIONS:
        raise KeyError(f"no session {sid!r}")
    return _SESSIONS[sid]


def governor_next(session_id: str) -> dict:
    """The Governor's decision for the current item. Decides; does not execute."""
    s = _require(session_id)
    if s.done:
        return {"session_id": session_id, "done": True, "action": None}
    o = s.env.observe(s.state)
    left = s.budget - s.spent
    m = o["items_left"]
    k = affordable_upgrades(s.env, left, m)
    q = (s.calibration.predictor.predict_one(o["features"])
         if o.get("features") else 0.0)
    thr = s.calibration.dp.threshold(m, k)
    action = s.policy(o, left)
    return {"session_id": session_id, "t": o["t"], "action": action,
            "predicted_gain": round(float(q), 5),
            "opportunity_cost": (None if thr == float("inf") else round(thr, 5)),
            "affordable_upgrades": int(k), "budget_left": round(left, 1),
            "reason": ("infeasible" if k <= 0 else
                       "gain >= opportunity cost" if action == DEEP else
                       "reserve budget for a better slot")}


def ares_execute(session_id: str, action: str) -> dict:
    """Execute one action through ActionExecutor. The ONLY way a session advances."""
    s = _require(session_id)
    if s.done:
        raise RuntimeError("episode already finished")
    left = s.budget - s.spent
    r = s.execution.execute(action, s.state, left)
    if not r.ok:
        return {"session_id": session_id, "ok": False, "error": r.error,
                "budget_left": round(left, 1)}
    s.state, s.step = r.state, s.step + 1
    s.spent += r.consumed["tokens"]
    s.utility = r.utility
    s.done = s.step >= s.env.n_decisions
    s.log.append({"t": s.step - 1, "action": action,
                  "cost": r.consumed["tokens"], "utility": r.utility})
    return {"session_id": session_id, "ok": True, "action": action,
            "consumed_tokens": r.consumed["tokens"],
            "spent": round(s.spent, 1), "budget_left": round(s.budget - s.spent, 1),
            "utility": round(r.utility, 4), "done": s.done,
            "observation": _obs(r.observation)}


def governor_status(session_id: str) -> dict:
    s = _require(session_id)
    return {"session_id": session_id, "family": s.family, "step": s.step,
            "n_decisions": s.env.n_decisions, "done": s.done,
            "utility": round(s.utility, 4), "spent": round(s.spent, 1),
            "budget": s.budget, "log": s.log,
            "deep_calls": sum(1 for e in s.log if e["action"] == DEEP),
            "wall_s": round(time.time() - s.started, 2)}


def budget_status(session_id: str) -> dict:
    s = _require(session_id)
    left = s.budget - s.spent
    return {"session_id": session_id, "budget": s.budget,
            "spent": round(s.spent, 1), "remaining": round(left, 1),
            "utilization": round(s.spent / s.budget, 4),
            "cap_cheap": s.env.cap(CHEAP), "cap_deep": s.env.cap(DEEP),
            "affordable_upgrades": int(affordable_upgrades(
                s.env, left, max(s.env.n_decisions - s.step, 0))),
            "charged": "measured usage.total_tokens, never nominal"}


def m2_reason(session_id: str, budget: int) -> dict:
    """Call the deep arm on the CURRENT item at an explicit token budget.

    Goes through the same cache the environment reads, so a tool call and an
    experiment see identical responses for identical inputs.
    """
    from governor.phase4.collect import outcome
    s = _require(session_id)
    if s.done:
        raise RuntimeError("episode already finished")
    if budget not in (s.env.tokens[CHEAP], s.env.tokens[DEEP]):
        raise ValueError(f"budget must be one of "
                         f"{sorted(set(s.env.tokens.values()))}")
    item = s.state.items[s.state.t]
    o = outcome(s.env.cache, item, budget, s.env.grade)
    return {"session_id": session_id, "requested_budget": budget,
            "total_tokens": o["total_tokens"],
            "reasoning_tokens": o["reasoning_tokens"],
            "answer_tokens": o["answer_tokens"],
            "answered": bool(o["answered"]), "starved": bool(o["starved"]),
            "finish_reason": o["finish_reason"],
            "latency_s": round(o["latency_s"], 3)}


def graft_get_state(session_id: str) -> dict:
    """Cognitive state: observable only, and correctness is deliberately absent."""
    s = _require(session_id)
    o = s.env.observe(s.state)
    return {"session_id": session_id,
            "task_context": {"family": s.family, "n_decisions": s.env.n_decisions},
            "observations": _obs(o),
            "reasoning_history": [e for e in s.log],
            "actions": [e["action"] for e in s.log],
            "outcomes_observable": o.get("history", []),
            "uncertainty": _uncertainty(s, o),
            "remaining_budget": round(s.budget - s.spent, 1),
            "user_slots": s.statemgr,
            "note": "correctness is not here; nothing reveals it at run time"}


def _uncertainty(s: Session, o: dict) -> dict:
    if not o.get("features"):
        return {}
    q = float(s.calibration.predictor.predict_one(o["features"]))
    spread = float(np.std(s.calibration.predictor.q_samples)) or 0.0
    return {"predicted_gain": round(q, 5),
            "calibration_spread": round(spread, 5),
            "cv_r2": round(s.calibration.report.cv_r2, 4)}


def graft_update_state(session_id: str, key: str, value: Any) -> dict:
    """Write to a scratch slot. Deliberately cannot touch the controller state.

    Env 5 manufactured a +0.035 'cognitive' effect from a progress counter filed
    as cognition. A writable memory that fed the allocator would let that happen
    through a tool call, so this writes to an isolated dict the policy never
    reads.
    """
    s = _require(session_id)
    s.statemgr[str(key)] = value
    return {"session_id": session_id, "user_slots": s.statemgr,
            "note": "scratch only; the allocator does not read this"}


def reset_sessions() -> None:
    _SESSIONS.clear()
