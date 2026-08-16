"""Environment 6 — per-item difficulty with a noisy observable cue.

DESIGNED AGAINST THE NINE RECORDED FAILURES. Env 5 died because difficulty was a
property of the CONFIGURATION, so the optimal allocation collapsed to a constant
temporal rule ("never spend early"). Here difficulty is a property of the ITEM
and its POSITION IS RANDOM, so no fixed schedule can track it.

  4 items per episode, decided one at a time.
  Each item is hard with p=0.5, INDEPENDENTLY -- position carries no information.
  Before deciding, the agent sees a NOISY cue of that item's difficulty
  (flipped with probability CUE_NOISE). The cue is OBSERVABLE; true difficulty
  is not.

  H  (cheap, cost 0):  accuracy 0.90 easy / 0.50 hard
  M2 (deep,  cost 1):  accuracy 0.95 easy / 0.85 hard
  Budget = 2 deep calls for 4 items.

  Utility = fraction of items answered correctly.

WHY EACH FAILURE MODE IS BLOCKED
  #1 static schedule    difficulty position is uniform, so every constant
                        schedule has identical expected value by symmetry
  #2 regime leakage     there is no configuration id; every episode is one draw
                        from one distribution
  #3 progress variables t carries no information about difficulty
  #7 no allocation value the whole point of the design: WHICH items get the two
                        calls must vary per episode
  #9 constant rule      blocked by #1

WHAT COULD STILL FAIL. If CUE_NOISE is too high the cue is uninformative and
adaptive collapses to constant. If too low the problem is trivial. CUE_NOISE is
fixed at 0.15 here BEFORE the gate runs and is not tuned afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CHEAP, DEEP = "H", "M2"
P_HARD, CUE_NOISE = 0.5, 0.15
ACC = {(CHEAP, 0): 0.90, (CHEAP, 1): 0.50, (DEEP, 0): 0.95, (DEEP, 1): 0.85}
COST = {CHEAP: 0.0, DEEP: 1.0}


@dataclass(slots=True)
class S:
    t: int
    correct: int
    hard: np.ndarray      # hidden
    cue: np.ndarray       # observable
    roll: np.ndarray      # pre-drawn uniforms: outcome fixed per (item, mode)


class Env6:
    n_decisions = 4

    def __init__(self, seed: int = 0, n: int = 1000):
        rng = np.random.default_rng(seed)
        self.hard = (rng.random((n, 4)) < P_HARD).astype(int)
        flip = rng.random((n, 4)) < CUE_NOISE
        self.cue = np.where(flip, 1 - self.hard, self.hard)
        # COMMON RANDOM NUMBERS: one uniform per (episode, item), shared by both
        # modes, so H and M2 are compared on matched draws rather than on
        # independent noise. Without this the H-vs-M2 difference would carry
        # sampling variance that has nothing to do with the decision.
        self.roll = rng.random((n, 4))

    def modes(self):
        return [CHEAP, DEEP]

    def reset(self, ep: int) -> S:
        return S(t=0, correct=0, hard=self.hard[ep], cue=self.cue[ep],
                 roll=self.roll[ep])

    def observe(self, s: S) -> dict:
        return {"t": s.t, "cue": int(s.cue[s.t]) if s.t < 4 else -1}

    def step(self, s: S, mode: str):
        i = s.t
        ok = int(s.roll[i] < ACC[(mode, int(s.hard[i]))])
        return S(t=i + 1, correct=s.correct + ok, hard=s.hard, cue=s.cue,
                 roll=s.roll), COST[mode]

    def utility(self, s: S) -> float:
        return s.correct / 4.0
