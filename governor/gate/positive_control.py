"""Gate 0B — positive control. Adaptive beats EVERY constant schedule by
construction, so the gate machinery can be validated before it is trusted.

Without this, a buggy gate and a degenerate environment produce identical
output and every rejection is unfalsifiable. This is the only guard in the
protocol against a FALSE NEGATIVE.

Construction: 4 decisions, budget affords 2 deep calls. A signal revealed at
t=1 says whether the valuable deep call is at t=2 or t=3. Deep is worth +0.30
there and -0.05 anywhere else. No constant schedule can be right for both
signal values; an adaptive policy reading the signal is right always.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEEP, CHEAP = "M2", "H"
GAIN, PENALTY, COST = 0.30, -0.05, 1.0


@dataclass(slots=True)
class State:
    t: int
    signal: int          # -1 until revealed at t=1; then 2 or 3
    payoff: float
    target: int


class PositiveControlEnv:
    n_decisions = 4

    def __init__(self, seed: int = 0, n: int = 400):
        rng = np.random.default_rng(seed)
        self.targets = rng.choice([2, 3], size=n)

    def modes(self):
        return [CHEAP, DEEP]

    def reset(self, ep: int) -> State:
        return State(t=0, signal=-1, payoff=0.0, target=int(self.targets[ep]))

    def observe(self, s: State) -> dict:
        return {"t": s.t, "signal": s.signal}

    def step(self, s: State, mode: str):
        pay = 0.0
        if mode == DEEP:
            pay = GAIN if s.t == s.target else PENALTY
        # the signal becomes observable only AFTER the t=1 decision
        sig = s.target if s.t >= 1 else -1
        return State(t=s.t + 1, signal=sig, payoff=s.payoff + pay,
                     target=s.target), (COST if mode == DEEP else 0.0)

    def utility(self, s: State) -> float:
        return s.payoff
