"""Execution and metrics for Phase 4.

One function runs a policy, and it runs it through the canonical executor. The
telemetry (reasoning tokens, answer tokens, latency, starvation) is recovered by
replaying the executed mode sequence against the same cached responses -- a
lookup, not a second execution, so there is no way for the scored trajectory to
differ from the executed one. Run I in this project failed precisely because
those were two different code paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from governor.gate.executor import run_episode
from governor.phase4.collect import outcome
from governor.phase4.env import DEEP, P4Env

PolicyFactory = Callable[[int], Callable[[dict, float], str]]


@dataclass(slots=True)
class PolicyResult:
    name: str
    U: np.ndarray
    modes: list[tuple[str, ...]]
    spent: np.ndarray
    calls: list[list[dict]] = field(default_factory=list)

    # -- headline ---------------------------------------------------------------
    @property
    def mean(self) -> float:
        return float(self.U.mean())

    @property
    def ci(self) -> tuple[float, float]:
        se = float(self.U.std(ddof=1) / np.sqrt(len(self.U)))
        return self.mean - 1.96 * se, self.mean + 1.96 * se

    def _per_ep(self, field_name: str) -> np.ndarray:
        return np.array([sum(c[field_name] for c in ep) for ep in self.calls],
                        float)

    def metrics(self, budget: float) -> dict:
        lo, hi = self.ci
        tok = self.spent
        n_items = float(np.mean([len(m) for m in self.modes]))
        return {
            "U": self.mean, "ci_lo": lo, "ci_hi": hi,
            "n_episodes": int(len(self.U)),
            "total_tokens_per_episode": float(tok.mean()),
            "total_tokens_all": float(tok.sum()),
            "reasoning_tokens_per_episode": float(self._per_ep("reasoning_tokens").mean()),
            "answer_tokens_per_episode": float(self._per_ep("answer_tokens").mean()),
            "prompt_tokens_per_episode": float(self._per_ep("prompt_tokens").mean()),
            "deep_calls_per_episode": float(np.mean(
                [sum(m == DEEP for m in ms) for ms in self.modes])),
            "latency_s_per_episode": float(self._per_ep("latency_s").mean()),
            "starvation_rate": float(self._per_ep("starved").mean() / n_items),
            "answered_rate": float(self._per_ep("answered").mean() / n_items),
            "budget_utilization": float(tok.mean() / budget),
            "utility_per_ktoken": 1000.0 * self.mean / max(float(tok.mean()), 1e-9),
        }


def execute(env: P4Env, name: str, factory: PolicyFactory,
            eps: Sequence[int]) -> PolicyResult:
    U, modes, spent, calls = [], [], [], []
    for e in eps:
        tr = run_episode(env, factory(e), e, env.budget)
        U.append(tr.utility)
        modes.append(tuple(tr.modes))
        spent.append(tr.spent)
        st = env.reset(e)
        per_call = []
        for m in tr.modes:
            per_call.append(outcome(env.cache, st.items[st.t], env.tokens[m]))
            st, _ = env.step(st, m)
        # The replay must reproduce the executed spend exactly, or the telemetry
        # describes a different trajectory than the one that was scored.
        assert abs(sum(c["total_tokens"] for c in per_call) - tr.spent) < 1e-6
        calls.append(per_call)
    return PolicyResult(name, np.array(U, float), modes,
                        np.array(spent, float), calls)


def constant(policy) -> PolicyFactory:
    """Wrap a state-independent policy so every policy has the same signature."""
    return lambda _e: policy


def paired_ci(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.asarray(a) - np.asarray(b)
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"mean": m, "lo": m - 1.96 * se, "hi": m + 1.96 * se,
            "beats": bool(m - 1.96 * se > 0), "loses": bool(m + 1.96 * se < 0)}


def token_evidence(res: PolicyResult, env: P4Env) -> dict:
    """Per-call (requested cap, actual use, charged) for the token trap."""
    requested, actual, charged = [], [], []
    for ms, ep_calls in zip(res.modes, res.calls):
        for m, c in zip(ms, ep_calls):
            requested.append(env.cap(m))
            actual.append(float(c["total_tokens"]))
            charged.append(float(c["total_tokens"]))
    return {"requested": requested, "actual_used": actual, "charged": charged}
