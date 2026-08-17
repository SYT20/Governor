"""Phase 4 environment: four LLM items, one shared token budget.

SEQUENTIAL ARRIVAL. Items are seen one at a time. The policy never gets to look
at all four and pick the best two — that would be offline sorting, not
allocation under uncertainty, and it is not the situation a deployed controller
is in. Everything hard about the problem lives in this constraint.

RESOURCE SEMANTICS, MEASURED NOT NOMINAL.
  charge      = the provider's reported `usage.total_tokens` for the call
  feasibility = worst case, `cap(mode) = PROMPT_CAP + max_tokens(mode)`

The two differ, deliberately. A policy must reserve what a call COULD cost, and
is charged what it DID cost. That gap is real — it is why budget utilisation is
a reported metric rather than an assumption — and it is identical for every
policy. Charging the cap would be nominal accounting, which the trap checks
forbid; ignoring the cap would let a policy overspend and the executor would
raise.

NO API CALLS HAPPEN HERE. Every (item, budget) response was collected once and
cached before any policy ran, so all policies read identical responses. `step`
is a lookup. If a response is missing the environment raises rather than
silently reaching for the network mid-episode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from governor.phase4.collect import ResponseCache, outcome
from governor.phase4.tasks import Item, features

CHEAP, DEEP = "H", "M2"


@dataclass(slots=True)
class P4State:
    t: int
    correct: int
    items: tuple[Item, ...]
    log: list[dict]


class P4Env:
    """Four items per episode; `modes` are two token budgets."""

    n_decisions = 4

    def __init__(self, cache: ResponseCache, episodes: Sequence[Sequence[Item]],
                 low: int, high: int, budget: float, prompt_cap: int = 128):
        self.cache = cache
        self.episodes = [tuple(e) for e in episodes]
        self.tokens = {CHEAP: int(low), DEEP: int(high)}
        self.prompt_cap = int(prompt_cap)
        self.budget = float(budget)
        assert all(len(e) == self.n_decisions for e in self.episodes)

    # -- contract ------------------------------------------------------------

    def modes(self) -> list[str]:
        return [CHEAP, DEEP]

    def cap(self, mode: str) -> float:
        """Worst-case charge for one call in this mode."""
        return float(self.prompt_cap + self.tokens[mode])

    def feasible(self, mode: str, budget_left: float, items_left: int) -> bool:
        """Affordable now AND leaves enough to answer every remaining item.

        The second half matters: a policy that spends its way into being unable
        to call the model at all on item 4 has not saved anything, it has
        scored zero on an item it could have attempted.
        """
        return budget_left >= self.cap(mode) + (items_left - 1) * self.cap(CHEAP)

    def reset(self, ep: int) -> P4State:
        return P4State(t=0, correct=0, items=self.episodes[ep], log=[])

    def observe(self, s: P4State) -> dict:
        """Everything the policy may see. TEXT ONLY plus position bookkeeping.

        `n_ops`, `scale` and `framing` exist on the Item and are deliberately
        not here. The prompt is here because a deployed system has it.
        """
        if s.t >= self.n_decisions:
            return {"t": s.t, "prompt": "", "features": {}, "items_left": 0}
        it = s.items[s.t]
        return {"t": s.t, "prompt": it.prompt, "features": features(it.prompt),
                "items_left": self.n_decisions - s.t}

    def step(self, s: P4State, mode: str):
        it = s.items[s.t]
        o = outcome(self.cache, it, self.tokens[mode])
        cost = float(o["total_tokens"])
        if cost > self.cap(mode):
            raise RuntimeError(
                f"{it.item_id} charged {cost} > cap {self.cap(mode)}: the "
                f"worst-case bound policies reserve against is wrong")
        s2 = P4State(t=s.t + 1, correct=s.correct + o["correct"],
                     items=s.items, log=[*s.log, {"mode": mode, **o}])
        return s2, cost

    def utility(self, s: P4State) -> float:
        return s.correct / self.n_decisions

    # -- analysis (never visible to a policy) --------------------------------

    def realised_gain(self, it: Item) -> int:
        """correct(HIGH) - correct(LOW) for one item, from the cached calls.

        The teacher signal and the oracle's input. Legitimate offline on the
        CALIBRATION split; on the test split it is only ever used to compute the
        oracle upper bound, never to make a decision the Governor also makes.
        """
        return (outcome(self.cache, it, self.tokens[DEEP])["correct"]
                - outcome(self.cache, it, self.tokens[CHEAP])["correct"])

    def episode_log(self, ep: int, policy) -> list[dict]:
        from governor.gate.executor import run_episode
        s = self.reset(ep)
        run_episode(self, policy, ep, self.budget)
        return s.log


def make_episodes(pool: Sequence[Item], n_episodes: int, seed: int,
                  n_items: int = 4) -> list[list[Item]]:
    """Group pool items into episodes.

    The grouping seed is SEPARATE from the pool seed, so fresh episode
    structure costs no API calls while fresh ITEMS require a new pool. Both are
    varied in the robustness sweep, and only the second is a real new draw.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pool))
    need = n_episodes * n_items
    if need > len(pool):
        raise ValueError(f"pool of {len(pool)} cannot make {n_episodes} "
                         f"disjoint episodes of {n_items}")
    return [[pool[int(i)] for i in idx[k * n_items:(k + 1) * n_items]]
            for k in range(n_episodes)]
