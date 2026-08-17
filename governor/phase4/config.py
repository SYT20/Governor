"""Frozen Phase 4 constants. Imported by every script so there is exactly one
copy of each number, and changing one is a diff rather than a discrepancy.

The values here were fixed in PREREGISTRATION-phase4-nemotron.md before any
Phase 4 data existed.
"""
from __future__ import annotations

from governor.phase4.collect import GROQ, OPENROUTER

# -- curve / selection ---------------------------------------------------------
GRID = [300, 700, 1400, 2800]
SAT_TOL, MIN_GAP = 0.02, 0.15
PROMPT_CAP = 128
CURVE_N = 40

# -- splits --------------------------------------------------------------------
CAL_POOL_SEED, TEST_POOL_SEED = 1000, 20260817
CAL_GROUP_SEED, TEST_GROUP_SEED = 11, 22

# -- engines -------------------------------------------------------------------
ENGINES = {
    "nemotron": dict(provider=OPENROUTER, model="nvidia/nemotron-nano-9b-v2:free",
                     cache="results/p4_cache_nemotron.sqlite", workers=8, tpm=None),
    "qwen":     dict(provider=GROQ, model="qwen/qwen3.6-27b",
                     cache="results/p4_cache_qwen.sqlite", workers=6, tpm=8000),
    "gptoss":   dict(provider=GROQ, model="openai/gpt-oss-120b",
                     cache="results/p4_cache_gptoss.sqlite", workers=6, tpm=8000),
}
# Fixed a priori: the directive names nemotron, so nemotron wins any tie. Picking
# the engine with the largest measured gap would be selecting on the outcome.
PREFERENCE = ["nemotron", "qwen", "gptoss"]

HEURISTIC_FEATURES = ("chars", "numerals", "sum_numeral_log10", "words_n")
HEURISTIC_QUANTILES = (0.25, 0.4, 0.5, 0.6, 0.75)


TARGET_DEEP_CALLS = 2.0          # "about half the items can be upgraded"


def episode_budget(low: int, high: int) -> int:
    """SUPERSEDED by `budget_for_target_scarcity` (Amendment 1).

    Kept because E0001 recorded a selection made with it, and deleting the
    function would make that record unreproducible. It reserves each call at its
    CAP and assumed a call costs roughly its cap; at HIGH=2800 the engine stops
    after 817 of 2928 reserved tokens, so the budget it returns does not bind at
    all -- greedy upgrades every item and there is no allocation problem left to
    measure. See the amendment in PREREGISTRATION-phase4-nemotron.md.
    """
    cap_lo, cap_hi = PROMPT_CAP + low, PROMPT_CAP + high
    return int(4 * cap_lo + 2 * (cap_hi - cap_lo))


def budget_for_target_scarcity(env_factory, eps, low: int, high: int,
                               target: float = TARGET_DEEP_CALLS,
                               step: int = 50) -> tuple[int, list[dict]]:
    """Amendment 1: the smallest budget at which greedy realises `target` deep
    calls per episode ON CALIBRATION.

    Sets the SCARCITY of the resource, which is the experiment's independent
    variable. Computed on the calibration split before the test pool is touched;
    nothing about which policy wins enters it. The sweep is returned so the
    whole curve is recorded, not just the chosen point.
    """
    import numpy as np

    from governor.phase4.env import DEEP
    from governor.phase4.evaluate import constant, execute
    from governor.phase4.policies import greedy

    cap_lo, cap_hi = PROMPT_CAP + low, PROMPT_CAP + high
    grid = list(range(4 * cap_lo, 4 * cap_lo + 4 * (cap_hi - cap_lo) + step, step))
    sweep, chosen = [], None
    for b in grid:
        env = env_factory(float(b))
        res = execute(env, "greedy", constant(greedy(env)), eps)
        deep = float(np.mean([sum(m == DEEP for m in ms) for ms in res.modes]))
        sweep.append({"budget": b, "greedy_deep_calls": deep, "U": res.mean})
        if chosen is None and deep >= target:
            chosen = b
    return (chosen if chosen is not None else grid[-1]), sweep


def select_modes(acc: dict[int, float]) -> dict:
    """The frozen selection rule: HIGH = cheapest saturating budget, LOW = the
    dearest budget below it that is materially worse."""
    grid = sorted(acc)
    best = max(acc.values())
    high = next(b for b in grid if acc[b] >= best - SAT_TOL)
    lows = [b for b in grid if b < high and acc[b] <= acc[high] - MIN_GAP]
    low = max(lows) if lows else None
    out = {"high": high, "low": low, "acc_high": acc[high],
           "acc_low": acc[low] if low is not None else None,
           "gap": (acc[high] - acc[low]) if low is not None else None,
           "qualifies": low is not None}
    if out["qualifies"]:
        out["episode_budget"] = episode_budget(low, high)
    return out
