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


def episode_budget(low: int, high: int) -> int:
    """4*cap(LOW) + 2*(cap(HIGH) - cap(LOW)); cap(m) = PROMPT_CAP + m.

    Worst-case room for exactly two upgrades out of four items. Because calls
    are charged what they USE and reserved at their CAP, a policy will often
    afford more than two -- that slack is measured and reported, not assumed
    away, and the robustness sweep varies this number.
    """
    cap_lo, cap_hi = PROMPT_CAP + low, PROMPT_CAP + high
    return int(4 * cap_lo + 2 * (cap_hi - cap_lo))


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
