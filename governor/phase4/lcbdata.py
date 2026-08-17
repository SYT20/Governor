"""LiveCodeBench sample-allocation family.

The allocation unit is the NUMBER OF SAMPLES a problem gets, k in 1..10, not a
token budget. Utility at k is "did any of the first k samples pass its tests",
cost is the summed generation tokens of those k samples. Both come from
LiveCodeBench's published submissions, so nothing is generated here.

Features are read from the PROBLEM STATEMENT plus the contest metadata that
ships with the benchmark and is available before any sample is drawn. `difficulty`
is a platform label, not a property of the model's attempts, so using it is not
outcome leakage -- but it is a human annotation the other families did not have,
so it is carried as its own feature and can be ablated.
"""
from __future__ import annotations

import re

import numpy as np

FEATURE_NAMES = ("chars", "lines", "words_n", "has_starter", "difficulty_ord",
                 "is_codeforces", "is_leetcode", "is_atcoder", "n_constraints",
                 "max_bound_log10", "n_examples", "n_digits")

_DIFF = {"easy": 0.0, "medium": 1.0, "hard": 2.0}
_NUM = re.compile(r"\d+")
_POW = re.compile(r"10\^\{?(\d+)")


def features(prompt: str, difficulty: str = "", platform: str = "",
             starter: str = "") -> dict[str, float]:
    pw = [int(x) for x in _POW.findall(prompt)] or [0]
    return {
        "chars": float(len(prompt)),
        "lines": float(prompt.count("\n") + 1),
        "words_n": float(len(prompt.split())),
        "has_starter": float(bool((starter or "").strip())),
        "difficulty_ord": _DIFF.get(str(difficulty).lower(), 1.0),
        "is_codeforces": float(platform == "codeforces"),
        "is_leetcode": float(platform == "leetcode"),
        "is_atcoder": float(platform == "atcoder"),
        "n_constraints": float(prompt.lower().count("constraint")
                               + prompt.count("\\le") + prompt.count("<=")),
        "max_bound_log10": float(max(pw)),
        "n_examples": float(prompt.lower().count("example")),
        "n_digits": float(sum(c.isdigit() for c in prompt[:4000])),
    }


def feature_vector(prompt, difficulty="", platform="", starter="",
                   names=FEATURE_NAMES) -> np.ndarray:
    f = features(prompt, difficulty, platform, starter)
    return np.array([f[k] for k in names], float)


def split_mask(qids) -> np.ndarray:
    """Frozen split by a hash of the question id -- deterministic, and it does
    not depend on file order or on how many problems happen to be loaded."""
    import hashlib
    return np.array([int(hashlib.sha256(str(q).encode()).hexdigest()[:8], 16) % 2 == 0
                     for q in qids])
