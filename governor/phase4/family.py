"""A task family as a first-class object.

WHY THIS EXISTS. The second-family end-to-end test failed with
`KeyError: 'n_people'` because `ValuePredictor` imported family one's feature
extractor directly, and the environment did the same. Both worked perfectly and
both were specialised to arithmetic. That is exactly the defect a generalization
test is for: the architecture claim is that only the TASK changes, and it was
not true until the family became a parameter instead of an import.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from governor.phase4 import s1data as _s1
from governor.phase4 import puzzles as _puz
from governor.phase4 import tasks as _arith


@dataclass(frozen=True, slots=True)
class Family:
    name: str
    features: Callable[[str], dict]
    feature_names: tuple[str, ...]
    system_prompt: str
    grade: Callable | None = None          # None = binary exact match
    # Features the single-threshold heuristic baseline may use, and the one the
    # smoke fixture treats as "difficulty". Both were hardcoded to family one
    # until the second family hit KeyError on them.
    heuristic_features: tuple[str, ...] = ()
    difficulty_feature: str = ""

    def vector(self, prompt: str, names=None) -> np.ndarray:
        f = self.features(prompt)
        return np.array([f[k] for k in (names or self.feature_names)], float)


ARITHMETIC = Family(
    name="arithmetic",
    features=_arith.features,
    feature_names=_arith.FEATURE_NAMES,
    system_prompt=_arith.SYSTEM_PROMPT,
    grade=None,
    heuristic_features=("chars", "numerals", "sum_numeral_log10", "words_n"),
    difficulty_feature="sum_numeral_log10")

PUZZLES = Family(
    name="puzzles",
    features=_puz.features,
    feature_names=_puz.FEATURE_NAMES,
    system_prompt=_puz.SYSTEM_PROMPT_PUZZLE,
    grade=_puz.grade,
    heuristic_features=("n_clues", "n_people", "clues_ordering",
                        "clues_per_person"),
    difficulty_feature="n_clues")

S1MATH = Family(
    name="s1math",
    features=_s1.features,
    feature_names=_s1.FEATURE_NAMES,
    system_prompt="(external: s1 released generations)",
    grade=_s1.grade_passthrough,
    heuristic_features=("chars", "words_n", "latex_cmds", "n_equations"),
    difficulty_feature="chars")

FAMILIES = {f.name: f for f in (ARITHMETIC, PUZZLES, S1MATH)}
