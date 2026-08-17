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

from governor.phase4 import puzzles as _puz
from governor.phase4 import tasks as _arith


@dataclass(frozen=True, slots=True)
class Family:
    name: str
    features: Callable[[str], dict]
    feature_names: tuple[str, ...]
    system_prompt: str
    grade: Callable | None = None          # None = binary exact match

    def vector(self, prompt: str, names=None) -> np.ndarray:
        f = self.features(prompt)
        return np.array([f[k] for k in (names or self.feature_names)], float)


ARITHMETIC = Family(
    name="arithmetic",
    features=_arith.features,
    feature_names=_arith.FEATURE_NAMES,
    system_prompt=_arith.SYSTEM_PROMPT,
    grade=None)

PUZZLES = Family(
    name="puzzles",
    features=_puz.features,
    feature_names=_puz.FEATURE_NAMES,
    system_prompt=_puz.SYSTEM_PROMPT_PUZZLE,
    grade=_puz.grade)

FAMILIES = {f.name: f for f in (ARITHMETIC, PUZZLES)}
