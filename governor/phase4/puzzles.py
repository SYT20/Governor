"""Step 11 — second task family: constraint-satisfaction seating puzzles.

DELIBERATELY UNLIKE THE FIRST FAMILY, on every axis the directive names:

    arithmetic chains          seating puzzles
    ------------------------   -----------------------------------------
    cue: numeral magnitude     cue: constraint count and constraint TYPE
    reward: exact integer      reward: fraction of slots placed correctly
    features: numeric scale    features: structural / categorical counts
    one answer token           an assignment of N entities to N positions

What is kept is exactly the machinery under test: the Governor interface, the
M2 interface, the executor, and the token accounting. If the architecture only
works when the reward is binary and the difficulty cue is "how big are the
numbers", it has not been shown to generalise -- it has been shown to fit one
generator.

PARTIAL CREDIT IS THE POINT. Under a binary reward the realised gain is in
{-1, 0, +1} and the value predictor is doing classification wearing a
regression's clothes. Here the gain is continuous, so the DP's expectation is
taken over a genuinely continuous distribution and the allocation problem has
finer structure than "which items flip".

Every puzzle is verified UNIQUELY SOLVABLE by brute force at generation time.
A puzzle with two solutions would score a correct answer as wrong, which is the
kind of silent mislabelling that the first family's operator-precedence bug
caused before a test caught it.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

NAMES = ("Ada", "Bo", "Cyd", "Dee", "Eli", "Fay")

SYSTEM_PROMPT_PUZZLE = (
    "Solve the seating puzzle. Reply with only the final assignment in the "
    "form NAME=SEAT, comma separated, e.g. 'Ada=1, Bo=2'. No explanation.")


@dataclass(frozen=True, slots=True)
class Puzzle:
    item_id: str
    prompt: str
    answer: dict[str, int]          # name -> seat
    n_entities: int                 # HIDDEN generative axes, analysis only
    n_clues: int
    clue_mix: tuple[int, int, int]  # (position, ordering, adjacency)

    # The env and cache only ever touch `.item_id` and `.prompt`.


# -- clue vocabulary -----------------------------------------------------------

def _clues_for(perm: dict[str, int], rng: np.random.Generator,
               names: tuple[str, ...]) -> list[tuple[str, str]]:
    """All true clues about this arrangement, as (kind, text)."""
    out: list[tuple[str, str]] = []
    for a, b in itertools.permutations(names, 2):
        if perm[a] < perm[b]:
            out.append(("ordering", f"{a} sits somewhere to the left of {b}."))
        if perm[a] == perm[b] - 1:
            out.append(("adjacency", f"{a} sits immediately to the left of {b}."))
    for a in names:
        out.append(("position", f"{a} sits in seat {perm[a]}."))
        wrong = [s for s in range(1, len(names) + 1) if s != perm[a]]
        s = int(rng.choice(wrong))
        out.append(("position", f"{a} does not sit in seat {s}."))
    rng.shuffle(out)
    return out


def _solutions(clues: list[str], names: tuple[str, ...], limit: int = 2) -> list[dict]:
    """Brute force. N <= 6, so 720 permutations at worst."""
    checks = [_compile(c, names) for c in clues]
    found = []
    for p in itertools.permutations(range(1, len(names) + 1)):
        cand = dict(zip(names, p))
        if all(f(cand) for f in checks):
            found.append(cand)
            if len(found) >= limit:
                break
    return found


_RE_LEFT_IMM = re.compile(r"^(\w+) sits immediately to the left of (\w+)\.$")
_RE_LEFT = re.compile(r"^(\w+) sits somewhere to the left of (\w+)\.$")
_RE_AT = re.compile(r"^(\w+) sits in seat (\d+)\.$")
_RE_NOT_AT = re.compile(r"^(\w+) does not sit in seat (\d+)\.$")


def _compile(clue: str, names: tuple[str, ...]):
    if (m := _RE_LEFT_IMM.match(clue)):
        a, b = m.groups()
        return lambda d: d[a] == d[b] - 1
    if (m := _RE_LEFT.match(clue)):
        a, b = m.groups()
        return lambda d: d[a] < d[b]
    if (m := _RE_AT.match(clue)):
        a, s = m.group(1), int(m.group(2))
        return lambda d: d[a] == s
    if (m := _RE_NOT_AT.match(clue)):
        a, s = m.group(1), int(m.group(2))
        return lambda d: d[a] != s
    raise ValueError(f"unparseable clue: {clue!r}")


# -- generation ----------------------------------------------------------------

def make_puzzle(rng: np.random.Generator, idx: int) -> Puzzle | None:
    n = int(rng.integers(3, 6))                      # 3..5 entities
    names = NAMES[:n]
    perm = dict(zip(names, rng.permutation(np.arange(1, n + 1)).tolist()))
    pool = _clues_for(perm, rng, names)

    # Add clues until the solution is unique, then STOP -- the last clue is what
    # makes it solvable, so the puzzle is never over-determined by construction.
    chosen: list[tuple[str, str]] = []
    for kind_text in pool:
        chosen.append(kind_text)
        if len(_solutions([t for _, t in chosen], names)) == 1:
            break
    else:
        return None                                  # never became unique
    sols = _solutions([t for _, t in chosen], names)
    if len(sols) != 1 or sols[0] != perm:
        return None

    mix = tuple(sum(1 for k, _ in chosen if k == want)
                for want in ("position", "ordering", "adjacency"))
    body = " ".join(f"({i + 1}) {t}" for i, (_, t) in enumerate(chosen))
    prompt = (f"{n} people ({', '.join(names)}) sit in seats numbered 1 to {n}, "
              f"left to right, one person per seat.\n{body}\n"
              f"Give the seat of every person.")
    return Puzzle(item_id=f"p{idx:06d}", prompt=prompt, answer=perm,
                  n_entities=n, n_clues=len(chosen), clue_mix=mix)


def make_pool(seed: int, n: int) -> list[Puzzle]:
    rng = np.random.default_rng(seed)
    out, idx = [], 0
    while len(out) < n:
        p = make_puzzle(rng, seed * 1_000_000 + idx)
        idx += 1
        if p is not None:
            out.append(p)
        if idx > 40 * n:
            raise RuntimeError("generator failing to produce unique puzzles")
    return out


# -- observable features --------------------------------------------------------

FEATURE_NAMES = (
    "n_people", "n_clues", "clues_position", "clues_ordering", "clues_adjacency",
    "clues_negative", "clues_per_person", "chars", "words_n",
)

_SEATS = re.compile(r"seats numbered 1 to (\d+)")
_NUMBERED = re.compile(r"\(\d+\)")


def features(prompt: str) -> dict[str, float]:
    """Structural counts read off the TEXT. No entity is named as a feature and
    the arrangement is not recoverable from any of these."""
    m = _SEATS.search(prompt)
    n_people = float(m.group(1)) if m else 0.0
    n_clues = float(len(_NUMBERED.findall(prompt)))
    return {
        "n_people": n_people,
        "n_clues": n_clues,
        "clues_position": float(prompt.count(" sits in seat ")),
        "clues_ordering": float(prompt.count(" somewhere to the left of ")),
        "clues_adjacency": float(prompt.count(" immediately to the left of ")),
        "clues_negative": float(prompt.count(" does not sit in seat ")),
        "clues_per_person": float(n_clues / n_people) if n_people else 0.0,
        "chars": float(len(prompt)),
        "words_n": float(len(prompt.split())),
    }


def feature_vector(prompt: str, names=FEATURE_NAMES) -> np.ndarray:
    f = features(prompt)
    return np.array([f[k] for k in names], float)


# -- grading --------------------------------------------------------------------

_ASSIGN = re.compile(r"([A-Z][a-z]+)\s*=\s*(\d+)")
_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"


def parse_assignment(text: str | None) -> dict[str, int]:
    """Last assignment block in the reply, after any reasoning trace.

    An unterminated `<think>` means the model ran out of budget mid-thought and
    produced no answer -- scored zero, not scored on whatever fragments of the
    trace happen to look like assignments.
    """
    if not text:
        return {}
    if _THINK_OPEN in text:
        if _THINK_CLOSE not in text:
            return {}
        text = text.split(_THINK_CLOSE, 1)[1]
    out: dict[str, int] = {}
    for name, seat in _ASSIGN.findall(text):
        out[name] = int(seat)              # later mentions win
    return out


def grade(item: Puzzle, text: str | None) -> float:
    """Fraction of people placed in the right seat. PARTIAL CREDIT.

    This is the reward rule that differs from family one. A reply that gets
    three of four seats right is genuinely three-quarters of the way there, and
    a controller allocating budget should see that.
    """
    got = parse_assignment(text)
    if not got:
        return 0.0
    n = len(item.answer)
    return sum(1 for k, v in item.answer.items() if got.get(k) == v) / n


def is_correct(item: Puzzle, text: str | None) -> bool:
    return grade(item, text) >= 1.0 - 1e-9


def pool_stats(pool: list[Puzzle]) -> dict[str, Any]:
    return {"n": len(pool),
            "n_entities": {k: sum(1 for p in pool if p.n_entities == k)
                           for k in (3, 4, 5)},
            "mean_clues": float(np.mean([p.n_clues for p in pool])),
            "mean_chars": float(np.mean([len(p.prompt) for p in pool]))}
