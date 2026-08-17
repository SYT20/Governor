"""Phase 4 task family: arithmetic chains with verifiable integer answers.

DIFFICULTY IS NOT INJECTED. Env 6 had a designed latent `hard` bit and a cue
flipped at a hand-set rate; the Governor's job was to invert a noise process I
had written. That is a fair test of the allocation architecture and a poor test
of anything else, because I chose the signal-to-noise ratio.

Here there is no `hard` bit. Items vary along generative axes (chain length,
magnitude, whether division is involved, how the problem is phrased), and
"difficulty" is whatever makes the model need more tokens -- a MEASURED property
of (item, engine), not a label. An item is hard for Nemotron if Nemotron gets it
wrong at a small budget and right at a large one. Nobody sets that rate.

FRAMING IS RANDOMIZED INDEPENDENTLY OF ARITHMETIC. The same chain is rendered
either as a bare expression or as a warehouse word problem, chosen by a separate
draw. This is what stops surface features from being a perfect difficulty
detector: text length varies for reasons unrelated to how hard the arithmetic
is, so a length-based heuristic is genuinely noisy without me choosing a flip
probability.

The generative axes (`n_ops`, `scale`, `framing`) are recorded for ANALYSIS ONLY
and are never exposed through `features()`. `oracle_leakage` checks the feature
names; the split here is what makes that check meaningful.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SYSTEM_PROMPT = ("Solve the problem. Reply with only the final integer, "
                 "no words, no units, no commas.")


@dataclass(frozen=True, slots=True)
class Item:
    item_id: str
    prompt: str
    answer: int
    # HIDDEN generative axes -- analysis only, never features
    n_ops: int
    scale: int
    framing: str
    expr_str: str = ""                 # nested, left-to-right, unambiguous
    steps: tuple[str, ...] = field(default=())


# -- generation ---------------------------------------------------------------

def _chain(rng: np.random.Generator):
    """Build an arithmetic chain evaluated STRICTLY LEFT TO RIGHT.

    The expression rendering is fully parenthesised. The first version emitted
    `438 * 266 - 31750 // 5 + 457`, which Python (and any competent solver)
    evaluates by precedence as 110615, while the word rendering of the same
    chain means 17408. The two framings of one item had different correct
    answers and the ground truth was right for only one of them -- a silent
    50% mislabelling of every 3-and-4-op expression item. Parenthesising makes
    the two renderings provably the same question.
    """
    n_ops = int(rng.integers(1, 5))          # 1..4 operations
    scale = int(rng.integers(0, 2))          # 0 small, 1 large

    a = int(rng.integers(12, 50) if scale == 0 else rng.integers(120, 500))
    b = int(rng.integers(11, 40) if scale == 0 else rng.integers(110, 400))
    v = a * b
    e_str = f"{a} * {b}"
    steps = [e_str]
    words = [f"A depot has {a} crates holding {b} bolts each."]

    if n_ops >= 2:
        c = int(rng.integers(1, max(2, v // 3)))
        v -= c
        e_str = f"({e_str}) - {c}"
        steps.append(f"- {c}")
        words.append(f"{c} of the bolts are defective and discarded.")
    if n_ops >= 3:
        d = int(rng.integers(3, 10))
        v //= d
        e_str = f"({e_str}) // {d}"
        steps.append(f"// {d}")
        words.append(f"The remaining bolts are packed into full boxes of {d} "
                     f"(any leftovers stay loose).")
    if n_ops >= 4:
        f_ = int(rng.integers(10, 999))
        v += f_
        e_str = f"({e_str}) + {f_}"
        steps.append(f"+ {f_}")
        words.append(f"{f_} more of those boxes arrive from another depot.")
    return v, n_ops, scale, e_str, steps, words


_QUESTION = {1: "How many bolts are there in total?",
             2: "How many usable bolts remain?",
             3: "How many full boxes are made?",
             4: "How many boxes are there in the end?"}


def make_item(rng: np.random.Generator, idx: int) -> Item:
    answer, n_ops, scale, e_str, steps, words = _chain(rng)
    framing = "expr" if rng.random() < 0.5 else "word"
    if framing == "expr":
        # The note deliberately avoids the "//" glyph. The first version wrote
        # "(// is integer division)" on EVERY expression item, which made the
        # has_intdiv feature a framing detector instead of a division detector.
        prompt = f"Compute: {e_str}\n(Division rounds down to a whole number.)"
    else:
        prompt = " ".join(words) + " " + _QUESTION[n_ops]
    return Item(item_id=f"i{idx:06d}", prompt=prompt, answer=answer,
                n_ops=n_ops, scale=scale, framing=framing,
                expr_str=e_str, steps=tuple(steps))


def make_pool(seed: int, n: int) -> list[Item]:
    """A pool of items. Episodes are groups of four drawn from a pool.

    Calibration and test use DISJOINT pools generated from different seeds, so
    no test item is ever seen during training -- the split is a property of the
    data, not of a shuffle I promise I did.
    """
    rng = np.random.default_rng(seed)
    return [make_item(rng, seed * 1_000_000 + i) for i in range(n)]


# -- observable features -------------------------------------------------------

_NUM = re.compile(r"\d+")
_OPWORDS = ("box", "boxes", "defective", "discarded", "remaining", "leftover",
            "arrive", "depot", "crates")

# Names are deliberately free of every string in traps.FORBIDDEN_FEATURES.
FEATURE_NAMES = (
    "chars", "words_n", "numerals", "max_numeral_log10", "sum_numeral_log10",
    "has_intdiv", "has_minus", "has_plus", "is_expr", "opwords", "lines",
)


def features(prompt: str) -> dict[str, float]:
    """Everything the Governor may see about an item. Derived from TEXT ONLY.

    `n_ops` is the thing that would make this easy and it is not here. What is
    here is what a deployed system actually has before it decides: the string.
    """
    nums = [int(x) for x in _NUM.findall(prompt)]
    lo = [np.log10(n + 1) for n in nums] or [0.0]
    return {
        "chars": float(len(prompt)),
        "words_n": float(len(prompt.split())),
        "numerals": float(len(nums)),
        "max_numeral_log10": float(max(lo)),
        "sum_numeral_log10": float(sum(lo)),
        "has_intdiv": float("//" in prompt or "full boxes" in prompt),
        "has_minus": float(" - " in prompt or "defective" in prompt),
        "has_plus": float(" + " in prompt or "arrive" in prompt),
        "is_expr": float(prompt.startswith("Compute:")),
        "opwords": float(sum(prompt.lower().count(w) for w in _OPWORDS)),
        "lines": float(prompt.count("\n") + 1),
    }


def feature_vector(prompt: str) -> np.ndarray:
    f = features(prompt)
    return np.array([f[k] for k in FEATURE_NAMES], float)


# -- answer checking -----------------------------------------------------------

_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"


def parse_answer(text: str | None) -> int | None:
    """Last integer in the reply, AFTER stripping an inline reasoning trace.

    Qwen on Groq emits its reasoning inside `<think>...</think>` in `content`
    rather than in a separate field. Taking the last integer of a TRUNCATED
    trace would score a half-finished intermediate result as the model's answer
    -- occasionally correct by luck, which is exactly the kind of noise that
    makes a starvation curve look like a competence curve. An unterminated
    `<think>` block means no answer was produced. That is a failure, not a zero.
    """
    if not text:
        return None
    if _THINK_OPEN in text:
        if _THINK_CLOSE not in text:
            return None                     # ran out of budget mid-thought
        text = text.split(_THINK_CLOSE, 1)[1]
    m = _NUM.findall(text.replace(",", "").replace("−", "-"))
    if not m:
        return None
    idx = text.replace(",", "").rfind(m[-1])
    neg = idx > 0 and text.replace(",", "")[idx - 1] == "-"
    return -int(m[-1]) if neg else int(m[-1])


def is_correct(item: Item, text: str | None) -> bool:
    return parse_answer(text) == item.answer


def pool_stats(pool: list[Item]) -> dict[str, Any]:
    return {
        "n": len(pool),
        "n_ops": {k: sum(1 for i in pool if i.n_ops == k) for k in (1, 2, 3, 4)},
        "framing": {k: sum(1 for i in pool if i.framing == k)
                    for k in ("expr", "word")},
        "scale": {k: sum(1 for i in pool if i.scale == k) for k in (0, 1)},
        "mean_chars": float(np.mean([len(i.prompt) for i in pool])),
    }
