"""Second task family: uniqueness, partial credit, and no leakage.

The first family shipped a silent 50% mislabelling of its multi-op expression
items until a test recomputed every answer. The equivalent risk here is a puzzle
with two valid arrangements, which would score a correct answer as wrong. So
uniqueness is re-verified from the PROMPT TEXT, not trusted from the generator.
"""
from __future__ import annotations

import numpy as np
import pytest

from governor.harness.traps import oracle_leakage, progress_as_cognition
from governor.phase4.puzzles import (
    FEATURE_NAMES, NAMES, _solutions, feature_vector, features, grade,
    is_correct, make_pool, parse_assignment, pool_stats,
)


@pytest.fixture(scope="module")
def pool():
    return make_pool(seed=77, n=120)


def _split(body: str) -> list[str]:
    import re
    return [c.strip() for c in re.split(r"\(\d+\)", body) if c.strip()]


def test_every_puzzle_has_exactly_one_solution_and_it_is_the_recorded_one(pool):
    """Re-solved from the TEXT, so a generator bug cannot hide behind itself."""
    for p in pool:
        names = NAMES[:p.n_entities]
        sols = _solutions(_split(p.prompt.split("\n")[1]), names, limit=3)
        assert len(sols) == 1, (p.item_id, len(sols))
        assert sols[0] == p.answer, p.item_id


def test_puzzles_span_a_difficulty_range(pool):
    s = pool_stats(pool)
    assert set(s["n_entities"]) == {3, 4, 5}
    assert all(v > 0 for v in s["n_entities"].values()), s
    clues = np.array([p.n_clues for p in pool], float)
    assert clues.std() > 0.5, "every puzzle has the same clue count"


def test_reward_is_partial_credit_not_binary(pool):
    """The reward rule that distinguishes this family from the first."""
    p = pool[0]
    names = list(p.answer)
    full = ", ".join(f"{k}={v}" for k, v in p.answer.items())
    assert grade(p, full) == pytest.approx(1.0)
    swapped = dict(p.answer)
    a, b = names[0], names[1]
    swapped[a], swapped[b] = swapped[b], swapped[a]
    partial = ", ".join(f"{k}={v}" for k, v in swapped.items())
    g = grade(p, partial)
    assert 0.0 < g < 1.0, g
    assert g == pytest.approx((len(names) - 2) / len(names))
    assert not is_correct(p, partial)


def test_truncated_reasoning_scores_zero_not_lucky_fragments(pool):
    """An unterminated <think> means no answer was produced. Scoring whatever
    looks like an assignment inside a half-finished trace turns a starvation
    curve into a competence curve."""
    p = pool[0]
    full = ", ".join(f"{k}={v}" for k, v in p.answer.items())
    assert grade(p, f"<think>\nLet me try {full}") == 0.0
    assert grade(p, f"<think>\nthinking\n</think>\n{full}") == pytest.approx(1.0)
    assert grade(p, "") == 0.0
    assert grade(p, None) == 0.0


def test_later_assignments_win(pool):
    p = pool[0]
    full = ", ".join(f"{k}={v}" for k, v in p.answer.items())
    assert grade(p, "Ada=99, Bo=99\nActually: " + full) == pytest.approx(1.0)


def test_features_are_structural_and_pass_the_leakage_traps():
    ok, detail = oracle_leakage(FEATURE_NAMES)
    assert ok, detail
    ok, detail = progress_as_cognition(FEATURE_NAMES)
    assert ok, detail


def test_features_cannot_recover_the_arrangement(pool):
    """Two puzzles with the same structure must look identical to the features
    even when their answers differ -- otherwise the 'observable' encoding is a
    channel for the solution."""
    seen: dict[tuple, list] = {}
    for p in pool:
        seen.setdefault(tuple(feature_vector(p.prompt)), []).append(p)
    collided = [v for v in seen.values() if len(v) > 1]
    assert collided, "no two puzzles share a feature vector; test is vacuous"
    assert any(a.answer != b.answer for v in collided for a, b in zip(v, v[1:])), (
        "identical features always imply identical answers -- features leak")


def test_feature_names_match_the_vector():
    p = make_pool(seed=5, n=1)[0]
    assert feature_vector(p.prompt).shape == (len(FEATURE_NAMES),)
    assert set(features(p.prompt)) == set(FEATURE_NAMES)


def test_pools_from_different_seeds_are_disjoint():
    a = {p.prompt for p in make_pool(seed=10, n=60)}
    b = {p.prompt for p in make_pool(seed=11, n=60)}
    assert len(a & b) == 0


def test_parse_assignment_handles_realistic_replies():
    assert parse_assignment("Ada=1, Bo=2, Cyd=3") == {"Ada": 1, "Bo": 2, "Cyd": 3}
    assert parse_assignment("Ada = 1 , Bo = 2") == {"Ada": 1, "Bo": 2}
    assert parse_assignment("no assignment here") == {}
