"""The Phase 4 task family must be verifiable, and its features must not leak."""
from __future__ import annotations

import numpy as np

from governor.harness.traps import oracle_leakage, progress_as_cognition
from governor.phase4.tasks import (
    FEATURE_NAMES, Item, _chain, feature_vector, features, is_correct,
    make_item, make_pool, parse_answer,
)


def test_answers_are_recomputable_from_the_chain():
    """Every answer must be reproducible from the printed expression, or the
    ground truth is an assertion rather than a fact."""
    for it in make_pool(seed=1, n=300):
        assert eval(it.expr_str) == it.answer, it  # noqa: S307


def test_prompt_contains_every_operand():
    for it in make_pool(seed=2, n=100):
        if it.framing == "expr":
            assert it.prompt.startswith("Compute:")
        for tok in " ".join(it.steps).split():
            if tok.isdigit():
                assert tok in it.prompt


def test_both_framings_of_a_chain_have_the_same_answer():
    """The defect this caught: precedence made the expression rendering mean
    something different from the word rendering for every 3-and-4-op item."""
    import numpy as np
    for seed in range(40):
        rng = np.random.default_rng(seed)
        answer, n_ops, _, e_str, _, _ = _chain(rng)
        assert eval(e_str) == answer, (e_str, answer)      # noqa: S307
        assert n_ops < 3 or "(" in e_str, "multi-op chain left unparenthesised"


def test_framing_is_independent_of_complexity():
    """The property that makes surface features a NOISY cue. If framing tracked
    n_ops, text length would be a perfect difficulty detector and the allocation
    problem would be trivial."""
    pool = make_pool(seed=3, n=2000)
    ops = np.array([i.n_ops for i in pool])
    expr = np.array([i.framing == "expr" for i in pool], float)
    r = float(np.corrcoef(ops, expr)[0, 1])
    assert abs(r) < 0.08, f"framing correlates with complexity: r={r:+.3f}"


def test_length_is_an_imperfect_complexity_signal():
    """Length must be informative but far from deterministic -- otherwise the
    text heuristic already solves the task and the Governor has nothing to add."""
    pool = make_pool(seed=4, n=2000)
    ops = np.array([i.n_ops for i in pool], float)
    chars = np.array([len(i.prompt) for i in pool], float)
    r = float(np.corrcoef(ops, chars)[0, 1])
    assert 0.15 < r < 0.95, f"length-complexity correlation {r:+.3f} out of range"


def test_features_are_text_only_and_pass_the_leakage_traps():
    ok, detail = oracle_leakage(FEATURE_NAMES)
    assert ok, detail
    ok, detail = progress_as_cognition(FEATURE_NAMES)
    assert ok, detail


def test_features_depend_only_on_the_prompt_string():
    """Two items with identical text must produce identical features, whatever
    their hidden axes -- the only way `features` cannot read them."""
    a = Item("x", "Compute: 12 * 13", 156, n_ops=1, scale=0, framing="expr")
    b = Item("y", "Compute: 12 * 13", 156, n_ops=4, scale=1, framing="word")
    assert features(a.prompt) == features(b.prompt)
    assert feature_vector(a.prompt).shape == (len(FEATURE_NAMES),)


def test_pools_from_different_seeds_are_disjoint():
    a = {i.item_id for i in make_pool(seed=10, n=200)}
    b = {i.item_id for i in make_pool(seed=11, n=200)}
    assert not (a & b)
    pa = {i.prompt for i in make_pool(seed=10, n=200)}
    pb = {i.prompt for i in make_pool(seed=11, n=200)}
    assert len(pa & pb) <= 2, "test pool overlaps calibration pool"


def test_answer_parsing():
    assert parse_answer("391") == 391
    assert parse_answer("The answer is 1,234.") == 1234
    assert parse_answer("") is None
    assert parse_answer(None) is None
    assert parse_answer("no digits here") is None
    it = make_item(np.random.default_rng(0), 0)
    assert is_correct(it, str(it.answer))
    assert not is_correct(it, str(it.answer + 1))
    assert not is_correct(it, "")
