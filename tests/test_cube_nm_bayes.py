"""Regression tests for the exact CUBE-NM posterior.

The myopic-vs-non-myopic result rests entirely on this likelihood being the
generator's. A silent drift here would not raise -- it would quietly hand the
myopic arm a broken model and produce a confident wrong conclusion. These tests
pin the properties that make the comparison meaningful.
"""

from __future__ import annotations

import numpy as np
import pytest

from governor.envs.cube_nm_bayes import CubeNMBayes
from governor.envs.cube_nm_repro import BLOCK_SIZE, N_LABELS, CubeNMRepro


@pytest.fixture(scope="module")
def ds():
    return CubeNMRepro(n_samples=800, seed=11)


@pytest.fixture(scope="module")
def bayes(ds):
    return CubeNMBayes(ds)


def test_hypothesis_grid_shape(ds, bayes):
    assert bayes.H == ds.n_contexts * N_LABELS == 40
    assert bayes.MU.shape == bayes.SD.shape == (40, ds.n_features)
    assert (bayes.SD > 0).all()


def test_informative_columns_are_tight(ds, bayes):
    """Exactly three block columns per hypothesis carry the low-noise code."""
    for c in (0, 3):
        for y in (0, 5, 7):
            h = c * N_LABELS + y
            tight = np.flatnonzero(
                bayes.SD[h, ds.n_contexts:] == ds.informative_feature_std)
            assert sorted(tight) == sorted((c * BLOCK_SIZE + (y + j) % BLOCK_SIZE)
                                           for j in range(3))


def test_posterior_is_normalised_and_recovers_context(ds, bayes):
    allc = list(range(ds.n_features))
    for i in range(30):
        logL = bayes.loglik_cols(ds.features[i], allc)
        p = np.exp(logL - logL.max())
        p /= p.sum()
        assert abs(bayes.label_posterior(logL).sum() - 1.0) < 1e-9
        pc = p.reshape(bayes.K, N_LABELS).sum(axis=1)
        assert int(np.argmax(pc)) == int(ds.context[i])


def test_empty_observation_is_uniform(bayes):
    py = bayes.label_posterior(np.zeros(bayes.H))
    assert np.allclose(py, 1.0 / N_LABELS)


def test_wrong_block_alone_is_chance(ds, bayes):
    """The non-selected blocks must carry no label information at all."""
    hits = 0
    for i in range(len(ds.labels)):
        w = (int(ds.context[i]) + 1) % ds.n_contexts
        s = ds.n_contexts + w * BLOCK_SIZE
        hits += int(bayes.predict(ds.features[i], list(range(s, s + BLOCK_SIZE)))
                    == ds.labels[i])
    assert abs(hits / len(ds.labels) - 1.0 / N_LABELS) < 0.05


def test_myopic_never_repeats_a_group(ds, bayes):
    got, preds = bayes.run_myopic(ds.features[0], 6)
    assert len(got) == len(set(got)) == 6
    assert len(preds) == 6


def test_free_groups_do_not_consume_budget(ds, bayes):
    """free_groups is the bracketing arm; it must grant information for free."""
    got, _ = bayes.run_myopic(ds.features[0], 5, free_groups=(0,))
    assert len(got) == 5 and 0 not in got


def test_forced_first_is_honoured_then_myopic_resumes(ds, bayes):
    got, _ = bayes.run_myopic(ds.features[0], 4, forced_first=0)
    assert got[0] == 0 and 0 not in got[1:]


def test_exact_scorer_is_deterministic(ds, bayes):
    """No RNG anywhere in the default path -- repeated runs must be identical.

    This is what makes the three Bayes arms matched by construction rather than
    by common-random-number bookkeeping.
    """
    runs = [bayes.run_myopic(ds.features[3], 6) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_myopic_declines_the_context_from_the_empty_state(bayes):
    """I(y; c) = 0, so at step 1 the context has exactly zero myopic value.

    This is the one place the original 'myopic never buys the context'
    intuition holds. It fails from step 2 onward -- see the correction in
    cube_nm_repro's docstring.
    """
    assert bayes.myopic_step_exact(np.zeros(bayes.H),
                                   list(range(bayes.n_groups))) != 0


def test_myopic_wants_the_context_once_a_block_feature_is_seen(ds, bayes):
    """The refutation of the original claim, pinned as a test.

    After observing block features, knowing the context is what says whether
    those values are signal or noise, so I(y; c | x) > 0 and the myopic policy
    starts asking for it.
    """
    wants = 0
    for i in range(25):
        got, _ = bayes.run_myopic(ds.features[i], 4)
        wants += int(0 in got)
    assert wants > 12, f"context bought in only {wants}/25 rows"


def test_sampled_scorer_still_reachable_for_audit(ds, bayes):
    """The MC path is retained deliberately: it documents why exact was needed."""
    got, _ = bayes.run_myopic(ds.features[0], 4, np.random.default_rng(0), n_mc=16)
    assert len(got) == 4
