"""Invariants for the gated task family.

The family's whole purpose is that deliberation is sometimes harmful, and that
configurations are indistinguishable until observed. Both properties are easy to
destroy with an innocent-looking parameter edit, so they are pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from governor.envs.gated_family import (
    CODE_BITS,
    N_LABELS,
    GateConfig,
    GatedBayes,
    GatedTask,
)


@pytest.fixture(scope="module")
def helpful():
    t = GatedTask(cfg=GateConfig(sigma_other=1.5), n_samples=400, seed=3)
    return t, GatedBayes(t)


@pytest.fixture(scope="module")
def useless():
    t = GatedTask(cfg=GateConfig(sigma_other=0.1), n_samples=400, seed=3)
    return t, GatedBayes(t)


def test_configs_are_superficially_identical(helpful, useless):
    """The anti-lookup property: nothing observable-without-acquiring differs."""
    (a, ba), (b, bb) = helpful, useless
    assert a.features.shape == b.features.shape
    assert a.cfg.n_groups == b.cfg.n_groups
    assert ba.H == bb.H
    assert np.array_equal(ba.cost, bb.cost)


def test_primary_block_noise_is_config_invariant(helpful, useless):
    """sigma_sig must not move with sigma_other, or the comparison confounds."""
    (_, ba), (_, bb) = helpful, useless
    assert ba.cfg.sigma_sig == bb.cfg.sigma_sig


def test_exactly_three_code_columns_per_block(helpful):
    t, b = helpful
    for ctx in (0, 2):
        for y in (0, 5, 7):
            h = ctx * N_LABELS + y
            base = b.K + ctx * b.M
            tight = [c for c in range(base, base + b.M)
                     if b.SD[h, c] == t.cfg.sigma_sig]
            assert sorted(tight) == sorted(base + (y + j) % b.M
                                           for j in range(CODE_BITS))


def test_posterior_normalised_and_uniform_when_empty(helpful):
    _, b = helpful
    py = b.label_posterior(np.zeros(b.H))
    assert np.allclose(py, 1.0 / N_LABELS)


def test_gate_is_informative_about_context_only(helpful):
    """I(y; gate) = 0 from the empty state -- the non-myopic trap survives."""
    _, b = helpful
    g = b.gains(np.zeros(b.H), list(range(b.n_groups)))
    assert g[0] < 1e-9, f"gate gain from empty state should be ~0, got {g[0]}"
    assert max(v for k, v in g.items() if k != 0) > 1e-6


def test_myopic_respects_cost_and_budget(helpful):
    t, b = GatedTask(cfg=GateConfig(gate_cost=3.0), n_samples=50, seed=1), None
    b = GatedBayes(t)
    got, _, spent = b.run(t.features[0], 4.0)
    assert spent <= 4.0 + 1e-9
    assert len(got) == len(set(got))


def test_expensive_gate_is_declined_at_tight_budget():
    t = GatedTask(cfg=GateConfig(sigma_other=1.5, gate_cost=3.0),
                  n_samples=30, seed=1)
    b = GatedBayes(t)
    got, _, _ = b.run(t.features[0], 3.0)
    assert 0 not in got or len(got) == 1


def test_run_is_deterministic(helpful):
    t, b = helpful
    assert b.run(t.features[5], 5.0) == b.run(t.features[5], 5.0)


def test_forced_first_charges_the_gate_cost():
    t = GatedTask(cfg=GateConfig(gate_cost=2.0), n_samples=30, seed=1)
    b = GatedBayes(t)
    got, _, spent = b.run(t.features[0], 6.0, forced_first=0)
    assert got[0] == 0 and spent <= 6.0 + 1e-9


def test_useless_config_makes_every_block_informative(useless):
    """If the gate is worthless it must be because the blocks are equal."""
    t, b = useless
    hits = 0
    for i in range(120):
        wrong = (int(t.context[i]) + 1) % t.cfg.n_contexts
        s = b.K + wrong * b.M
        hits += int(b.predict(t.features[i], list(range(s, s + b.M)))
                    == t.labels[i])
    assert hits / 120 > 0.5, "non-primary block should still identify the label"
