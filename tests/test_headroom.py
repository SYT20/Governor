"""The headroom law. Closed form, so it is testable against simulation exactly.

This module exists because two families were designed by intuition and rejected
after their ceilings were measured. A law that predicts the ceiling turns a
day of API quota into a microsecond.
"""
from __future__ import annotations

import numpy as np
import pytest

from governor.phase4.headroom import (
    best_k, ceiling_approx, ceiling_asymptote, ceiling_binary, ceiling_general,
    design, n_for_target, realisation_ratio,
)


def test_closed_form_matches_simulation():
    """If these disagree, the algebra is wrong and every design number is too."""
    for n, k, p in ((4, 2, 0.475), (6, 3, 0.5), (8, 4, 0.5), (10, 3, 0.3),
                    (12, 6, 0.6), (5, 1, 0.2)):
        pool = np.array([1.0] * int(p * 10000) + [0.0] * (10000 - int(p * 10000)))
        sim = ceiling_general(pool, n, k, trials=60000, seed=1)
        assert ceiling_binary(n, k, p) == pytest.approx(sim, abs=0.004), (n, k, p)


def test_ceiling_is_zero_when_allocation_cannot_matter():
    for n, p in ((6, 0.5), (10, 0.3)):
        assert ceiling_binary(n, 0, p) == 0.0        # nothing to allocate
        assert ceiling_binary(n, n, p) == 0.0        # everything gets upgraded
    for n, k in ((6, 3), (8, 4)):
        assert ceiling_binary(n, k, 0.0) == pytest.approx(0.0)   # nothing helps
        assert ceiling_binary(n, k, 1.0) == pytest.approx(0.0)   # all helps


def test_ceiling_peaks_at_p_one_half():
    peak = max(((p, best_k(8, p)[1]) for p in np.linspace(0.05, 0.95, 19)),
               key=lambda t: t[1])
    assert abs(peak[0] - 0.5) < 0.06, peak


def test_ceiling_grows_with_episode_length_toward_the_asymptote():
    """The lever an '8-12 items' rule was reaching for, made quantitative."""
    vals = [best_k(n, 0.5)[1] for n in (4, 6, 8, 12, 20, 50)]
    assert vals == sorted(vals), vals
    assert vals[0] < 0.16 and vals[-1] > 0.22
    assert all(v < ceiling_asymptote(0.5) for v in vals)


def test_asymptotic_approximation_is_accurate():
    for n in (8, 12, 20, 50, 200):
        assert ceiling_approx(n, 0.5) == pytest.approx(best_k(n, 0.5)[1], abs=0.004)


def test_short_episodes_are_fragile_to_operating_point_drift():
    """Why Phase 4R failed held-out: k drifted 1.50 -> 1.25 and the ideal
    ceiling halved. At n=12 the same drift costs 5%."""
    def loss(n):
        k, c = best_k(n, 0.5)
        return 1.0 - ceiling_binary(n, k - 1, 0.5) / c
    assert loss(4) > 0.25
    assert loss(12) < 0.08
    assert loss(4) > loss(8) > loss(12)


def test_n_for_target_is_monotone_and_respects_the_asymptote():
    assert n_for_target(0.5, 0.30) is None, "0.30 exceeds p(1-p)=0.25"
    a, b = n_for_target(0.5, 0.16), n_for_target(0.5, 0.20)
    assert a is not None and b is not None and b > a


def test_realisation_ratio_reproduces_the_two_recorded_failures():
    """Phase 4 destroyed 60% of its headroom; the Phase 4R redesign kept 91%."""
    p4 = realisation_ratio(0.0455, ceiling_binary(4, 2, 0.557))
    p4r = realisation_ratio(0.1389, ceiling_binary(6, 2, 0.438))
    assert p4 < 0.45, p4
    assert p4r > 0.85, p4r


def test_design_reports_the_finite_episode_loss():
    d = design(4, 0.5)
    assert d.k_upgrades == 2 and d.finite_episode_loss > 0.35
    assert design(24, 0.5).finite_episode_loss < 0.20


def test_general_ceiling_handles_continuous_gains():
    """The second task family has partial credit, so gains are not binary."""
    rng = np.random.default_rng(0)
    g = np.clip(rng.beta(2, 2, 4000), 0, 1)
    c = ceiling_general(g, 8, 4, trials=20000)
    assert 0.0 < c < 0.25
    assert ceiling_general(np.full(4000, 0.5), 8, 4) == pytest.approx(0.0, abs=1e-3)
