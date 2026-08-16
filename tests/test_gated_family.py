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
    REGIME_GRID,
    GateConfig,
    GatedTask,
    ObservableBayes,
    OracleBayes,
)


@pytest.fixture(scope="module")
def helpful():
    t = GatedTask(cfg=GateConfig(sigma_other=1.5), n_samples=400, seed=3)
    return t, OracleBayes(t)


@pytest.fixture(scope="module")
def useless():
    t = GatedTask(cfg=GateConfig(sigma_other=0.1), n_samples=400, seed=3)
    return t, OracleBayes(t)


# -- the observability contract ------------------------------------------------
# These are the tests that were missing. The first version of this module
# asserted in prose that the configuration was hidden from the policy while the
# scorer read cfg.sigma_other to build its likelihood. Prose does not execute.


def test_observable_scorer_cannot_tell_regimes_apart_before_observing():
    """THE regression test. From the empty state nothing has been seen, so two
    configurations differing only in sigma_other MUST produce identical gains.

    The oracle scorer fails this by design (0.5537 vs 0.2939), which is what
    exposed the leak.
    """
    g = []
    for so in (0.10, 1.50):
        t = GatedTask(cfg=GateConfig(sigma_other=so), n_samples=5, seed=1)
        b = ObservableBayes(t)
        g.append(b.gains(np.zeros(b.H), list(range(b.n_groups))))
    for k in g[0]:
        assert abs(g[0][k] - g[1][k]) < 1e-9, f"group {k} leaks the regime"


def test_oracle_scorer_does_see_the_regime():
    """Pins the distinction rather than leaving it implicit -- if this ever
    starts passing as 'identical', the oracle has silently become observable."""
    g = []
    for so in (0.10, 1.50):
        t = GatedTask(cfg=GateConfig(sigma_other=so), n_samples=5, seed=1)
        b = OracleBayes(t)
        g.append(b.gains(np.zeros(b.H), list(range(b.n_groups))))
    assert abs(g[0][5] - g[1][5]) > 0.05


def test_observable_posterior_starts_at_the_prior_and_moves():
    t = GatedTask(cfg=GateConfig(sigma_other=1.5), n_samples=60, seed=2)
    b = ObservableBayes(t)
    assert np.allclose(b.regime_posterior(np.zeros(b.H)), 1.0 / len(REGIME_GRID))
    # after observing a whole wrong block, the regime belief must have updated
    wrong = (int(t.context[0]) + 1) % t.cfg.n_contexts
    s = b.K + wrong * b.M
    post = b.regime_posterior(b.loglik_cols(t.features[0], list(range(s, s + b.M))))
    assert abs(post.max() - 1.0 / len(REGIME_GRID)) > 0.05


def test_observable_regime_belief_concentrates_on_the_truth():
    """Averaged over instances, seeing enough evidence should favour the truth."""
    for so, want in ((0.10, 0), (1.50, len(REGIME_GRID) - 1)):
        t = GatedTask(cfg=GateConfig(sigma_other=so), n_samples=40, seed=5)
        b = ObservableBayes(t)
        acc = np.zeros(len(REGIME_GRID))
        for i in range(40):
            acc += b.regime_posterior(
                b.loglik_cols(t.features[i], list(range(b.nf))))
        assert int(np.argmax(acc)) == want, f"sigma_other={so} -> {acc / 40}"


def test_observable_hypothesis_space_is_the_product():
    t = GatedTask(cfg=GateConfig(), n_samples=5, seed=1)
    b = ObservableBayes(t)
    assert b.H == len(REGIME_GRID) * t.cfg.n_contexts * N_LABELS
    assert not b.knows_regime
    assert OracleBayes(t).knows_regime


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
    t = GatedTask(cfg=GateConfig(gate_cost=3.0), n_samples=50, seed=1)
    b = OracleBayes(t)
    got, _, spent = b.run(t.features[0], 4.0)
    assert spent <= 4.0 + 1e-9
    assert len(got) == len(set(got))


def test_expensive_gate_is_declined_at_tight_budget():
    t = GatedTask(cfg=GateConfig(sigma_other=1.5, gate_cost=3.0),
                  n_samples=30, seed=1)
    b = OracleBayes(t)
    got, _, _ = b.run(t.features[0], 3.0)
    assert 0 not in got or len(got) == 1


def test_run_is_deterministic(helpful):
    t, b = helpful
    assert b.run(t.features[5], 5.0) == b.run(t.features[5], 5.0)


def test_forced_first_charges_the_gate_cost():
    t = GatedTask(cfg=GateConfig(gate_cost=2.0), n_samples=30, seed=1)
    b = OracleBayes(t)
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


def test_regime_posterior_starts_at_the_preregistered_prior():
    """The prior is part of the decision problem, so it is pinned, not implied.

    A prior quietly inherited from the frequency of regimes in whatever grid an
    experiment sweeps would be privileged information in a Bayesian costume.
    """
    from governor.envs.gated_family import REGIME_PRIOR
    for so in REGIME_GRID:
        t = GatedTask(cfg=GateConfig(sigma_other=so), n_samples=4, seed=1)
        b = ObservableBayes(t)
        assert np.allclose(b.regime_posterior(b.prior_logL()), REGIME_PRIOR)


def test_non_uniform_prior_is_honoured_and_validated():
    t = GatedTask(cfg=GateConfig(), n_samples=4, seed=1)
    from governor.envs.gated_family import GatedBayes
    p = (0.5, 0.2, 0.1, 0.1, 0.1)
    b = GatedBayes(t, regimes=REGIME_GRID, prior=p)
    assert np.allclose(b.regime_posterior(b.prior_logL()), p)
    with pytest.raises(ValueError):
        GatedBayes(t, regimes=REGIME_GRID, prior=(0.5, 0.5))


# -- Environment 5 mode invariants ---------------------------------------------

def test_env5_mode_cost_ordering_and_coverage():
    """0 = C(M0) < C(M1) < C(M2), sharing a primitive. The construction gate."""
    from governor.envs.probe_family import (ProbeTask, ObservableProbeBayes,
                                            make_config)
    from governor.envs.env5_modes import InstrumentedBayes, ModeRunner
    t = ProbeTask(cfg=make_config(0.60, 1.0, 0.05), n_samples=6, seed=1)
    ib = InstrumentedBayes(ObservableProbeBayes(t))
    logL, avail, x = ib.prior_logL(), list(range(ib.n_groups)), t.features[0]
    for g in (7, 19):
        avail.remove(g)
        logL = logL + ib.b.loglik_cols(x, ib.group_cols[g])
    c = {}
    for m in ("M0", "M1", "M2"):
        ib.reset_counters()
        r = ModeRunner(ib=ib, b_tool=5.0, b_compute=1e9)
        _, _, d = r.invoke(m, logL, avail, x, "candidate_evals")
        c[m] = d.as_dict()
    shared = [k for k in c["M1"] if c["M1"][k] > 0 and c["M2"][k] > 0]
    assert shared, "M1 and M2 share no primitive -- no budget can make them compete"
    for k in shared:
        assert c["M0"][k] < c["M1"][k] < c["M2"][k]


def test_env5_m0_is_free():
    """M0 is a reflex. C(M0)=0 must be true, not asserted."""
    from governor.envs.probe_family import (ProbeTask, ObservableProbeBayes,
                                            make_config)
    from governor.envs.env5_modes import InstrumentedBayes, ModeRunner
    t = ProbeTask(cfg=make_config(0.35, 1.0, 0.05), n_samples=4, seed=2)
    ib = InstrumentedBayes(ObservableProbeBayes(t))
    ib.reset_counters()
    r = ModeRunner(ib=ib, b_tool=5.0, b_compute=1e9)
    _, a, d = r.invoke("M0", ib.prior_logL(), list(range(ib.n_groups)),
                       t.features[0], "candidate_evals")
    assert a is not None
    assert all(v == 0 for v in d.as_dict().values())


def test_env5_deliberation_acquires_nothing():
    """PREREGISTERED invariant: delta tool_calls == 0 across M1 and M2."""
    from governor.envs.probe_family import (ProbeTask, ObservableProbeBayes,
                                            make_config)
    from governor.envs.env5_modes import InstrumentedBayes, ModeRunner
    t = ProbeTask(cfg=make_config(1.50, 1.0, 0.05), n_samples=4, seed=3)
    ib = InstrumentedBayes(ObservableProbeBayes(t))
    for m in ("M1", "M2"):
        r = ModeRunner(ib=ib, b_tool=5.0, b_compute=1e9)
        before = r.tool_spent
        r.invoke(m, ib.prior_logL(), list(range(ib.n_groups)),
                 t.features[0], "candidate_evals")
        assert r.tool_spent == before


def test_env5_m1_output_is_constant_at_the_prior():
    """At t=0 nothing is observed, so M1 MUST return the same assessment.

    The first version of the variation test below queried M1 from the prior
    state and failed -- correctly. Every instance presents the identical
    posterior before any acquisition, exactly as measured in the gated family
    (state-feature variance 3.01e-27). Pinning it here so the distinction
    between "M1 is uninformative" and "t=0 has no state" cannot be confused
    again.
    """
    from governor.envs.probe_family import (ProbeTask, ObservableProbeBayes,
                                            make_config)
    from governor.envs.env5_modes import InstrumentedBayes, ModeRunner
    t = ProbeTask(cfg=make_config(0.60, 1.0, 0.05), n_samples=6, seed=4)
    ib = InstrumentedBayes(ObservableProbeBayes(t))
    seen = set()
    for i in range(6):
        r = ModeRunner(ib=ib, b_tool=6.0, b_compute=1e9)
        r.invoke("M1", ib.prior_logL(), list(range(ib.n_groups)),
                 t.features[i], "candidate_evals")
        seen.add(round(r.trace[-1]["assessment"], 6))
    assert len(seen) == 1, f"prior state should be identical across rows: {seen}"


def test_env5_m1_output_varies_once_evidence_exists():
    """M1 must be informative: constant output means it can be deleted.

    Evaluated after acquisitions, which is where the M0/M2 choice is actually
    made -- the t=0 decision has no state to condition on and is excluded by
    construction.
    """
    from governor.envs.probe_family import (ProbeTask, ObservableProbeBayes,
                                            make_config)
    from governor.envs.env5_modes import InstrumentedBayes, ModeRunner
    seen = set()
    for so in (0.10, 0.60, 1.50):
        t = ProbeTask(cfg=make_config(so, 1.0, 0.05), n_samples=8, seed=4)
        ib = InstrumentedBayes(ObservableProbeBayes(t))
        for i in range(8):
            x = t.features[i]
            logL, avail = ib.prior_logL(), list(range(ib.n_groups))
            for _ in range(2):                      # advance into real states
                g = ib.b.myopic_step(logL, avail, 6.0)
                avail.remove(g)
                logL = logL + ib.b.loglik_cols(x, ib.group_cols[g])
            r = ModeRunner(ib=ib, b_tool=6.0, b_compute=1e9)
            r.invoke("M1", logL, avail, x, "candidate_evals")
            v = r.trace[-1]["assessment"]
            if v is not None:
                seen.add(round(v, 3))
    assert len(seen) > 3, f"M1 assessment nearly constant: {seen}"


# -- sequential execution: the regression that would have caught the run-I bug --

def test_h_and_m2_produce_DIFFERENT_next_states():
    """The invalid binding-budget run advanced every trajectory under H, then
    scored all policies by summing local dstar from that one trajectory. It
    therefore assumed s_2^M2 == s_2^H.

    This test falsifies that assumption directly. If choosing M2 at t=1 does not
    change the state reaching t=2, the sequential experiment is vacuous and any
    'allocation' result is meaningless.
    """
    import numpy as np
    from governor.envs.probe_family import (ProbeTask, ObservableProbeBayes,
                                            make_config)
    from governor.envs.env5_modes import (InstrumentedBayes, h_gate_first,
                                          m2_plan)
    t = ProbeTask(cfg=make_config(0.60, 1.0, 0.05), n_samples=12, seed=7)
    ib = InstrumentedBayes(ObservableProbeBayes(t))
    differed = 0
    for i in range(12):
        x = t.features[i]
        logL, av = ib.prior_logL(), list(range(ib.n_groups))
        # reach a non-trivial first decision point
        g = ib.b.myopic_step(logL, av, 6.0)
        av.remove(g); logL = logL + ib.b.loglik_cols(x, ib.group_cols[g])

        ah = h_gate_first(ib, logL, av, 4.0, acquired=True)
        am = m2_plan(ib, logL, av, 4.0)
        if ah is None or am is None or ah == am:
            continue
        # EXECUTE each choice and compare the resulting belief states
        lh = logL + ib.b.loglik_cols(x, ib.group_cols[ah])
        lm = logL + ib.b.loglik_cols(x, ib.group_cols[am])
        differed += int(not np.allclose(lh, lm))
    # MEASURED, not guessed: H and M2 select the SAME action in 10 of 12 states
    # at this decision point, so only a minority of states can have non-zero
    # Delta* at all. The first version of this test asserted differed >= 5 --
    # a threshold I invented -- and failed at 2. Lowering it to pass would be
    # gate-shopping; the honest assertion is the structural one, that execution
    # branches at all, with the RATE reported as a diagnostic.
    assert differed >= 1, (
        "executing H vs M2 never changed the next state; a sequential "
        "allocation experiment would be vacuous")


def test_executing_m2_changes_what_is_affordable_later():
    """Second half of the same guard: the compute budget must actually bind.

    If invoking M2 does not reduce what remains, there is no opportunity cost
    and the 'Governor' is a selector, which is what the single-decision test
    turned out to be measuring.
    """
    from governor.envs.probe_family import (ProbeTask, ObservableProbeBayes,
                                            make_config)
    from governor.envs.env5_modes import InstrumentedBayes, ModeRunner
    t = ProbeTask(cfg=make_config(0.60, 1.0, 0.05), n_samples=4, seed=7)
    ib = InstrumentedBayes(ObservableProbeBayes(t))
    r = ModeRunner(ib=ib, b_tool=6.0, b_compute=5000.0)
    before = r.compute_spent
    r.invoke("M2", ib.prior_logL(), list(range(ib.n_groups)), t.features[0],
             "candidate_evals")
    after_m2 = r.compute_spent
    r.invoke("M0", ib.prior_logL(), list(range(ib.n_groups)), t.features[0],
             "candidate_evals")
    after_m0 = r.compute_spent
    assert after_m2 > before, "M2 consumed no compute; the budget cannot bind"
    assert after_m0 == after_m2, "M0 consumed compute; it is not a free reflex"
