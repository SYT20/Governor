"""Stage 0/1 invariants. Pure stdlib: `python3 -m unittest discover -s tests`.

These are not smoke tests. Each one guards a specific claim the Decision Record
makes, and several are the CI drift checks named in section 3.
"""

from __future__ import annotations

import math
import unittest

from governor.accounting.meter import (
    Accountant,
    BudgetExhausted,
    BudgetViolation,
    Envelope,
)
from governor.cognitive.belief import (
    Channel,
    ChannelBook,
    ChannelEstimator,
    conservative_channel,
    entropy,
    uniform,
    update_belief,
)
from governor.core.estimate import Estimate, ProvenanceError, require_provenance
from governor.envs.synthbug import (
    Action,
    Mode,
    PatchState,
    SynthBug,
    SynthConfig,
    Tier,
    all_actions,
    make_task,
)


class TestProvenance(unittest.TestCase):
    """Section D: no number reaches the policy without declared provenance."""

    def test_bare_float_rejected(self):
        with self.assertRaises(ProvenanceError):
            require_provenance(0.42, "p_success")

    def test_measured_needs_no_interval(self):
        e = Estimate.measured(1234.0, unit="tokens")
        self.assertEqual(e.source, "measured")
        self.assertEqual(e.ucb, 1234.0)

    def test_fitted_requires_ci_model_and_data_version(self):
        with self.assertRaises(ProvenanceError):
            Estimate(value=0.5, source="fitted")
        with self.assertRaises(ProvenanceError):
            Estimate(value=0.5, source="fitted", ci=(0.4, 0.6))
        with self.assertRaises(ProvenanceError):
            Estimate(value=0.5, source="fitted", ci=(0.4, 0.6), model_id="v1")
        ok = Estimate.fitted(0.5, ci=(0.4, 0.6), model_id="v1", data_version="c0ffee")
        self.assertAlmostEqual(ok.ci_width, 0.2, places=9)

    def test_value_must_lie_inside_its_own_interval(self):
        with self.assertRaises(ProvenanceError):
            Estimate.fitted(0.9, ci=(0.1, 0.5), model_id="v1", data_version="d")

    def test_ucb_drives_pessimistic_admissibility(self):
        e = Estimate.fitted(10.0, ci=(6.0, 22.0), model_id="cost", data_version="d")
        self.assertEqual(e.ucb, 22.0)  # p90-style upper bound, not the mean
        self.assertFalse(e.is_confident(max_ci_width=5.0))
        self.assertTrue(e.is_confident(max_ci_width=20.0))

    def test_unknown_source_rejected(self):
        with self.assertRaises(ProvenanceError):
            Estimate(value=1.0, source="vibes")  # type: ignore[arg-type]


class TestAccountant(unittest.TestCase):
    """Section H.1: enforcement is structural, not statistical."""

    def setUp(self):
        self.env = Envelope(tokens=10_000, cost=1.0, wall_s=300, tool_calls=50)
        self.acc = Accountant(envelope=self.env)

    def test_charges_accumulate_and_reconcile(self):
        self.acc.charge("explore", tokens=1200, cost=0.02, wall_s=4, tool_calls=1)
        self.acc.charge("verify", tokens=800, cost=0.01, wall_s=9, tool_calls=1)
        self.assertAlmostEqual(self.acc.consumed()["tokens"], 2000)
        self.acc.reconcile()  # drift check #2
        self.assertFalse(self.acc.violated())

    def test_overrun_raises_rather_than_logging(self):
        with self.assertRaises(BudgetViolation):
            self.acc.charge("runaway", tokens=10_001)
        # The failed charge must not have been partially applied.
        self.assertEqual(self.acc.consumed()["tokens"], 0.0)
        self.acc.reconcile()

    def test_bvr_is_zero_under_adversarial_charging(self):
        """Drift check: BVR = 0 must hold no matter what the executor attempts."""
        rejected = 0
        for i in range(500):
            try:
                self.acc.charge(f"a{i}", tokens=97, cost=0.011, tool_calls=1)
            except BudgetViolation:
                rejected += 1
        self.assertGreater(rejected, 0, "test did not actually stress the envelope")
        self.assertFalse(self.acc.violated())
        self.acc.reconcile()

    def test_admissible_uses_upper_bound_not_mean(self):
        self.acc.charge("spend", tokens=9_500)
        cheap_mean_wide_tail = Estimate.fitted(
            300.0, ci=(100.0, 900.0), model_id="cost", data_version="d"
        )
        # Mean (300) fits in the 500 remaining; the p90 tail (900) does not.
        self.assertFalse(self.acc.admissible({"tokens": cheap_mean_wide_tail}))
        self.assertTrue(
            self.acc.admissible({"tokens": Estimate.fitted(
                300.0, ci=(200.0, 400.0), model_id="cost", data_version="d")})
        )

    def test_reserve_blocks_actions_that_would_strand_the_episode(self):
        self.acc.charge("spend", tokens=9_000)
        est = Estimate.fitted(600.0, ci=(500.0, 700.0), model_id="c", data_version="d")
        self.assertTrue(self.acc.admissible({"tokens": est}))
        # With 400 tokens reserved to verify and report, it no longer fits.
        self.assertFalse(self.acc.admissible({"tokens": est}, reserve={"tokens": 400.0}))

    def test_dispatch_refused_when_floor_does_not_fit(self):
        self.acc.charge("spend", tokens=9_990)
        with self.assertRaises(BudgetExhausted):
            self.acc.dispatch_or_refuse({"tokens": 50.0})

    def test_fraction_remaining_is_the_tightest_dimension(self):
        self.acc.charge("x", tokens=5_000, cost=0.9)  # 50% tokens, 10% cost left
        self.assertAlmostEqual(self.acc.fraction_remaining(), 0.10, places=6)

    def test_envelope_scaling_for_degradation_curve(self):
        half = self.env.scaled(0.5)
        self.assertEqual(half.tokens, 5_000)
        self.assertEqual(half.cost, 0.5)

    def test_negative_and_unknown_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            self.acc.charge("bad", tokens=-1)
        with self.assertRaises(KeyError):
            self.acc.charge("bad", gpu_hours=1)


class TestBeliefUpdate(unittest.TestCase):
    """Section G.3: the corrected two-parameter likelihood."""

    def test_hand_worked_example_to_six_decimals(self):
        """Three hypotheses, uniform prior, positive evidence for h0.

        alpha=0.8, beta=0.2. Unnormalised posterior is
            h0: 1/3 * 0.8 = 0.266667
            h1: 1/3 * 0.2 = 0.066667
            h2: 1/3 * 0.2 = 0.066667
        Total 0.4 -> normalised (2/3, 1/6, 1/6).
        """
        ch = Channel("t", alpha=0.8, beta=0.2)
        post = update_belief(uniform(3), ch, target=0, observation=1)
        self.assertAlmostEqual(post[0], 2 / 3, places=6)
        self.assertAlmostEqual(post[1], 1 / 6, places=6)
        self.assertAlmostEqual(post[2], 1 / 6, places=6)
        self.assertAlmostEqual(sum(post), 1.0, places=12)

    def test_asymmetric_channel_differs_from_the_old_symmetric_formula(self):
        """The bug rev 1 contained, pinned as a regression test.

        Rev 1 assumed beta = 1 - alpha. With alpha=0.9, beta=0.3 the two disagree,
        and the corrected version is the one that matches P(o|H).
        """
        n = 3
        alpha, beta = 0.9, 0.3
        correct = Channel("c", alpha=alpha, beta=beta)
        corrected_post = update_belief(uniform(n), correct, target=0, observation=1)

        # What rev 1 would have computed: 1 - alpha for the non-target hypotheses.
        legacy_lik = [alpha if i == 0 else (1 - alpha) for i in range(n)]
        tot = sum(p * l for p, l in zip(uniform(n), legacy_lik))
        legacy_post = [p * l / tot for p, l in zip(uniform(n), legacy_lik)]

        self.assertNotAlmostEqual(corrected_post[0], legacy_post[0], places=3)
        # Corrected: 0.3/(0.9+0.3+0.3)... check against explicit arithmetic.
        expected0 = alpha / (alpha + beta + beta)
        self.assertAlmostEqual(corrected_post[0], expected0, places=9)

    def test_negative_observation_moves_belief_away(self):
        ch = Channel("t", alpha=0.85, beta=0.15)
        post = update_belief(uniform(4), ch, target=2, observation=0)
        self.assertLess(post[2], 0.25)
        self.assertGreater(post[0], 0.25)

    def test_likelihood_ratios_match_diagnostic_form(self):
        ch = Channel("t", alpha=0.9, beta=0.2)
        self.assertAlmostEqual(ch.lr_positive, 0.9 / 0.2, places=9)
        self.assertAlmostEqual(ch.lr_negative, 0.1 / 0.8, places=9)

    def test_uninformative_channel_rejected(self):
        with self.assertRaises(ValueError):
            Channel("bad", alpha=0.3, beta=0.5)

    def test_conservative_fallback_caps_the_likelihood_ratio(self):
        ch = conservative_channel(cap=3.0)
        self.assertAlmostEqual(ch.lr_positive, 3.0, places=9)
        b = uniform(3)
        for _ in range(12):  # repeated confirming evidence
            b = update_belief(b, ch, target=0, observation=1)
        self.assertLess(b[0], 1.0)  # never reaches certainty on one channel

    def test_entropy_is_normalised_and_monotone(self):
        self.assertAlmostEqual(entropy(uniform(4)), 1.0, places=9)
        peaked = [0.97, 0.01, 0.01, 0.01]
        self.assertLess(entropy(peaked), 0.2)

    def test_belief_survives_total_refutation(self):
        ch = Channel("t", alpha=0.999, beta=0.001)
        b = [1.0, 0.0, 0.0]
        out = update_belief(b, ch, target=1, observation=1)
        self.assertAlmostEqual(sum(out), 1.0, places=9)


class TestChannelEstimation(unittest.TestCase):
    """The fitted channel must converge to the parameters that generated the data."""

    def test_recovers_true_alpha_and_beta_from_synthbug(self):
        cfg = SynthConfig(n_hypotheses=4)
        tier = Tier.T1
        book = ChannelBook()
        name = f"explore@{tier}"

        for seed in range(400):
            task = make_task(seed, cfg)
            for h in range(cfg.n_hypotheses):
                obs = task.step(Action(Mode.EXPLORE, tier, h))
                book.observe(
                    name,
                    target_was_true=(h == task.true_cause),
                    observation=obs.value,
                )

        est = book.estimator(name)
        self.assertAlmostEqual(est.mean_alpha(), cfg.alpha[tier], delta=0.05)
        self.assertAlmostEqual(est.mean_beta(), cfg.beta[tier], delta=0.05)
        self.assertTrue(est.is_usable())

    def test_thin_data_falls_back_instead_of_pretending(self):
        book = ChannelBook()
        book.observe("explore@T0", target_was_true=True, observation=1)
        ch = book.channel("explore@T0")
        self.assertIn("fallback", ch.name)


class TestSynthBug(unittest.TestCase):
    """The environment must be deterministic and honest about ground truth."""

    def test_same_seed_gives_identical_episode(self):
        script = [
            Action(Mode.EXPLORE, Tier.T1, 0),
            Action(Mode.EXPLORE, Tier.T1, 1),
            Action(Mode.EXPLOIT, Tier.T1, 0),
            Action(Mode.VERIFY, Tier.T1),
        ]
        runs = []
        for _ in range(2):
            t = make_task(7)
            runs.append([(t.step(a).value, tuple(sorted(t.cost_of(a).items()))) for a in script])
        self.assertEqual(runs[0], runs[1])  # drift check #3: determinism

    def test_different_seeds_diverge(self):
        a = [make_task(s).true_cause for s in range(50)]
        self.assertGreater(len(set(a)), 1)

    def test_correct_repair_can_succeed_wrong_repair_cannot(self):
        cfg = SynthConfig(p_fix={Tier.T0: 1.0, Tier.T1: 1.0, Tier.T2: 1.0})
        t = SynthBug(config=cfg, seed=3)
        t.step(Action(Mode.EXPLOIT, Tier.T1, t.true_cause))
        self.assertIs(t.patch, PatchState.CORRECT)
        t.step(Action(Mode.STOP_UNVERIFIED))
        self.assertTrue(t.succeeded())

        wrong = (t.true_cause + 1) % cfg.n_hypotheses
        t2 = SynthBug(config=cfg, seed=3)
        t2.step(Action(Mode.EXPLOIT, Tier.T1, wrong))
        t2.step(Action(Mode.STOP_UNVERIFIED))
        self.assertFalse(t2.succeeded())

    def test_giving_up_never_counts_as_success(self):
        cfg = SynthConfig(p_fix={Tier.T0: 1.0, Tier.T1: 1.0, Tier.T2: 1.0})
        t = SynthBug(config=cfg, seed=11)
        t.step(Action(Mode.EXPLOIT, Tier.T1, t.true_cause))
        t.step(Action(Mode.STOP_FAILURE))
        self.assertIs(t.patch, PatchState.CORRECT)
        self.assertFalse(t.succeeded())

    def test_higher_tier_evidence_is_more_discriminative(self):
        cfg = SynthConfig()
        hits = {}
        for tier in (Tier.T0, Tier.T2):
            correct = 0
            for seed in range(600):
                t = make_task(seed, cfg)
                o = t.step(Action(Mode.EXPLORE, tier, t.true_cause))
                correct += o.value
            hits[tier] = correct / 600
        self.assertGreater(hits[Tier.T2], hits[Tier.T0] + 0.1)

    def test_terminal_actions_end_the_episode(self):
        t = make_task(1)
        t.step(Action(Mode.STOP_FAILURE))
        with self.assertRaises(RuntimeError):
            t.step(Action(Mode.VERIFY, Tier.T0))

    def test_action_space_is_complete(self):
        acts = all_actions(4)
        modes = {a.mode for a in acts}
        self.assertEqual(modes, set(Mode))
        self.assertEqual(len([a for a in acts if a.mode is Mode.EXPLORE]), 4 * 3)

    def test_cost_scales_with_tier(self):
        cfg = SynthConfig()
        means = {}
        for tier in (Tier.T0, Tier.T2):
            tot = 0.0
            for seed in range(300):
                tot += make_task(seed, cfg).cost_of(Action(Mode.EXPLOIT, tier, 0))["tokens"]
            means[tier] = tot / 300
        self.assertGreater(means[Tier.T2], means[Tier.T0] * 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
