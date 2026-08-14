"""Action-conditioned features, including the closed-form information gain.

Section 14 of the brief worried that "information gain" would become another
hallucinated number. These tests pin it to arithmetic: EIG is computed from the
belief vector and the *measured* channel parameters, and behaves the way the
definition requires.
"""

from __future__ import annotations

import unittest

from governor.cognitive.belief import Channel, uniform, update_belief
from governor.models.action_features import (
    ACTION_FEATURE_NAMES,
    action_features,
    expected_info_gain,
)


class TestExpectedInfoGain(unittest.TestCase):
    def setUp(self):
        self.ch = Channel("t", alpha=0.78, beta=0.20)

    def test_uncertain_belief_has_more_to_learn_than_confident_one(self):
        vague = expected_info_gain(uniform(4), 0, self.ch)
        sure = expected_info_gain([0.97, 0.01, 0.01, 0.01], 0, self.ch)
        self.assertGreater(vague, sure)

    def test_probing_a_likely_hypothesis_beats_probing_an_unlikely_one(self):
        b = [0.70, 0.10, 0.10, 0.10]
        self.assertGreater(
            expected_info_gain(b, 0, self.ch), expected_info_gain(b, 1, self.ch)
        )

    def test_gain_is_never_negative(self):
        """Expected posterior entropy cannot exceed prior entropy."""
        for b in (uniform(3), [0.5, 0.3, 0.2], [0.98, 0.01, 0.01], [0.4, 0.4, 0.2]):
            for t in range(len(b)):
                self.assertGreaterEqual(expected_info_gain(b, t, self.ch), -1e-9)

    def test_sharper_channel_yields_more_information(self):
        blunt = Channel("blunt", alpha=0.60, beta=0.40)
        sharp = Channel("sharp", alpha=0.95, beta=0.05)
        self.assertGreater(
            expected_info_gain(uniform(4), 0, sharp),
            expected_info_gain(uniform(4), 0, blunt),
        )

    def test_matches_explicit_posterior_computation(self):
        """Cross-check the closed form against literally doing the Bayes update."""
        from governor.cognitive.belief import entropy

        b, t = [0.5, 0.3, 0.2], 0
        p1 = b[t] * self.ch.alpha + (1 - b[t]) * self.ch.beta
        manual = entropy(b) - (
            p1 * entropy(update_belief(b, self.ch, t, 1))
            + (1 - p1) * entropy(update_belief(b, self.ch, t, 0))
        )
        self.assertAlmostEqual(expected_info_gain(b, t, self.ch), manual, places=10)

    def test_out_of_range_target_is_safe(self):
        self.assertEqual(expected_info_gain([0.5, 0.5], 7, self.ch), 0.0)
        self.assertEqual(expected_info_gain([], 0, self.ch), 0.0)


class TestActionFeatures(unittest.TestCase):
    def setUp(self):
        self.chs = {"T0": Channel("T0", alpha=0.62, beta=0.34),
                    "T1": Channel("T1", alpha=0.78, beta=0.20),
                    "T2": Channel("T2", alpha=0.91, beta=0.08)}

    def _f(self, **kw):
        base = dict(belief=[0.6, 0.3, 0.1], mode="EXPLORE", tier="T1", target=0,
                    channels=self.chs, n_exploit=0.0, n_verify=0.0)
        base.update(kw)
        return action_features(**base)

    def test_all_declared_features_present(self):
        f = self._f()
        self.assertEqual(set(f), set(ACTION_FEATURE_NAMES))

    def test_belief_in_target_is_read_from_the_belief_vector(self):
        self.assertAlmostEqual(self._f(target=1)["belief_in_target"], 0.3, places=9)

    def test_info_gain_only_applies_to_explore(self):
        self.assertGreater(self._f(mode="EXPLORE")["expected_info_gain"], 0.0)
        self.assertEqual(self._f(mode="EXPLOIT")["expected_info_gain"], 0.0)
        self.assertEqual(self._f(mode="VERIFY", target=None)["expected_info_gain"], 0.0)

    def test_mode_indicators_are_mutually_exclusive(self):
        for mode in ("EXPLORE", "EXPLOIT", "VERIFY"):
            f = self._f(mode=mode)
            self.assertEqual(f["is_explore"] + f["is_exploit"] + f["is_verify"], 1.0)

    def test_unverified_patch_flag_tracks_edits_without_tests(self):
        self.assertEqual(self._f(n_exploit=2, n_verify=1)["has_unverified_patch"], 1.0)
        self.assertEqual(self._f(n_exploit=1, n_verify=1)["has_unverified_patch"], 0.0)

    def test_target_is_argmax_flag(self):
        self.assertEqual(self._f(target=0)["target_is_argmax"], 1.0)
        self.assertEqual(self._f(target=2)["target_is_argmax"], 0.0)

    def test_tier_index_is_monotone_and_lr_increases_with_tier(self):
        idx = [self._f(tier=t)["tier_index"] for t in ("T0", "T1", "T2")]
        lrs = [self._f(tier=t)["channel_lr_plus"] for t in ("T0", "T1", "T2")]
        self.assertEqual(idx, [0.0, 1.0, 2.0])
        self.assertLess(lrs[0], lrs[1])
        self.assertLess(lrs[1], lrs[2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
