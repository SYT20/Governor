"""The budget-conditional policy: interpolation behaviour and monotonicity."""
from __future__ import annotations
import unittest
from governor.arms.adaptive import FITTED, AdaptiveArm, thresholds_for
from governor.envs.families import BY_NAME


class TestThresholdInterpolation(unittest.TestCase):
    def test_fitted_points_are_reproduced_exactly(self):
        for scale, params in FITTED.items():
            got = thresholds_for(scale)
            self.assertAlmostEqual(got["confident"], params["confident"], places=9)
            self.assertEqual(got["max_exploits"], params["max_exploits"])

    def test_confidence_threshold_falls_as_budget_tightens(self):
        """With money, wait for near-certainty. Without, commit on a bare majority."""
        seq = [thresholds_for(s)["confident"] for s in (1.0, 0.75, 0.5, 0.35, 0.25)]
        self.assertTrue(all(a >= b for a, b in zip(seq, seq[1:])), seq)

    def test_reserve_grows_as_budget_tightens(self):
        seq = [thresholds_for(s)["low_budget"] for s in (1.0, 0.75, 0.5, 0.35, 0.25)]
        self.assertTrue(all(a <= b for a, b in zip(seq, seq[1:])), seq)

    def test_tier_escalation_switches_off_under_scarcity(self):
        """Escalation keys on fraction-of-remaining, which cannot distinguish 60%
        of a generous envelope from 60% of a starvation one. Measured cost at 25%
        budget was -12pp, so the fitted policy turns it off."""
        self.assertTrue(thresholds_for(1.0)["use_tier_escalation"])
        self.assertFalse(thresholds_for(0.25)["use_tier_escalation"])

    def test_extrapolates_beyond_fitted_range_without_error(self):
        for s in (0.01, 0.1, 1.5, 4.0):
            t = thresholds_for(s)
            self.assertGreater(t["confident"], 0.0)
            self.assertLessEqual(t["confident"], 1.0)

    def test_unseen_scales_lie_between_neighbouring_fits(self):
        mid = thresholds_for(0.75)["confident"]
        self.assertLess(mid, FITTED[1.00]["confident"])
        self.assertGreater(mid, FITTED[0.50]["confident"])


class TestAdaptiveArm(unittest.TestCase):
    def test_reset_required_before_act(self):
        arm = AdaptiveArm(scale=0.5)
        with self.assertRaises(AssertionError):
            arm.act(None, [])  # type: ignore[arg-type]

    def test_arm_picks_different_thresholds_per_envelope(self):
        task = BY_NAME["baseline"].task(1)
        lo, hi = AdaptiveArm(scale=0.25), AdaptiveArm(scale=1.0)
        lo.reset(task); hi.reset(task)
        self.assertLess(lo._inner.confident, hi._inner.confident)
        self.assertGreater(lo._inner.low_budget, hi._inner.low_budget)


if __name__ == "__main__":
    unittest.main(verbosity=2)
