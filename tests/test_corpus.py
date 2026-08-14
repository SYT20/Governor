"""Stage 2 invariants: clustering-adjusted sample size and family separation."""

from __future__ import annotations

import random
import unittest

from governor.corpus.gate import effective_sample_size, intra_class_correlation
from governor.envs.families import BY_NAME, FAMILIES, HELDOUT, TRAIN, heldout_families


class TestEffectiveSampleSize(unittest.TestCase):
    """The correction the external review asked for (J.8), and its consequence."""

    def test_constant_within_cluster_gives_icc_one(self):
        """Episode outcome labels are constant within an episode by construction."""
        groups = [f"ep{i}" for i in range(50) for _ in range(8)]
        values = [float(i % 2) for i in range(50) for _ in range(8)]
        self.assertAlmostEqual(intra_class_correlation(values, groups), 1.0, places=6)

    def test_ess_collapses_to_cluster_count_when_icc_is_one(self):
        """400 checkpoints from 50 episodes carry the information of 50 observations.

        This is the whole point of replacing revision 1's raw-count gate: counting
        correlated checkpoints as independent lets a corpus declare itself ready on
        8x less information than it appears to have.
        """
        groups = [f"ep{i}" for i in range(50) for _ in range(8)]
        values = [float(i % 2) for i in range(50) for _ in range(8)]
        ess, icc, deff = effective_sample_size(values, groups)
        self.assertEqual(len(values), 400)
        self.assertAlmostEqual(icc, 1.0, places=6)
        self.assertAlmostEqual(deff, 8.0, places=6)
        self.assertAlmostEqual(ess, 50.0, places=6)

    def test_independent_data_keeps_full_sample_size(self):
        rng = random.Random(0)
        groups = [f"ep{i}" for i in range(60) for _ in range(6)]
        values = [float(rng.random() < 0.5) for _ in groups]
        ess, icc, _ = effective_sample_size(values, groups)
        self.assertLess(icc, 0.25)
        self.assertGreater(ess, 0.7 * len(values))

    def test_degenerate_inputs_do_not_explode(self):
        self.assertEqual(effective_sample_size([], []), (0.0, 0.0, 1.0))
        self.assertEqual(intra_class_correlation([1.0], ["a"]), 0.0)


class TestFamilies(unittest.TestCase):
    """Family split is the stand-in for SWE-bench's repo-level split (J.1)."""

    def test_train_and_heldout_are_disjoint(self):
        self.assertFalse(set(TRAIN) & set(HELDOUT))
        self.assertEqual(len(TRAIN) + len(HELDOUT), len(FAMILIES))

    def test_enough_training_families_for_the_gate(self):
        self.assertGreaterEqual(len(TRAIN), 8)

    def test_families_are_genuinely_distinct(self):
        """Cosmetic variants would inflate the diversity count without adding signal."""
        sigs = {
            (
                f.config.n_hypotheses,
                round(f.config.alpha[list(f.config.alpha)[1]], 3),
                round(f.config.p_fix[list(f.config.p_fix)[1]], 3),
                round(f.config.verify_acc[list(f.config.verify_acc)[1]], 3),
                round(f.config.work_mu["EXPLORE@T1"], 3),
            )
            for f in FAMILIES
        }
        self.assertEqual(len(sigs), len(FAMILIES))

    def test_heldout_families_are_reachable_and_runnable(self):
        for fam in heldout_families():
            task = fam.task(1)
            self.assertGreaterEqual(task.true_cause, 0)
            self.assertLess(task.true_cause, fam.config.n_hypotheses)

    def test_named_lookup_matches_tuple(self):
        for f in FAMILIES:
            self.assertIs(BY_NAME[f.name], f)


if __name__ == "__main__":
    unittest.main(verbosity=2)
