"""A large rank change among low-ranked classes is not semantic generation."""

import unittest

from scripts.test_vlm_p_gradient_semantics import verdict


class GradientVerdictTest(unittest.TestCase):
    def summaries(self):
        first = dict(target_prob_mean=1e-6, target_rank_mean=66,
                     frac_probe_predicts_target=0, probe_target_rank_mean=801,
                     probe_top5_correct_mean=0, rms255_mean=0)
        last = dict(target_prob_mean=1, target_rank_mean=1,
                    frac_probe_predicts_target=0, probe_target_rank_mean=419,
                    probe_top5_correct_mean=0, rms255_mean=5)
        return [first, last]

    def test_large_low_rank_improvement_does_not_establish_semantics(self):
        result = verdict(self.summaries(), 100)
        self.assertEqual(result["label"], "P-CONFIDENCE-WITHOUT-PROBE-RECOGNITION")
        self.assertTrue(result["probe_moved"])
        self.assertFalse(result["probe_recognizes"])

    def test_probe_recognition_is_transfer_not_proof_of_semantics(self):
        summary = self.summaries()
        summary[-1]["probe_top5_correct_mean"] = 1
        self.assertEqual(verdict(summary, 100)["label"], "PROBE-TRANSFER")

    def test_unchanged_initial_q_is_expected_control(self):
        first = self.summaries()[0]
        self.assertEqual(verdict([first, first], 100, initial_q_control=True)["label"],
                         "EXPECTED-ZERO-FIELD")


if __name__ == "__main__":
    unittest.main()
