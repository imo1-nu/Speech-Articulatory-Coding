"""Verify that raw experiments never filter finite acoustic estimates."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

from experiments.feedback.control_run import Evaluator, primary_from
from experiments.feedback.raw_acoustics import extract_raw_acoustics, raw_summary


SETTINGS = yaml.safe_load(Path("configs/feedback_experiment1.yaml").read_text())["formants"]


class RawAcousticsTests(unittest.TestCase):
    def test_raw_median_uses_all_finite_values_without_thresholds(self):
        result = raw_summary([50, 5000, 6000, np.nan])
        self.assertEqual(result["target_hz"], 5000)
        self.assertEqual(result["available_frames"], 3)
        self.assertEqual(result["total_frames"], 4)
        self.assertIsNone(raw_summary([np.nan, np.inf])["target_hz"])

    def make_sound(self):
        sound = Mock()
        sound.to_formant_burg.return_value.get_value_at_time.side_effect = lambda j, t: [1800, 100, 6000][j-1]
        sound.to_formant_burg.return_value.get_bandwidth_at_time.return_value = 3000
        sound.to_pitch_ac.return_value.get_value_at_time.return_value = np.nan
        return sound

    def test_unvoiced_large_bandwidth_out_of_range_unordered_values_are_kept(self):
        with patch("experiments.feedback.raw_acoustics.parselmouth.Sound", return_value=self.make_sound()):
            data = extract_raw_acoustics(np.sin(np.arange(16000)), 16000, .3, .34, SETTINGS)
        self.assertEqual(len(data["time_s"]), 4)  # below previous minimum 10
        np.testing.assert_array_equal(data["filtered_hz"], data["raw_hz"])
        self.assertTrue(data["valid"].all())
        primary, observed = primary_from(data, ["F1", "F2"])
        np.testing.assert_array_equal(primary, [1800, 100])
        self.assertTrue(np.isnan(observed[0]))
        self.assertEqual(observed[3], 6000)
        self.assertFalse(data["summary"]["quality_filtering"])

    def test_evaluator_does_not_apply_tracking_or_quality_rejection_in_raw_mode(self):
        code = {"ema": np.zeros((50, 12), np.float32), "pitch": np.full((50, 1), 125.),
                "loudness": np.ones((50, 1)), "spk_emb": np.ones(64)}
        config = {"feature_policy": "raw", "primary_features": ["F1", "F2"]}
        coder = Mock()
        coder.output_sr = 16000
        coder.decode.return_value = np.sin(np.arange(16000)).astype(np.float32)
        with tempfile.TemporaryDirectory() as folder, patch(
                "experiments.feedback.raw_acoustics.parselmouth.Sound", return_value=self.make_sound()), patch(
                "experiments.feedback.control_run.extract_formants", side_effect=AssertionError("Quality extraction used")):
            ev = Evaluator(coder, code, np.ones(12), np.ones(50),
                           {"start_s": .3, "end_s": .34}, SETTINGS, config, Path(folder), "cpu")
            first = ev.evaluate(np.zeros(12), "baseline")
            ev.baseline = first
            second = ev.evaluate(np.ones(12) * .01, "perturbed")
            np.testing.assert_array_equal(second["a"], [1800, 100])
            self.assertIsNone(ev.records[-1]["tracking_ok"])
            self.assertIsNone(ev.records[-1]["quality"])
            self.assertIsNone(ev.records[-1]["rejection_reasons"])
            ev.log.close()


if __name__ == "__main__":
    unittest.main()
