"""Model-free checks; these tests never load SPARC weights or require a GPU."""

from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.signal import lfilter
import yaml

from experiments.feedback.adapter import encode_decode
from experiments.feedback.acoustics import compare_acoustics, feature_arrays
from experiments.feedback.formants import FormantExtractionError, extract_formants
from experiments.feedback.run import measure_and_save


SETTINGS = yaml.safe_load(Path("configs/feedback_experiment1.yaml").read_text())["formants"]


class FormantTests(unittest.TestCase):
    def test_silent_nonfinite_or_out_of_bounds_fails_explicitly(self):
        for waveform, start, end in [
            (np.zeros(16000), .2, .6),
            (np.full(16000, np.nan), .2, .6),
            (np.ones((16000, 2)), .2, .6),
            (np.sin(np.arange(16000)), .8, 1.2),
        ]:
            with self.subTest(start=start, end=end), self.assertRaises(FormantExtractionError):
                extract_formants(waveform, 16000, start, end, SETTINGS)

    def test_extraction_error_is_saved_as_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaises(FormantExtractionError):
                measure_and_save(path, "silent", np.zeros(16000), 16000, .3, .7, SETTINGS)
            error = json.loads((path / "silent_formants_error.json").read_text())
            self.assertEqual(error["status"], "extraction_failed")
            self.assertIn("non-silent", error["error"])

    def test_known_source_filter_vowel_has_accurate_formants(self):
        sr = 16000
        signal = np.zeros(sr)
        signal[::128] = 1  # 125 Hz periodic excitation
        expected = [700, 1200, 2600]
        # Match the five-resonance model order used by Burg. A three-resonance
        # signal with a five-formant fit introduces spurious broad LPC poles.
        for frequency, bandwidth in zip(expected + [3500, 4300], [70, 90, 130, 200, 250]):
            radius = np.exp(-np.pi * bandwidth / sr)
            signal = lfilter([1], [1, -2 * radius * np.cos(2 * np.pi * frequency / sr), radius ** 2], signal)
        signal /= np.max(np.abs(signal))
        data = extract_formants(signal, sr, .3, .7, SETTINGS)
        self.assertGreater(data["summary"]["voiced_fraction"], .9)
        self.assertEqual(data["summary"]["f0"]["status"], "ok")
        self.assertAlmostEqual(data["summary"]["f0"]["target_hz"], 125, delta=1)
        self.assertTrue(all(f["status"] == "ok" for f in data["summary"]["formants"]))
        np.testing.assert_allclose([f["target_hz"] for f in data["summary"]["formants"]],
                                   expected, atol=100, rtol=0)

    def test_missing_formant_remains_nan_and_has_no_target(self):
        sound = Mock()
        formant = sound.to_formant_burg.return_value
        formant.get_value_at_time.side_effect = lambda j, t: [700, np.nan, 2600][j - 1]
        formant.get_bandwidth_at_time.return_value = 100
        sound.to_pitch_ac.return_value.get_value_at_time.return_value = 125
        with patch("experiments.feedback.formants.parselmouth.Sound", return_value=sound):
            data = extract_formants(np.sin(np.arange(16000)), 16000, .3, .7, SETTINGS)
        self.assertTrue(np.isnan(data["filtered_hz"][:, 1]).all())
        self.assertIsNone(data["summary"]["formants"][1]["target_hz"])
        self.assertTrue(all("missing" in row[1] for row in data["reasons"]))
        self.assertEqual(data["summary"]["f0"]["target_hz"], 125)
        self.assertEqual(feature_arrays(data)["raw_hz"].shape, (40, 4))
        comparison = compare_acoustics(data, data, .8)
        self.assertEqual(comparison["feature_names"], ["F0", "F1", "F2", "F3"])
        self.assertEqual(comparison["baseline_target_hz"], [125, 700, None, 2600])
        self.assertFalse(comparison["baseline_target_usable"])
        self.assertEqual(comparison["comparison"][0]["paired_rmse_hz"], 0)
        self.assertIsNone(comparison["comparison"][2]["median_difference_hz"])

    def test_f0_comparison_uses_shared_voiced_times_and_keeps_missing_values(self):
        sound = Mock()
        sound.to_formant_burg.return_value.get_value_at_time.side_effect = lambda j, t: [700, 1200, 2600][j - 1]
        sound.to_formant_burg.return_value.get_bandwidth_at_time.return_value = 100
        sound.to_pitch_ac.return_value.get_value_at_time.side_effect = lambda t: np.nan if t < .32 else 125
        with patch("experiments.feedback.formants.parselmouth.Sound", return_value=sound):
            data = extract_formants(np.sin(np.arange(16000)), 16000, .3, .7, SETTINGS)
        self.assertTrue(np.isnan(data["f0_filtered_hz"][:2]).all())
        self.assertEqual(data["summary"]["f0"]["valid_frames"], 38)
        baseline = copy.deepcopy(data)
        baseline["f0_hz"] *= 1.1
        baseline["f0_filtered_hz"] *= 1.1
        baseline["summary"]["f0"]["target_hz"] *= 1.1
        compared = compare_acoustics(data, baseline, .8)["comparison"][0]
        self.assertEqual(compared["paired_valid_frames"], 38)
        self.assertAlmostEqual(compared["median_difference_hz"], 12.5)
        self.assertAlmostEqual(compared["paired_mae_hz"], 12.5)
        self.assertAlmostEqual(compared["paired_rmse_hz"], 12.5)
        self.assertAlmostEqual(compared["median_difference_cents"], 1200 * np.log2(1.1))
        with self.assertRaisesRegex(ValueError, "same analysis times"):
            compare_acoustics(data, dict(baseline, time_s=baseline["time_s"] + .01), .8)

    def test_f0_without_voiced_frames_has_no_target_or_error(self):
        sound = Mock()
        sound.to_formant_burg.return_value.get_value_at_time.side_effect = lambda j, t: [700, 1200, 2600][j - 1]
        sound.to_formant_burg.return_value.get_bandwidth_at_time.return_value = 100
        sound.to_pitch_ac.return_value.get_value_at_time.return_value = np.nan
        with patch("experiments.feedback.formants.parselmouth.Sound", return_value=sound):
            data = extract_formants(np.sin(np.arange(16000)), 16000, .3, .7, SETTINGS)
        self.assertEqual(data["summary"]["f0"]["valid_frames"], 0)
        self.assertIsNone(data["summary"]["f0"]["target_hz"])
        self.assertTrue(np.isnan(data["f0_filtered_hz"]).all())
        compared = compare_acoustics(data, data, .8)["comparison"][0]
        self.assertIsNone(compared["median_difference_hz"])
        self.assertIsNone(compared["paired_rmse_hz"])

    def test_noise_is_not_a_voiced_formant_target(self):
        signal = np.random.default_rng(42).normal(size=16000)
        data = extract_formants(signal, 16000, .3, .7, SETTINGS)
        self.assertTrue(all(f["target_hz"] is None for f in data["summary"]["formants"]))

    def test_bandwidth_and_local_outliers_are_recorded(self):
        sound = Mock()
        formant = sound.to_formant_burg.return_value
        formant.get_value_at_time.side_effect = lambda j, t: (
            1100 if j == 1 and abs(t - .405) < .001 else [700, 1200, 2600][j - 1])
        formant.get_bandwidth_at_time.side_effect = lambda j, t: 1000 if j == 2 else 100
        sound.to_pitch_ac.return_value.get_value_at_time.return_value = 125
        with patch("experiments.feedback.formants.parselmouth.Sound", return_value=sound):
            data = extract_formants(np.sin(np.arange(16000)), 16000, .3, .7, SETTINGS)
        self.assertTrue(any("local_instability" in row[0] for row in data["reasons"]))
        self.assertTrue(all("bandwidth" in row[1] for row in data["reasons"]))
        self.assertIsNone(data["summary"]["formants"][1]["target_hz"])


class AdapterTests(unittest.TestCase):
    def make_coder(self):
        coder = Mock()
        coder.encode.return_value = {
            "ema": np.ones((50, 12)), "pitch": np.full((50, 1), 125),
            "loudness": np.ones((50, 1)), "periodicity": np.ones((50, 1)),
            "pitch_stats": np.array([125, 1]), "spk_emb": np.ones(64), "ft_len": np.int64(50),
        }
        coder.decode.return_value = np.sin(np.arange(16000)).astype(np.float32)
        return coder

    def test_checks_repeat_decode_and_all_conditions(self):
        coder = self.make_coder()
        _, _, timing, checks = encode_decode(coder, Path("unused.wav"), "cpu")
        self.assertEqual(coder.decode.call_count, 2)
        self.assertEqual(len(timing["decode_s"]), 2)
        self.assertTrue(checks["decode_bitwise_equal"])
        self.assertTrue(checks["code_unchanged"])
        self.assertEqual(checks["decode_max_abs_difference"], 0)

    def test_condition_mutation_is_detected(self):
        coder = self.make_coder()
        def decode(**code):
            code["pitch"][0, 0] += 1
            return np.ones(16000)
        coder.decode.side_effect = decode
        with self.assertRaisesRegex(RuntimeError, "changed condition pitch"):
            encode_decode(coder, Path("unused.wav"), "cpu")


if __name__ == "__main__":
    unittest.main()
