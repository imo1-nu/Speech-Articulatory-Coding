import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from sparc.compare_jsut_token_rates import build_report, validate_runs
from sparc.model_jsut_tokens import checkpoint_hash, write_json
from sparc.trajectory_vq import CHANNELS


class TokenRateReportTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runs = {rate: self.root / str(rate) for rate in (50, 25)}
        self.relative = "basic5000/A.npz"
        self.wave = 0.2 * np.sin(2 * np.pi * np.arange(960) / 50)
        for rate, run in self.runs.items():
            folder = run / "audio_comparison/basic5000/A"
            folder.mkdir(parents=True)
            (run / "best.pt").write_bytes(str(rate).encode())
            config = {"files": {self.relative: {"frames": 3}}, "seed": 42, "codebook_size": 128,
                      "latent_dim": 16, "batch_size": 512, "learning_rate": 0.001,
                      "commitment": 0.25, "max_epochs": 100, "patience": 10,
                      "window_frames": 2, "window_left_frames": 0, "window_right_frames": 1,
                      "token_rate": rate, "token_stride": 50 // rate, "channels": list(CHANNELS)}
            write_json(run / "config.json", config)
            write_json(run / "splits.json", {"train": [], "validation": [], "test": [self.relative]})
            np.savez_compressed(run / "normalization.npz", mean=np.zeros(12), std=np.ones(12))
            metrics = {"normalized_mse": 0.1, "delta_mse": 0.1, "velocity_energy_ratio": 1.0,
                       "trajectory_6_to_10_hz_energy_ratio": 1.0, "frames": 3,
                       "used_codes": 2, "code_perplexity": 2.0, "tokens": (3 if rate == 50 else 2)}
            digest = checkpoint_hash(run / "best.pt")
            write_json(run / "window_evaluation.json", {"checkpoint_sha256": digest,
                       "splits": {name: {"uniform": metrics} for name in ("validation", "test")}})
            gain = 0.5 if rate == 50 else 0.25
            for name in ("original", "original_ema", "reconstructed_ema"):
                sf.write(folder / f"{name}.wav", self.wave * gain, 16000, subtype="PCM_16")
            np.savez_compressed(folder / "conditions.npz", pitch=np.ones((3, 1)) * 180,
                                loudness=np.ones((3, 1)), spk_emb=np.ones(64),
                                ema=np.zeros((3, 12)), reconstruction=np.ones((3, 12)) / rate,
                                time=np.arange(3) / 50)
            audio = {key: f"basic5000/A/{key}.wav" for key in ("original", "original_ema", "reconstructed_ema")}
            write_json(run / "audio_comparison/manifest.json", {"checkpoint_sha256": digest,
                       "records": [{"feature_path": self.relative, "utterance_id": "A", "text": "比較<&>",
                                    "synthesis_common_gain": gain, "audio": audio}]})

    def test_report_matches_levels_and_preserves_reference_audio(self):
        source = self.runs[50] / "audio_comparison/basic5000/A/reconstructed_ema.wav"
        original = source.read_bytes()
        result = build_report(self.runs[50], self.runs[25], self.root / "report", audio_limit=1)
        self.assertEqual([row["test_tokens"] for row in result["rows"]], [3, 2])
        self.assertEqual([row["nominal_token_id_bps"] for row in result["rows"]], [350, 175])
        folder = self.root / "report/listening/basic5000/A"
        a, rate = sf.read(folder / "tokens_50hz.wav")
        b, _ = sf.read(folder / "tokens_25hz.wav")
        np.testing.assert_allclose(a, b, atol=2 / 32768)
        self.assertEqual(rate, 16000)
        combined, _ = sf.read(folder / "ab_50_then_25.wav")
        self.assertEqual(len(combined), 2 * 960 + 8000)
        self.assertEqual(source.read_bytes(), original)
        report = (self.root / "report/index.html").read_text()
        self.assertIn("比較&lt;&amp;&gt;", report)
        self.assertEqual(report.count("<audio "), 5)
        self.assertTrue((self.root / "report/trajectory_comparison.png").is_file())

    def test_mismatched_splits_and_stale_evaluations_are_rejected(self):
        run25 = self.runs[25]
        splits = run25 / "splits.json"
        original = splits.read_bytes()
        write_json(splits, {"train": [], "validation": [], "test": ["other.npz"]})
        with self.assertRaises(ValueError):
            validate_runs(self.runs[50], run25)
        splits.write_bytes(original)
        path = run25 / "window_evaluation.json"
        data = json.loads(path.read_text())
        data["checkpoint_sha256"] = "stale"
        write_json(path, data)
        with self.assertRaises(ValueError):
            build_report(self.runs[50], run25, self.root / "report", audio_limit=1)


if __name__ == "__main__":
    unittest.main()
