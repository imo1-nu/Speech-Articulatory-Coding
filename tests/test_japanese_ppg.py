import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from sparc.extract_jsut_ppg import main
from sparc.japanese_ppg import (
    HOP, MODEL_REVISION, PHONEMES, JapaneseLogMel, JapanesePPGExtractor,
    JapanesePPGModel, compare_posteriors, frame_times, js_divergence, load_audio, sha256_file,
)


class JapanesePPGMathTest(unittest.TestCase):
    def test_js_divergence_identity_symmetry_and_known_maximum(self):
        first, second = np.zeros((2, 35)), np.zeros((2, 35))
        first[:, 0], second[:, 1] = 1, 1
        np.testing.assert_array_equal(js_divergence(first, first), np.zeros(2))
        np.testing.assert_allclose(js_divergence(first, second), np.log(2))
        np.testing.assert_array_equal(js_divergence(first, second), js_divergence(second, first))
        invalid = first.copy()
        invalid[0, 0] = -1
        with self.assertRaises(ValueError):
            js_divergence(first, invalid)

    def test_comparison_uses_common_absolute_time_without_stretching(self):
        first = np.full((100, 35), 1 / 35)
        second = np.full((98, 35), 1 / 35)
        time, distance = compare_posteriors(first, frame_times(100), second, frame_times(98))
        self.assertEqual(len(distance), 98)
        np.testing.assert_array_equal(time, frame_times(98))
        self.assertAlmostEqual(time[0], 0.005)
        self.assertAlmostEqual(time[-1], 0.975)
        np.testing.assert_array_equal(distance, np.zeros(98))
        with self.assertRaises(ValueError):
            compare_posteriors(first, frame_times(100), second, frame_times(98) + 0.01)

    def test_stereo_resampling_lengths_and_silence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stereo.wav"
            waveform = np.sin(np.arange(48001) / 30) * 0.1
            sf.write(path, np.column_stack((waveform, waveform * 0.5)), 48000)
            audio, info = load_audio(path)
            self.assertEqual(tuple(audio.shape), (1, 1, 16001))
            self.assertEqual(info["source_channels"], 2)
            self.assertEqual(info["source_samples"], 48001)
            self.assertEqual(info["resampled_samples"], 16001)
            self.assertEqual(info["source_sample_rate"], 48000)
            silent = Path(temporary) / "silent.wav"
            sf.write(silent, np.zeros(1600), 16000)
            audio, _ = load_audio(silent)
            frontend = JapaneseLogMel()
            mel = frontend(audio)
            self.assertEqual(tuple(mel.shape), (1, 80, 10))
            self.assertTrue(torch.isfinite(mel).all())
            self.assertEqual(len(frame_times(audio.shape[-1] // HOP)), 10)

    def test_model_has_35_outputs_and_masks_padding(self):
        torch.set_num_threads(1)
        model = JapanesePPGModel().eval()
        with torch.inference_mode():
            values = model(torch.zeros(2, 80, 12), torch.tensor([12, 9]))
            self.assertEqual(tuple(values.shape), (2, len(PHONEMES), 12))
            torch.testing.assert_close(values[1, :, 9:], torch.zeros(35, 3))
            self.assertTrue(torch.isfinite(values).all())
            torch.testing.assert_close(values.softmax(1).sum(1), torch.ones(2, 12))

    def test_wrong_language_head_is_rejected_without_loading_a_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wrong.pt"
            torch.save({"model": {"output_layer.weight": torch.zeros(40, 256, 5)}}, path)
            with patch("sparc.japanese_ppg.hf_hub_download", return_value=str(path)):
                with self.assertRaisesRegex(ValueError, "35-phone"):
                    JapanesePPGExtractor()


class FakeExtractor:
    checkpoint_sha256 = "fixture-model-sha256"

    def extract(self, audio):
        frames = audio.shape[-1] // HOP
        first_probability = 0.5 + float(audio.abs().mean())
        result = np.full((frames, 35), (1 - first_probability) / 34, dtype=np.float32)
        result[:, 0] = first_probability
        return result


class JapanesePPGCLITest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "run"
        self.output = self.root / "ppg"
        folder = self.run / "audio_comparison/basic5000/A"
        folder.mkdir(parents=True)
        self.audio_paths = {}
        for variant, sample_rate, samples, amplitude in (
            ("original", 48000, 48000, 0.1), ("original_ema", 16000, 15680, 0.2),
            ("reconstructed_ema", 16000, 15680, 0.3)
        ):
            path = folder / f"{variant}.wav"
            waveform = amplitude * np.sin(2 * np.pi * 180 * np.arange(samples) / sample_rate)
            sf.write(path, waveform, sample_rate)
            self.audio_paths[variant] = path
        (self.run / "best.pt").write_bytes(b"fixture VQ model")
        (self.run / "config.json").write_text(json.dumps({"window_frames": 2, "codebook_size": 128}))
        (self.run / "splits.json").write_text(json.dumps({"test": ["basic5000/A.npz"]}))
        manifest = {"checkpoint_sha256": sha256_file(self.run / "best.pt"), "records": [{
            "utterance_id": "A", "feature_path": "basic5000/A.npz", "text": "テスト<&>", "split": "test",
            "audio": {key: f"basic5000/A/{key}.wav" for key in self.audio_paths}}]}
        self.manifest = self.run / "audio_comparison/manifest.json"
        self.manifest.write_text(json.dumps(manifest))
        model = patch("sparc.extract_jsut_ppg.JapanesePPGExtractor", return_value=FakeExtractor())
        self.load_model = model.start()
        self.addCleanup(model.stop)

    def cli(self, *arguments):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(["--run-dir", str(self.run), "--output-dir", str(self.output),
                         "--device", "cpu", "--threads", "1", *map(str, arguments)])

    def test_generates_three_ppgs_common_time_metrics_and_report_without_changing_sources(self):
        before = {variant: path.read_bytes() for variant, path in self.audio_paths.items()}
        self.assertEqual(self.cli(), 0)
        directory = self.output / "basic5000/A"
        for variant, frames in (("original", 100), ("original_ema", 98), ("reconstructed_ema", 98)):
            with np.load(directory / f"{variant}.npz", allow_pickle=False) as data:
                self.assertEqual(data["ppg"].shape, (frames, 35))
                np.testing.assert_allclose(data["ppg"].sum(1), 1, atol=1e-6)
                np.testing.assert_array_equal(data["phonemes"], PHONEMES)
                np.testing.assert_array_equal(data["time"], frame_times(frames))
                self.assertEqual(data["frame_rate"], 100)
                self.assertEqual(str(data["model_revision"]), MODEL_REVISION)
                self.assertEqual(str(data["audio_sha256"]), sha256_file(self.audio_paths[variant]))
            self.assertEqual(self.audio_paths[variant].read_bytes(), before[variant])
            self.assertEqual((directory / f"{variant}.wav").read_bytes(), before[variant])
        with np.load(directory / "comparisons.npz", allow_pickle=False) as data:
            self.assertEqual(len(data["original__reconstructed_ema_js_nats"]), 98)
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["utterances"], 1)
        self.assertEqual(len(summary["records"]), 3)
        self.assertTrue(all(record["frames"] == 98 for record in summary["records"]))
        self.assertTrue((directory / "ppg_comparison.png").is_file())
        report = (self.output / "index.html").read_text()
        self.assertEqual(report.count("<audio "), 3)
        self.assertIn("テスト&lt;&amp;&gt;", report)
        self.assertEqual((self.run / "best.pt").read_bytes(), b"fixture VQ model")

    def test_existing_output_is_preserved_before_model_download(self):
        self.output.mkdir()
        (self.output / "keep.txt").write_text("keep")
        with self.assertRaises(SystemExit):
            self.cli()
        self.assertEqual((self.output / "keep.txt").read_text(), "keep")
        self.load_model.assert_not_called()

    def test_stale_audio_manifest_is_rejected_before_model_download(self):
        data = json.loads(self.manifest.read_text())
        data["checkpoint_sha256"] = "different model"
        self.manifest.write_text(json.dumps(data))
        with self.assertRaises(SystemExit):
            self.cli()
        self.load_model.assert_not_called()

    def test_invalid_audio_is_rejected_before_model_download(self):
        sf.write(self.audio_paths["original"], np.full(48000, np.nan), 48000, subtype="FLOAT")
        with self.assertRaises(SystemExit):
            self.cli()
        self.load_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
