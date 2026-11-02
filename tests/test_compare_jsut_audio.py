import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
import torch

from sparc.compare_jsut_audio import main, select_test
from sparc.model_jsut_tokens import checkpoint_hash
from sparc.trajectory_vq import CHANNELS, TrajectoryVQVAE, overlap_average


class AudioComparisonTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "run"
        self.features = self.root / "features"
        self.jsut = self.root / "jsut"
        self.output = self.root / "listening"
        self.relative = "basic5000/BASIC5000_0002.npz"
        self.run.mkdir()
        (self.run / "best.pt").write_bytes(b"fixture checkpoint")
        self.source = self.jsut / "basic5000/wav/BASIC5000_0002.wav"
        self.source.parent.mkdir(parents=True)
        self.waveform = np.sin(2 * np.pi * 180 * np.arange(48000) / 48000)
        sf.write(self.source, self.waveform * 0.1, 48000)
        original = self.features / self.relative
        original.parent.mkdir(parents=True)
        self.ema = np.ones((50, 12), dtype=np.float32)
        self.reconstruction = self.ema * 2
        self.pitch = np.full((50, 1), 180, dtype=np.float32)
        np.savez_compressed(original, ema=self.ema, pitch=self.pitch,
                            periodicity=np.ones((50, 1)), loudness=np.ones((50, 1)) * 0.2,
                            source=np.str_("basic5000/wav/BASIC5000_0002.wav"),
                            channels=np.asarray(CHANNELS), time=np.arange(50) / 50,
                            frame_rate=50, text=np.str_("音声<&>の比較。"))
        config = {"features_dir": str(self.features), "files": {
            self.relative: {"size": original.stat().st_size, "mtime_ns": original.stat().st_mtime_ns}}}
        (self.run / "config.json").write_text(json.dumps(config))
        (self.run / "splits.json").write_text(json.dumps({
            "train": ["basic5000/BASIC5000_0001.npz"], "validation": [], "test": [self.relative]}))
        self.token = self.run / "tokens" / self.relative
        self.token.parent.mkdir(parents=True)
        np.savez_compressed(self.token, reconstruction=self.reconstruction,
                            time=np.arange(50) / 50, split=np.str_("test"),
                            feature_path=np.str_(self.relative),
                            checkpoint_sha256=np.str_(checkpoint_hash(self.run / "best.pt")))
        self.coder = Mock(ft_sr=50, output_sr=16000)
        self.coder.inverter.return_value = {"wav": object(), "acoustics": np.zeros((1, 50, 1024))}
        self.coder.speaker_encoder.return_value = {"spk_emb": np.ones((1, 64))}
        self.coder.decode.side_effect = [self.waveform[::3] * 0.9, self.waveform[::3] * 1.8]
        model = patch("sparc.compare_jsut_audio.load_model", return_value=self.coder)
        self.load_model = model.start()
        self.addCleanup(model.stop)

    def cli(self, *arguments):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(["--run-dir", str(self.run), "--jsut-dir", str(self.jsut),
                         "--output-dir", str(self.output), "--device", "cpu", *map(str, arguments)])

    def test_comparison_changes_only_ema_and_preserves_source_and_auxiliary_features(self):
        original_bytes = self.source.read_bytes()
        self.assertEqual(self.cli(), 0)
        calls = self.coder.decode.call_args_list
        np.testing.assert_array_equal(calls[0].kwargs["ema"], self.ema)
        np.testing.assert_array_equal(calls[1].kwargs["ema"], self.reconstruction)
        for key in ("pitch", "loudness", "spk_emb"):
            np.testing.assert_array_equal(calls[0].kwargs[key], calls[1].kwargs[key])
        np.testing.assert_array_equal(calls[0].kwargs["pitch"], self.pitch)
        self.coder.source_extractor.assert_not_called()
        self.coder.encode.assert_not_called()
        self.assertTrue(self.load_model.call_args.kwargs["config"]["zero_pad"])
        directory = self.output / "basic5000/BASIC5000_0002"
        self.assertEqual(self.source.read_bytes(), original_bytes)
        self.assertEqual((directory / "original.wav").read_bytes(), original_bytes)
        baseline, sample_rate = sf.read(directory / "original_ema.wav")
        reconstructed, second_rate = sf.read(directory / "reconstructed_ema.wav")
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(second_rate, sample_rate)
        self.assertEqual(len(baseline), len(reconstructed))
        self.assertLessEqual(np.max(np.abs(reconstructed)), 0.991)
        np.testing.assert_allclose(baseline * 2, reconstructed, atol=2 / 32768)
        combined, _ = sf.read(directory / "comparison.wav")
        self.assertEqual(len(combined), 4 * 16000)  # Three 1s clips, two 0.5s gaps.
        np.testing.assert_array_equal(combined[16000:24000], 0)
        with np.load(directory / "conditions.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["pitch"], self.pitch)
            np.testing.assert_array_equal(data["ema"], self.ema)
        report = (self.output / "index.html").read_text()
        self.assertIn("音声&lt;&amp;&gt;の比較。", report)
        self.assertEqual(report.count("<audio "), 4)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["records"][0]["split"], "test")
        self.assertLess(manifest["records"][0]["synthesis_common_gain"], 1)

    def test_training_utterances_are_rejected_before_loading_model(self):
        with self.assertRaises(SystemExit):
            self.cli("--utterances", "BASIC5000_0001")
        self.load_model.assert_not_called()

    def test_weighted_comparison_uses_saved_tokens_and_does_not_modify_run(self):
        torch.manual_seed(42)
        vq = TrajectoryVQVAE(codebook_size=2, latent_dim=2)
        mean, std = np.zeros(12, dtype=np.float32), np.ones(12, dtype=np.float32)
        torch.save({"state_dict": vq.state_dict(), "mean": torch.from_numpy(mean),
                    "std": torch.from_numpy(std)}, self.run / "best.pt")
        config_path = self.run / "config.json"
        config = json.loads(config_path.read_text())
        config.update(codebook_size=2, latent_dim=2)
        config_path.write_text(json.dumps(config))
        tokens = np.tile([0, 1], 25).astype(np.int64)
        with torch.inference_mode():
            windows = vq.decoder(vq.codebook.weight).numpy().reshape(2, 5, 12)[tokens]
        uniform = overlap_average(windows)
        expected = overlap_average(windows, [1, 2, 3, 2, 1])
        with np.load(self.token, allow_pickle=False) as stored:
            arrays = dict(stored)
        arrays.update(tokens=tokens, reconstruction=uniform,
                      checkpoint_sha256=np.str_(checkpoint_hash(self.run / "best.pt")))
        np.savez_compressed(self.token, **arrays)
        before = self.token.read_bytes()
        model_before = (self.run / "best.pt").read_bytes()
        self.coder.decode.side_effect = [self.waveform[::3] * gain for gain in (0.4, 0.5, 0.6)]
        self.assertEqual(self.cli("--include-weighted"), 0)
        calls = self.coder.decode.call_args_list
        self.assertEqual(len(calls), 3)
        np.testing.assert_allclose(calls[1].kwargs["ema"], uniform)
        np.testing.assert_allclose(calls[2].kwargs["ema"], expected)
        self.assertFalse(np.allclose(expected, uniform))
        for key in ("pitch", "loudness", "spk_emb"):
            np.testing.assert_array_equal(calls[1].kwargs[key], calls[2].kwargs[key])
        self.assertEqual(self.token.read_bytes(), before)
        self.assertEqual((self.run / "best.pt").read_bytes(), model_before)
        directory = self.output / "basic5000/BASIC5000_0002"
        self.assertTrue((directory / "weighted_ema.wav").is_file())
        audio, rate = sf.read(directory / "weighted_ab.wav")
        self.assertEqual(rate, 16000)
        self.assertEqual(len(audio), 40000)
        with np.load(directory / "conditions.npz", allow_pickle=False) as data:
            np.testing.assert_allclose(data["weighted_reconstruction"], expected)
            np.testing.assert_array_equal(data["overlap_weights"], [1, 2, 3, 2, 1])
        self.assertEqual((self.output / "index.html").read_text().count("<audio "), 6)

    def test_stale_reconstruction_is_rejected_before_loading_model(self):
        with np.load(self.token, allow_pickle=False) as data:
            arrays = dict(data)
        arrays["checkpoint_sha256"] = np.str_("other model")
        np.savez_compressed(self.token, **arrays)
        with self.assertRaises(SystemExit):
            self.cli()
        self.load_model.assert_not_called()

    def test_existing_output_is_protected(self):
        self.output.mkdir()
        (self.output / "keep.txt").write_text("keep")
        with self.assertRaises(SystemExit):
            self.cli()
        self.load_model.assert_not_called()
        self.assertEqual((self.output / "keep.txt").read_text(), "keep")

    def test_selection_limits_test_utterances(self):
        splits = {"test": ["basic5000/A.npz", "basic5000/B.npz"]}
        self.assertEqual(select_test(splits, None, 1), ["basic5000/A.npz"])
        self.assertEqual(select_test(splits, ["B"], 5), ["basic5000/B.npz"])


if __name__ == "__main__":
    unittest.main()
