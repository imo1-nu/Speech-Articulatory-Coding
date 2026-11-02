import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from sparc.extract_jsut import CHANNELS, main


class JSUTExtractionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.jsut = self.root / "jsut_ver1.1"
        self.output = self.root / "output"
        self.source = self.make_audio("basic5000", "BASIC5000_0001")
        (self.source.parent.parent / "transcript_utf8.txt").write_text(
            "BASIC5000_0001:テスト:音声です。\n", encoding="utf-8"
        )
        self.coder = Mock(ft_sr=50)
        self.coder.inverter.return_value = {
            "wav": object(), "ema": np.ones((1, 49, 12), dtype=np.float32)
        }

        def extract_source(waveform, outputs):
            return dict(outputs, pitch=np.full((1, 50, 1), 180.0),
                        periodicity=np.full((1, 50, 1), 0.8),
                        loudness=np.full((1, 50, 1), 0.2))

        self.coder.source_extractor.side_effect = extract_source
        model = patch("sparc.extract_jsut.load_model", return_value=self.coder)
        self.load_model = model.start()
        self.addCleanup(model.stop)

    def make_audio(self, subset, utterance):
        source = self.jsut / subset / "wav" / f"{utterance}.wav"
        source.parent.mkdir(parents=True, exist_ok=True)
        waveform = 0.2 * np.sin(2 * np.pi * 180 * np.arange(48000) / 48000)
        sf.write(source, np.column_stack((waveform, waveform)), 48000)
        return source

    def run_cli(self, *arguments):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(["--jsut-dir", str(self.jsut), "--output-dir", str(self.output),
                         "--device", "cpu", *map(str, arguments)])

    def destination(self, utterance="BASIC5000_0001", subset="basic5000"):
        return self.output / subset / f"{utterance}.npz"

    def records(self):
        return [json.loads(line) for line in (self.output / "manifest.jsonl").read_text().splitlines()]

    def test_saves_aligned_ema_pitch_and_utf8_metadata(self):
        self.assertEqual(self.run_cli(), 0)
        self.coder.inverter.assert_called_once_with(self.source, outputs={})
        self.coder.source_extractor.assert_called_once()
        self.coder.encode.assert_not_called()
        self.coder.decode.assert_not_called()
        self.assertEqual(self.load_model.call_args.args, ("feature_extraction",))
        with np.load(self.destination(), allow_pickle=False) as data:
            self.assertEqual(data["ema"].shape, (49, 12))
            self.assertEqual(data["ema"].dtype, np.float32)
            np.testing.assert_array_equal(data["pitch"], np.full((49, 1), 180.0))
            self.assertEqual(data["periodicity"].shape, (49, 1))
            self.assertEqual(data["loudness"].shape, (49, 1))
            np.testing.assert_allclose(data["time"], np.arange(49) / 50)
            np.testing.assert_array_equal(data["channels"], CHANNELS)
            self.assertEqual(data["ft_len"], 49)
            self.assertEqual(data["source_sample_rate"], 48000)
            self.assertEqual(data["duration_seconds"], 1.0)
            self.assertEqual(str(data["text"]), "テスト:音声です。")
        self.assertEqual(self.records()[0]["status"], "saved")

    def test_resume_skips_without_model_load_and_overwrite_recomputes(self):
        self.run_cli()
        self.load_model.reset_mock()
        self.coder.inverter.reset_mock()
        self.assertEqual(self.run_cli(), 0)
        self.load_model.assert_not_called()
        self.coder.inverter.assert_not_called()
        self.assertEqual(self.records()[0]["status"], "skipped")
        self.assertEqual(self.run_cli("--overwrite"), 0)
        self.coder.inverter.assert_called_once()

    def test_resume_reprocesses_corrupt_output_and_changed_input(self):
        self.run_cli()
        self.destination().write_bytes(b"PK\x03\x04truncated archive")
        self.assertEqual(self.run_cli(), 0)
        self.assertEqual(self.records()[0]["status"], "saved")
        sf.write(self.source, np.sin(np.arange(32000)), 16000)
        self.assertEqual(self.run_cli(), 0)
        self.assertEqual(self.records()[0]["status"], "saved")

    def test_failed_audio_is_recorded_and_other_utterances_continue(self):
        sf.write(self.source, np.zeros(48000), 48000)
        self.make_audio("basic5000", "BASIC5000_0002")
        self.assertEqual(self.run_cli(), 1)
        self.assertEqual([record["status"] for record in self.records()], ["failed", "saved"])
        self.assertIn("non-silent", self.records()[0]["error"])
        self.assertFalse(self.destination().exists())
        self.assertTrue(self.destination("BASIC5000_0002").is_file())
        self.coder.inverter.assert_called_once()

    def test_subset_selection_and_total_limit(self):
        self.make_audio("travel1000", "TRAVEL1000_0001")
        self.make_audio("travel1000", "TRAVEL1000_0002")
        self.assertEqual(self.run_cli("--subsets", "travel1000", "--limit", "1"), 0)
        self.assertFalse(self.destination().exists())
        self.assertTrue(self.destination("TRAVEL1000_0001", "travel1000").is_file())
        self.assertFalse(self.destination("TRAVEL1000_0002", "travel1000").exists())
        self.assertEqual(len(self.records()), 1)

    def test_bad_arguments_do_not_load_model(self):
        for arguments in (("--limit", "0"), ("--subsets", "missing"),
                          ("--subsets", "all", "basic5000"), ("--device", "invalid")):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit) as caught:
                self.run_cli(*arguments)
            self.assertEqual(caught.exception.code, 2)
        self.load_model.assert_not_called()

    def test_invalid_model_output_does_not_publish_partial_npz(self):
        self.coder.inverter.return_value["ema"][0, 0, 0] = np.nan
        self.assertEqual(self.run_cli(), 1)
        self.assertFalse(self.destination().exists())
        self.assertEqual(self.records()[0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
