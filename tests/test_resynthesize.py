import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from sparc.resynthesize import main


class ResynthesisCLITest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        gpu = patch("sparc.resynthesize.torch.cuda.is_available", return_value=True)
        gpu.start()
        self.addCleanup(gpu.stop)
        self.directory = Path(self.temp.name)
        self.input = self.directory / "input.wav"
        self.waveform = 0.2 * np.sin(2 * np.pi * 180 * np.arange(16000) / 16000)
        sf.write(self.input, self.waveform, 16000)

    def run_cli(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main([str(arg) for arg in args])

    @patch("sparc.resynthesize.load_model")
    def test_saves_audio_at_synthesis_rate_and_numeric_features(self, load):
        output = self.directory / "nested" / "output.wav"
        features = self.directory / "nested" / "features.npz"
        code = {"ema": np.zeros((25, 12)), "spk_emb": np.ones(64), "ft_len": np.int64(25)}
        coder = load.return_value
        coder.encode.return_value = code
        coder.decode.return_value = self.waveform.astype(np.float32)
        coder.output_sr = 8000
        self.run_cli(self.input, "-o", output, "--features", features)
        audio, sample_rate = sf.read(output)
        self.assertEqual(sample_rate, 8000)
        self.assertEqual(sf.info(output).subtype, "PCM_16")
        np.testing.assert_allclose(audio, self.waveform, atol=1 / 32768)
        with np.load(features, allow_pickle=False) as stored:
            np.testing.assert_array_equal(stored["ema"], code["ema"])
            np.testing.assert_array_equal(stored["spk_emb"], code["spk_emb"])
            self.assertEqual(stored["ft_len"], 25)

    @patch("sparc.resynthesize.load_model")
    def test_rejects_invalid_audio_before_downloading_models(self, load):
        missing = self.directory / "missing.wav"
        silent = self.directory / "silent.wav"
        short = self.directory / "short.wav"
        invalid = self.directory / "invalid.wav"
        sf.write(silent, np.zeros(16000), 16000)
        sf.write(short, self.waveform[:100], 16000)
        sf.write(invalid, np.full(16000, np.nan), 16000, subtype="FLOAT")
        for source in (missing, silent, short, invalid):
            with self.subTest(source=source), self.assertRaises(SystemExit) as caught:
                self.run_cli(source)
            self.assertEqual(caught.exception.code, 2)
        load.assert_not_called()

    @patch("sparc.resynthesize.load_model")
    def test_protects_input_and_existing_output(self, load):
        original = self.input.read_bytes()
        with self.assertRaises(SystemExit):
            self.run_cli(self.input, "-o", self.input, "--overwrite")
        self.assertEqual(self.input.read_bytes(), original)
        output = self.directory / "existing.wav"
        output.write_bytes(b"keep this file")
        with self.assertRaises(SystemExit):
            self.run_cli(self.input, "-o", output)
        self.assertEqual(output.read_bytes(), b"keep this file")
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
