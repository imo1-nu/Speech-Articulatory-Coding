import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sparc.sweep_jsut_codebooks import main, validate_case
from sparc.trajectory_vq import CHANNELS


class CodebookSweepTest(unittest.TestCase):
    def test_two_windows_two_codebooks_use_identical_data_and_resume_without_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features = root / "features"
            directory = features / "basic5000"
            directory.mkdir(parents=True)
            for index in range(6):
                time = np.arange(64) / 50
                ema = np.sin(time[:, None] * np.arange(1, 13) + index).astype(np.float32)
                np.savez_compressed(directory / f"BASIC5000_{index:04d}.npz", ema=ema, time=time,
                                    channels=np.asarray(CHANNELS), frame_rate=50)
            output = root / "codebooks"
            arguments = ["--features-dir", str(features), "--output-dir", str(output),
                         "--reference-root", str(root / "no_reference"), "--codebooks", "4", "8",
                         "--windows-ms", "20", "40", "--epochs", "1", "--patience", "1",
                         "--skip-audio", "--device", "cpu", "--threads", "1"]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(arguments), 0)
                result = json.loads((output / "summary.json").read_text())
                self.assertEqual(result["completed_cases"], 4)
                self.assertEqual(result["expected_cases"], 4)
                self.assertEqual({(r["window_ms"], r["codebook_size"]) for r in result["rows"]},
                                 {(20, 4), (20, 8), (40, 4), (40, 8)})
                self.assertEqual({r["nominal_token_id_bps"] for r in result["rows"]}, {100, 150})
                self.assertTrue(all(np.isfinite(row["test_mse"]) for row in result["rows"]))
                states, split_reference, normalization = {}, None, None
                for row in result["rows"]:
                    run = Path(row["run_dir"])
                    states[run] = (run / "best.pt").read_bytes()
                    splits = json.loads((run / "splits.json").read_text())
                    if split_reference is None:
                        split_reference = splits
                    self.assertEqual(splits, split_reference)
                    with np.load(run / "normalization.npz", allow_pickle=False) as data:
                        if normalization is None:
                            normalization = (data["mean"], data["std"])
                        np.testing.assert_array_equal(data["mean"], normalization[0])
                        np.testing.assert_array_equal(data["std"], normalization[1])
                    config, evaluation = validate_case(run, row["window_ms"], row["codebook_size"],
                                                       features, splits, 1, 1, 512)
                    self.assertEqual(config["token_rate"], 50)
                    self.assertEqual(config["latent_dim"], 16)
                    self.assertEqual(evaluation["checkpoint_sha256"], row["checkpoint_sha256"])
                self.assertTrue((output / "index.html").is_file())
                self.assertTrue((output / "summary.csv").is_file())
                self.assertTrue((output / "codebook_comparison.png").is_file())
                self.assertEqual(main(arguments), 0)
                for run, data in states.items():
                    self.assertEqual((run / "best.pt").read_bytes(), data)
                with self.assertRaises(SystemExit):
                    main([*arguments, "--epochs", "2"])


if __name__ == "__main__":
    unittest.main()
