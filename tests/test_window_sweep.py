import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sparc.sweep_jsut_windows import main
from sparc.trajectory_vq import CHANNELS


class WindowSweepTest(unittest.TestCase):
    def test_train_evaluate_and_resume_fixed_splits_for_odd_and_even_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features = root / "features"
            source_dir = features / "basic5000"
            source_dir.mkdir(parents=True)
            for index in range(6):
                time = np.arange(64) / 50
                ema = np.sin(time[:, None] * np.arange(1, 13) + index).astype(np.float32)
                np.savez_compressed(source_dir / f"BASIC5000_{index:04d}.npz", ema=ema,
                                    time=time, channels=np.asarray(CHANNELS), frame_rate=50)
            output = root / "sweep"
            arguments = ["--features-dir", str(features), "--output-dir", str(output),
                         "--windows-ms", "20", "40", "180", "--epochs", "1", "--patience", "1",
                         "--skip-audio", "--device", "cpu", "--threads", "1"]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(arguments), 0)
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["completed_windows_ms"], [20, 40, 180])
                self.assertEqual(len(summary["rows"]), 6)
                splits = []
                checkpoints = []
                for ms in (20, 40, 180):
                    run = output / f"window_{ms:03d}ms"
                    config = json.loads((run / "config.json").read_text())
                    splits.append(json.loads((run / "splits.json").read_text()))
                    checkpoints.append((run / "best.pt").read_bytes())
                    self.assertEqual(config["window_frames"], ms // 20)
                    self.assertTrue(config["cache_windows"])
                    with np.load(run / "tokens" / splits[-1]["test"][0], allow_pickle=False) as data:
                        self.assertEqual(data["tokens"].shape, (64,))
                        self.assertEqual(data["weighted_reconstruction"].shape, (64, 12))
                        if ms <= 40:
                            np.testing.assert_array_equal(data["reconstruction"], data["weighted_reconstruction"])
                self.assertEqual(splits[0], splits[1])
                self.assertEqual(splits[1], splits[2])
                self.assertTrue((output / "window_comparison.png").is_file())
                self.assertTrue((output / "index.html").is_file())
                self.assertTrue(all(np.isfinite(row["test_mse"]) for row in summary["rows"]))
                self.assertEqual(main(arguments), 0)
                for index, ms in enumerate((20, 40, 180)):
                    self.assertEqual((output / f"window_{ms:03d}ms/best.pt").read_bytes(), checkpoints[index])
                with self.assertRaises(SystemExit):
                    main([*arguments, "--epochs", "2"])


if __name__ == "__main__":
    unittest.main()
