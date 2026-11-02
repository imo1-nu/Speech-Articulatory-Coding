import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from sparc.model_jsut_tokens import CachedWindowBatches, infer_utterance, load_run, main
from sparc.trajectory_vq import (
    CHANNELS, TrajectoryVQVAE, WindowDataset, fit_markov, normalization,
    overlap_average, score_markov, split_utterances, triangular_weights, window_padding,
)


class TrajectoryMathTest(unittest.TestCase):
    def test_25hz_windows_cover_odd_even_and_single_frame_utterances_without_interpolation(self):
        for length in (1, 2, 3, 4, 11, 12):
            with self.subTest(length=length):
                first = np.repeat(np.arange(length, dtype=np.float32)[:, None], 12, axis=1)
                second = np.full((3, 12), 100, dtype=np.float32)
                dataset = WindowDataset([first, second], np.zeros(12), np.ones(12), 2, stride=2)
                count = (length + 1) // 2
                self.assertEqual(len(dataset), count + 2)
                windows = np.stack([dataset[i].numpy() for i in range(count)]).reshape(count, 2, 12)
                np.testing.assert_array_equal(windows[:, 0], first[::2])
                np.testing.assert_array_equal(overlap_average(windows, stride=2, output_length=length), first)
                if length % 2:
                    np.testing.assert_array_equal(windows[-1, 1], first[-1])
                np.testing.assert_array_equal(dataset[count].reshape(2, 12), np.full((2, 12), 100))
                cached = torch.cat(list(CachedWindowBatches(dataset, 4, torch.device("cpu"))))
                torch.testing.assert_close(cached, torch.stack([dataset[i] for i in range(len(dataset))]))
        with self.assertRaises(ValueError):
            WindowDataset([first], np.zeros(12), np.ones(12), 1, stride=2)

    def test_all_nine_windows_preserve_alignment_and_cached_batch_order(self):
        first = np.repeat(np.arange(11, dtype=np.float32)[:, None], 12, axis=1)
        second = np.repeat((100 + np.arange(3, dtype=np.float32))[:, None], 12, axis=1)
        for width in range(1, 10):
            with self.subTest(window_frames=width):
                dataset = WindowDataset([first, second], np.zeros(12), np.ones(12), width)
                windows = np.stack([dataset[i].numpy() for i in range(len(first))]).reshape(-1, width, 12)
                left, right = window_padding(width)
                self.assertEqual(left + right, width - 1)
                np.testing.assert_array_equal(windows[:, left], first)
                np.testing.assert_array_equal(overlap_average(windows), first)
                np.testing.assert_array_equal(overlap_average(windows, triangular_weights(width)), first)
                cached = torch.cat(list(CachedWindowBatches(dataset, 4, torch.device("cpu"))))
                expected = torch.stack([dataset[i] for i in range(len(dataset))])
                torch.testing.assert_close(cached, expected)
                model = TrajectoryVQVAE(4, 4, width)
                result = model(cached)
                self.assertEqual(tuple(result["reconstruction"].shape), (len(dataset), width * 12))
                self.assertEqual(tuple(result["tokens"].shape), (len(dataset),))

    def test_weighted_overlap_emphasizes_center_and_normalizes_utterance_edges(self):
        windows = np.zeros((5, 5, 12), dtype=np.float32)
        windows[2] = 9
        weights = [1, 2, 3, 2, 1]
        result = overlap_average(windows, weights)
        np.testing.assert_allclose(result[:, 0], [1.5, 2.25, 3, 2.25, 1.5])
        self.assertAlmostEqual(float(overlap_average(windows)[2, 0]), 1.8, places=6)
        np.testing.assert_allclose(result, overlap_average(windows, np.asarray(weights) * 10))
        singleton = np.repeat(np.arange(1, 6)[None, :, None], 12, axis=2)
        np.testing.assert_array_equal(overlap_average(singleton, weights), np.full((1, 12), 3))
        for invalid in ([1, 2], [0, 1, 1, 1, 1], [1, -1, 1, 1, 1], [1, 1, np.nan, 1, 1]):
            with self.subTest(weights=invalid), self.assertRaises(ValueError):
                overlap_average(windows, invalid)

    def test_centered_windows_replicate_edges_without_crossing_utterances(self):
        first = np.repeat(np.arange(3, dtype=np.float32)[:, None], 12, axis=1)
        second = np.full((1, 12), 99, dtype=np.float32)
        dataset = WindowDataset([first, second], np.zeros(12), np.ones(12))
        np.testing.assert_array_equal(dataset[0].reshape(5, 12)[:, 0], [0, 0, 0, 1, 2])
        np.testing.assert_array_equal(dataset[2].reshape(5, 12)[:, 0], [0, 1, 2, 2, 2])
        np.testing.assert_array_equal(dataset[3].reshape(5, 12), np.full((5, 12), 99))
        original_windows = np.stack([dataset[i].numpy() for i in range(3)])
        np.testing.assert_array_equal(overlap_average(original_windows), first)

    def test_split_is_deterministic_disjoint_and_normalization_uses_training_only(self):
        paths = [str(i) for i in range(100)]
        groups = split_utterances(paths)
        self.assertEqual(groups, split_utterances(paths))
        self.assertEqual([len(value) for value in groups.values()], [90, 5, 5])
        self.assertEqual(set().union(*map(set, groups.values())), set(paths))
        self.assertFalse(set(groups["train"]) & set(groups["validation"]))
        training = [np.full((2, 12), int(path), dtype=np.float32) for path in groups["train"]]
        mean, std = normalization(training)
        np.testing.assert_allclose(mean, np.mean([int(path) for path in groups["train"]]))
        np.testing.assert_allclose(std, np.std([int(path) for path in groups["train"]]))
        _, constant_std = normalization([np.ones((2, 12))])
        np.testing.assert_array_equal(constant_std, np.ones(12))

    def test_markov_counts_self_transitions_and_never_joins_utterances(self):
        model = fit_markov([np.array([0, 0, 1]), np.array([2, 1])], 4, alpha=1.0)
        expected = np.zeros((4, 4), dtype=np.int64)
        expected[0, 0] = expected[0, 1] = expected[2, 1] = 1
        np.testing.assert_array_equal(model["transition_counts"], expected)
        np.testing.assert_allclose(model["transition_probs"][0], [2 / 6, 2 / 6, 1 / 6, 1 / 6])
        np.testing.assert_array_equal(model["transition_probs_empirical"][3], np.zeros(4))
        np.testing.assert_allclose(model["transition_probs"][3], np.full(4, 0.25))
        np.testing.assert_allclose(model["transition_probs"].sum(axis=1), 1)
        scores = score_markov([np.array([0, 0, 1]), np.array([2, 1])], model)
        expected_log = 2 * np.log(2 / 6) + np.log(2 / 5)
        self.assertEqual(scores["transitions"], 3)
        self.assertAlmostEqual(scores["log_likelihood"], expected_log)
        self.assertAlmostEqual(scores["perplexity"], np.exp(-expected_log / 3))
        self.assertEqual(score_markov([np.array([0])], model)["transitions"], 0)

    def test_vq_learns_and_checkpoint_restores_identical_tokens(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = TrajectoryVQVAE(codebook_size=4, latent_dim=4)
        targets = torch.stack([torch.ones(60), -torch.ones(60)] * 8)
        with torch.no_grad():
            model.codebook.weight[:2].copy_(model.encoder(targets[:2]))
        initial = model(targets)["reconstruction_loss"].item()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        for _ in range(80):
            result = model(targets)
            loss = result["reconstruction_loss"] + result["codebook_loss"] + 0.25 * result["commitment_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertLess(model(targets)["reconstruction_loss"].item(), initial * 0.5)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = TrajectoryVQVAE(4, 4)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        torch.testing.assert_close(model(targets)["tokens"], restored(targets)["tokens"])


class TokenPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.features = self.root / "features"
        self.run = self.root / "run"
        directory = self.features / "basic5000"
        directory.mkdir(parents=True)
        for index in range(8):
            time = np.arange(40 + index) / 50
            ema = np.sin(time[:, None] * np.arange(1, 13)[None] * 3 + index).astype(np.float32)
            np.savez_compressed(directory / f"BASIC5000_{index:04d}.npz", ema=ema, time=time,
                                channels=np.asarray(CHANNELS), frame_rate=50,
                                pitch=np.full((len(time), 1), 180.0),
                                periodicity=np.full((len(time), 1), 0.8), text=np.str_("テスト"))

    def cli(self, command, *arguments):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main([command, "--run-dir", str(self.run), "--threads", "1", *map(str, arguments)])

    def train(self):
        return self.cli("train", "--features-dir", self.features,
                        "--codebook-size", "4", "--latent-dim", "4", "--epochs", "3",
                        "--batch-size", "64", "--device", "cpu")

    def test_full_pipeline_preserves_alignment_split_and_auxiliary_features(self):
        self.assertEqual(self.train(), 0)
        self.assertEqual(self.cli("tokenize", "--device", "cpu"), 0)
        self.assertEqual(self.cli("transitions"), 0)
        config, splits, model, mean, std = load_run(self.run, torch.device("cpu"))
        self.assertEqual(sum(map(len, splits.values())), 8)
        training = []
        for path in splits["train"]:
            with np.load(self.features / path, allow_pickle=False) as data:
                training.append(data["ema"])
        expected_mean, expected_std = normalization(training)
        np.testing.assert_allclose(mean, expected_mean)
        np.testing.assert_allclose(std, expected_std)
        train_frames = 0
        for split, relatives in splits.items():
            for relative in relatives:
                with np.load(self.features / relative, allow_pickle=False) as original:
                    with np.load(self.run / "tokens" / relative, allow_pickle=False) as stored:
                        self.assertEqual(stored["tokens"].shape, (len(original["ema"]),))
                        self.assertEqual(str(stored["split"]), split)
                        self.assertEqual(str(stored["text"]), "テスト")
                        self.assertTrue(np.isfinite(stored["reconstruction"]).all())
                        np.testing.assert_array_equal(stored["time"], original["time"])
                        np.testing.assert_array_equal(stored["pitch"], original["pitch"])
                        inferred, _, _ = infer_utterance(model, original["ema"], mean, std, 512,
                                                         torch.device("cpu"))
                        np.testing.assert_array_equal(stored["tokens"], inferred)
                        if split == "train":
                            train_frames += len(inferred)
        with np.load(self.run / "transitions.npz", allow_pickle=False) as probabilities:
            self.assertEqual(probabilities["transition_counts"].sum(), train_frames - len(splits["train"]))
            np.testing.assert_allclose(probabilities["transition_probs"].sum(axis=1), 1)
        metrics = json.loads((self.run / "transition_metrics.json").read_text())
        self.assertTrue(np.isfinite(metrics["splits"]["test"]["perplexity"]))
        self.assertTrue((self.run / "plots/reconstruction_test.png").is_file())
        self.assertTrue((self.run / "plots/representative_trajectories.png").is_file())
        self.assertTrue((self.run / "plots/transitions_and_usage.png").is_file())
        with self.assertRaises(SystemExit):
            self.train()

    def test_changed_features_are_rejected_before_tokenization(self):
        self.train()
        config = json.loads((self.run / "config.json").read_text())
        source = self.features / next(iter(config["files"]))
        source.write_bytes(b"changed input")
        with self.assertRaises(SystemExit):
            self.cli("tokenize", "--device", "cpu")

    def test_25hz_training_tokenization_and_transitions_keep_50hz_reconstruction(self):
        self.assertEqual(self.cli("train", "--features-dir", self.features, "--window-ms", "40",
                                  "--token-rate", "25", "--cache-windows", "--codebook-size", "4",
                                  "--latent-dim", "4", "--epochs", "2", "--device", "cpu"), 0)
        self.assertEqual(self.cli("tokenize", "--device", "cpu"), 0)
        self.assertEqual(self.cli("transitions"), 0)
        config, splits, model, mean, std = load_run(self.run, torch.device("cpu"))
        self.assertEqual(config["token_rate"], 25)
        self.assertEqual(model.token_stride, 2)
        train_tokens = 0
        for split, relatives in splits.items():
            for relative in relatives:
                with np.load(self.features / relative, allow_pickle=False) as original:
                    length = len(original["ema"])
                    with np.load(self.run / "tokens" / relative, allow_pickle=False) as data:
                        self.assertEqual(data["tokens"].shape, ((length + 1) // 2,))
                        self.assertEqual(data["reconstruction"].shape, (length, 12))
                        self.assertEqual(data["ft_len"], length)
                        self.assertEqual(data["token_len"], (length + 1) // 2)
                        self.assertEqual(data["frame_rate"], 50)
                        self.assertEqual(data["token_rate"], 25)
                        np.testing.assert_array_equal(data["time"], original["time"])
                        np.testing.assert_array_equal(data["token_time"], original["time"][::2])
                        inferred, reconstructed, _ = infer_utterance(model, original["ema"], mean, std,
                                                                     512, torch.device("cpu"))
                        np.testing.assert_array_equal(inferred, data["tokens"])
                        np.testing.assert_allclose(reconstructed, data["reconstruction"])
                        if split == "train":
                            train_tokens += len(inferred)
        with np.load(self.run / "transitions.npz", allow_pickle=False) as probabilities:
            self.assertEqual(probabilities["transition_counts"].sum(), train_tokens - len(splits["train"]))

    def test_stale_checkpoint_tokens_are_rejected(self):
        self.train()
        self.cli("tokenize", "--device", "cpu")
        path = next((self.run / "tokens").rglob("*.npz"))
        with np.load(path, allow_pickle=False) as stored:
            arrays = dict(stored)
        arrays["checkpoint_sha256"] = np.str_("wrong model")
        np.savez_compressed(path, **arrays)
        with self.assertRaises(SystemExit):
            self.cli("transitions")
        self.assertFalse((self.run / "transitions.npz").exists())


if __name__ == "__main__":
    unittest.main()
