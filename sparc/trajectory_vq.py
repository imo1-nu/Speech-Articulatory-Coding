"""VQ-VAE and utterance-local windows for 50 Hz articulatory trajectories."""

import bisect

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


CHANNELS = ("TDX", "TDY", "TBX", "TBY", "TTX", "TTY",
            "LIX", "LIY", "ULX", "ULY", "LLX", "LLY")
WINDOW = 5


def window_padding(window_frames):
    if not isinstance(window_frames, int) or window_frames < 1:
        raise ValueError("window_frames must be a positive integer")
    left = (window_frames - 1) // 2
    return left, window_frames - 1 - left


def triangular_weights(window_frames):
    window_padding(window_frames)
    return np.minimum(np.arange(1, window_frames + 1), np.arange(window_frames, 0, -1)).astype(float)


def split_utterances(paths, seed=42):
    if len(paths) < 3:
        raise ValueError("At least three utterances are required for train/validation/test splits")
    order = np.random.default_rng(seed).permutation(len(paths))
    heldout = max(1, int(len(paths) * 0.05))
    groups = (order[:len(paths) - 2 * heldout],
              order[len(paths) - 2 * heldout:len(paths) - heldout], order[-heldout:])
    return {name: sorted(paths[i] for i in indices)
            for name, indices in zip(("train", "validation", "test"), groups)}


def normalization(utterances):
    """Compute population statistics without mixing validation/test frames."""
    count = sum(len(values) for values in utterances)
    mean = sum(values.sum(axis=0, dtype=np.float64) for values in utterances) / count
    variance = sum(((values.astype(np.float64) - mean) ** 2).sum(axis=0)
                   for values in utterances) / count
    std = np.sqrt(variance)
    return mean.astype(np.float32), np.where(std < 1e-6, 1.0, std).astype(np.float32)


class WindowDataset(Dataset):
    """Construct utterance-local windows at every ``stride`` original frames."""

    def __init__(self, utterances, mean, std, window_frames=WINDOW, stride=1):
        self.window_frames = window_frames
        if stride not in (1, 2) or window_frames < stride:
            raise ValueError("stride must be 1 or 2 and must not exceed window_frames")
        self.stride = stride
        padding = window_padding(window_frames)
        self.values = [torch.from_numpy(np.pad(
            ((values - mean) / std).astype(np.float32), (padding, (0, 0)), mode="edge"
        )) for values in utterances]
        self.ends = np.cumsum([(len(values) + stride - 1) // stride for values in utterances]).tolist()

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        utterance = bisect.bisect_right(self.ends, index)
        offset = index - (self.ends[utterance - 1] if utterance else 0)
        offset *= self.stride
        return self.values[utterance][offset:offset + self.window_frames].reshape(-1)


class TrajectoryVQVAE(nn.Module):
    def __init__(self, codebook_size=128, latent_dim=16, window_frames=WINDOW, token_stride=1):
        super().__init__()
        window_padding(window_frames)
        self.window_frames = window_frames
        if token_stride not in (1, 2) or window_frames < token_stride:
            raise ValueError("Token stride must be 1 or 2 and must not exceed the window")
        self.token_stride = token_stride
        self.encoder = nn.Sequential(nn.Linear(window_frames * 12, 128), nn.ReLU(),
                                     nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, latent_dim))
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(),
                                     nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, window_frames * 12))

    def quantize(self, latent):
        weights = self.codebook.weight
        distances = (latent.square().sum(-1, keepdim=True)
                     + weights.square().sum(-1)[None] - 2 * latent @ weights.T)
        tokens = distances.argmin(-1)
        return self.codebook(tokens), tokens

    def forward(self, windows):
        latent = self.encoder(windows)
        quantized, tokens = self.quantize(latent)
        straight_through = latent + (quantized - latent).detach()
        reconstruction = self.decoder(straight_through)
        return {
            "reconstruction": reconstruction,
            "tokens": tokens,
            "reconstruction_loss": F.mse_loss(reconstruction, windows),
            "codebook_loss": F.mse_loss(quantized, latent.detach()),
            "commitment_loss": F.mse_loss(latent, quantized.detach()),
        }


def overlap_average(windows, weights=None, stride=1, output_length=None):
    """Merge windows using per-offset weights, excluding padded positions.

    Weights follow each decoded window from its first to last frame.
    The default preserves the original uniform reconstruction.
    """
    windows = np.asarray(windows)
    if windows.ndim == 2:
        windows = windows.reshape(len(windows), -1, 12)
    if windows.ndim != 3 or windows.shape[2] != 12 or not len(windows):
        raise ValueError("Expected nonempty (T, window_frames, 12) decoded windows")
    window_frames = windows.shape[1]
    left, _ = window_padding(window_frames)
    weights = np.ones(window_frames) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (window_frames,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Overlap weights must be finite positive values, one per window frame")
    length = len(windows) * stride if output_length is None else output_length
    if (stride not in (1, 2) or window_frames < stride or length < 1
            or len(windows) != (length + stride - 1) // stride):
        raise ValueError("Decoded window count does not match stride/output_length")
    total = np.zeros((length, 12), dtype=np.float64)
    counts = np.zeros((length, 1), dtype=np.float64)
    centers = np.arange(len(windows)) * stride
    for index, offset in enumerate(range(-left, window_frames - left)):
        positions = centers + offset
        valid = (positions >= 0) & (positions < length)
        total[positions[valid]] += windows[valid, index] * weights[index]
        counts[positions[valid]] += weights[index]
    if np.any(counts == 0):
        raise ValueError("Decoded windows leave uncovered output frames")
    return (total / counts).astype(np.float32)


def usage_statistics(counts):
    counts = np.asarray(counts, dtype=np.int64)
    probabilities = counts[counts > 0] / counts.sum()
    return {"used_codes": int((counts > 0).sum()), "total_codes": len(counts),
            "usage_fraction": float((counts > 0).mean()),
            "code_perplexity": float(np.exp(-(probabilities * np.log(probabilities)).sum()))}


def fit_markov(sequences, codebook_size, alpha=1.0):
    """Count transitions within each utterance, including self transitions."""
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    counts = np.zeros((codebook_size, codebook_size), dtype=np.int64)
    frequencies = np.zeros(codebook_size, dtype=np.int64)
    for sequence in sequences:
        sequence = np.asarray(sequence)
        if (sequence.ndim != 1 or not np.issubdtype(sequence.dtype, np.integer)
                or not len(sequence) or sequence.min() < 0 or sequence.max() >= codebook_size):
            raise ValueError("Tokens must be a nonempty integer vector within the codebook")
        frequencies += np.bincount(sequence, minlength=codebook_size)
        np.add.at(counts, (sequence[:-1], sequence[1:]), 1)
    totals = counts.sum(axis=1, keepdims=True)
    empirical = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)
    return {"transition_counts": counts, "transition_probs_empirical": empirical,
            "transition_probs": (counts + alpha) / (totals + alpha * codebook_size),
            "token_counts": frequencies,
            "unigram_probs": (frequencies + alpha) / (frequencies.sum() + alpha * codebook_size)}


def score_markov(sequences, model):
    log_likelihood = 0.0
    baseline = 0.0
    count = 0
    for sequence in sequences:
        count += max(0, len(sequence) - 1)
        log_likelihood += np.log(model["transition_probs"][sequence[:-1], sequence[1:]]).sum()
        baseline += np.log(model["unigram_probs"][sequence[1:]]).sum()
    return {"transitions": count, "log_likelihood": float(log_likelihood),
            "nll_per_transition": float(-log_likelihood / count) if count else None,
            "perplexity": float(np.exp(-log_likelihood / count)) if count else None,
            "unigram_nll_per_transition": float(-baseline / count) if count else None,
            "unigram_perplexity": float(np.exp(-baseline / count)) if count else None}
