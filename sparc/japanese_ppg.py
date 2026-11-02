"""Pinned Japanese PPG model and frontend adapted from taisa1/ppgs_ja.

Upstream: d2dac95fd6f40ba0f2708b4a3561c4912e2490d1 (MIT).
Copyright (c) 2024 Interactive Audio Lab. See third_party/ppgs_ja/LICENSE.
"""

import hashlib
import math
from pathlib import Path

from huggingface_hub import hf_hub_download
import librosa
import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F
import torchaudio


MODEL_ID = "taisa1/ppgs-ja"
MODEL_REVISION = "c77bee7a8110b12d1d4fc10989b457e185e2a3b3"
SOURCE_COMMIT = "d2dac95fd6f40ba0f2708b4a3561c4912e2490d1"
SAMPLE_RATE = 16000
HOP = 160
FRAME_RATE = SAMPLE_RATE / HOP
FFT_SIZE = 1024
MAX_FRAMES = 5000
PHONEMES = (
    "a", "i", "u", "e", "o", "N", "w", "y", "j", "my", "ky", "dy", "gy", "ny",
    "hy", "ry", "py", "p", "t", "k", "ts", "ch", "b", "d", "g", "z", "m", "n",
    "s", "sh", "h", "f", "r", "q", "<silent>",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_times(frames):
    # Reflect padding is 432 samples; the 1024-sample window center is at 512.
    return (np.arange(frames, dtype=np.float64) * HOP + HOP / 2) / SAMPLE_RATE


def load_audio(path):
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError(f"Expected nonempty finite audio: {path}")
    info = {"source_sample_rate": sample_rate, "source_channels": waveform.shape[1],
            "source_samples": len(waveform), "source_duration_seconds": len(waveform) / sample_rate}
    mono = torch.from_numpy(waveform.mean(axis=1)).unsqueeze(0)
    if sample_rate != SAMPLE_RATE:
        mono = torchaudio.functional.resample(mono, sample_rate, SAMPLE_RATE)
    if mono.shape[-1] <= (FFT_SIZE - HOP) // 2:
        raise ValueError(f"Audio is too short for the PPG reflect padding: {path}")
    if mono.shape[-1] // HOP > MAX_FRAMES:
        raise ValueError(f"Audio exceeds the PPG model's {MAX_FRAMES}-frame limit: {path}")
    if not torch.isfinite(mono).all():
        raise ValueError(f"Resampled audio is not finite: {path}")
    info["resampled_samples"] = mono.shape[-1]
    return mono.unsqueeze(0), info  # (batch=1, channels=1, samples)


class JapaneseLogMel(nn.Module):
    """Upstream magnitude/log-Mel definitions, with explicit rounding and FP32 matmul."""

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(FFT_SIZE), persistent=False)
        basis = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=FFT_SIZE, n_mels=80)
        self.register_buffer("mel_basis", torch.from_numpy(basis), persistent=False)

    def forward(self, audio):
        if audio.ndim != 3 or audio.shape[1] != 1:
            raise ValueError("Expected audio with shape (batch, 1, samples)")
        padding = (FFT_SIZE - HOP) // 2
        padded = F.pad(audio.float(), (padding, padding), mode="reflect")
        spectrum = torch.stft(padded.squeeze(1), FFT_SIZE, hop_length=HOP,
                              window=self.window, center=False, normalized=False,
                              onesided=True, return_complex=True)
        components = torch.view_as_real(spectrum)
        magnitude = torch.sqrt(components.square().sum(-1) + 1e-6).to(torch.float16)
        mel = self.mel_basis @ magnitude.float()
        return torch.log(torch.clamp(mel, min=1e-5)).to(torch.float16).float()


class PositionalEncoding(nn.Module):
    def __init__(self, channels=256, max_len=MAX_FRAMES):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        index = torch.arange(max_len).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, channels, 2) * (-math.log(10000.0) / channels))
        encoding = torch.zeros(max_len, 1, channels)
        encoding[:, 0, 0::2] = torch.sin(index * frequency)
        encoding[:, 0, 1::2] = torch.cos(index * frequency)
        self.register_buffer("encoding", encoding)

    def forward(self, values):
        if len(values) > len(self.encoding):
            raise ValueError("PPG sequence exceeds positional encoding limit")
        return self.dropout(values + self.encoding[:len(values)])


class JapanesePPGModel(nn.Module):
    """State-dict-compatible, noncausal 35-phone Transformer for whole utterances."""

    def __init__(self):
        super().__init__()
        self.position = PositionalEncoding()
        self.input_layer = nn.Conv1d(80, 256, kernel_size=5, padding="same")
        layer = nn.TransformerEncoderLayer(256, 2)
        self.model = nn.TransformerEncoder(layer, 5, enable_nested_tensor=False)
        self.output_layer = nn.Conv1d(256, len(PHONEMES), kernel_size=5, padding="same")

    def forward(self, features, lengths=None):
        if lengths is None:
            lengths = torch.full((len(features),), features.shape[-1], device=features.device, dtype=torch.long)
        mask = torch.arange(features.shape[-1], device=features.device)[None] < lengths[:, None]
        values = self.input_layer(features) * mask[:, None]
        values = self.model(self.position(values.permute(2, 0, 1)),
                            src_key_padding_mask=~mask).permute(1, 2, 0)
        return self.output_layer(values) * mask[:, None]


class JapanesePPGExtractor:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        path = hf_hub_download(MODEL_ID, "mel_ja.pt", revision=MODEL_REVISION)
        self.checkpoint_sha256 = sha256_file(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint["model"] if "model" in checkpoint else checkpoint
        if (tuple(state["output_layer.weight"].shape) != (len(PHONEMES), 256, 5)
                or tuple(state["input_layer.weight"].shape) != (256, 80, 5)
                or tuple(state["position.encoding"].shape) != (MAX_FRAMES, 1, 256)):
            raise ValueError("Japanese checkpoint architecture does not match the pinned 35-phone model")
        self.model = JapanesePPGModel()
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.eval().requires_grad_(False).to(self.device)
        self.frontend = JapaneseLogMel().to(self.device)

    def extract(self, audio):
        with torch.inference_mode():
            features = self.frontend(audio.to(self.device))
            posterior = self.model(features).softmax(dim=1)[0].transpose(0, 1).cpu().numpy()
        expected = audio.shape[-1] // HOP
        if (posterior.shape != (expected, len(PHONEMES)) or not np.isfinite(posterior).all()
                or np.any(posterior < 0) or np.any(posterior > 1)
                or not np.allclose(posterior.sum(axis=1), 1, atol=1e-5, rtol=0)):
            raise ValueError("The PPG model did not produce a finite normalized (T, 35) posterior")
        return posterior.astype(np.float32, copy=False)


def js_divergence(first, second):
    """Standard per-frame Jensen-Shannon divergence in nats, without phone remapping."""
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != len(PHONEMES):
        raise ValueError("Expected two aligned (T, 35) posterior arrays")
    for values in (first, second):
        if (not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)
                or not np.allclose(values.sum(axis=1), 1, atol=1e-5, rtol=0)):
            raise ValueError("Expected normalized finite probability distributions")
    average = (first + second) / 2
    log_average = np.log(np.maximum(average, 1e-12))
    result = ((first * (np.log(np.maximum(first, 1e-12)) - log_average)).sum(axis=1)
              + (second * (np.log(np.maximum(second, 1e-12)) - log_average)).sum(axis=1)) / 2
    return np.maximum(result, 0)


def compare_posteriors(first, first_time, second, second_time):
    length = min(len(first), len(second))
    if (not length or len(first_time) != len(first) or len(second_time) != len(second)
            or not np.allclose(first_time[:length], second_time[:length], atol=1e-12, rtol=0)):
        raise ValueError("PPGs must share the same absolute frame-time grid")
    return np.asarray(first_time[:length]), js_divergence(first[:length], second[:length])
