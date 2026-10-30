"""Saved figures for input, baseline EMA, and formant quality."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from .adapter import EMA_NAMES


def plot_result(destination, original, original_sr, synthesized, output_sr,
                code, feature_rate, start, end, measurements):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    for col, (label, wav, sr) in enumerate([
        ("Original", original, original_sr), ("SPARC baseline", synthesized, output_sr)
    ]):
        t = np.arange(len(wav)) / sr
        axes[0, col].plot(t, wav, linewidth=.5)
        axes[0, col].axvspan(start, end, color="orange", alpha=.2)
        axes[0, col].set(title=label, xlabel="WAV time (s)", ylabel="Amplitude")
        frequencies, spec_times, power = spectrogram(
            wav, fs=sr, nperseg=int(.025 * sr), noverlap=int(.020 * sr))
        axes[1, col].pcolormesh(spec_times, frequencies,
                               10 * np.log10(np.maximum(power, 1e-12)),
                               cmap="magma", vmin=-100, vmax=-30, shading="auto")
        axes[1, col].set(ylim=(0, 5000), xlabel="WAV time (s)", ylabel="Frequency (Hz)")
        axes[1, col].axvspan(start, end, color="cyan", alpha=.12)
    fig.savefig(destination / "audio.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(6, 2, figsize=(13, 12), constrained_layout=True)
    times = np.arange(len(code["ema"])) / feature_rate
    for j, ax in enumerate(axes.flat):
        ax.plot(times, code["ema"][:, j], linewidth=1)
        ax.axvspan(start, end, color="orange", alpha=.2)
        ax.set(title=EMA_NAMES[j], xlabel="Nominal feature time (s)", ylabel="SPARC EMA units")
    fig.savefig(destination / "ema.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    for label, data in measurements.items():
        axes[0].plot(data["time_s"], data["f0_filtered_hz"], "o-", markersize=3, label=label)
        median = data["summary"]["f0"]["target_hz"]
        if median is not None:
            axes[0].axhline(median, linestyle="--", alpha=.5)
    axes[0].set(title="F0 measured from audio (Praat autocorrelation)", ylabel="F0 (Hz)")
    axes[0].legend()
    axes[1].plot(np.arange(len(code["pitch"])) / feature_rate, code["pitch"][:, 0],
                 label="SPARC pitch input (CREPE)")
    axes[1].axvspan(start, end, color="orange", alpha=.2)
    axes[1].set(xlim=(start, end), title="Fixed decoder pitch condition (nominal time)",
                xlabel="WAV / nominal feature time (s)", ylabel="Pitch input (Hz)")
    axes[1].legend()
    fig.savefig(destination / "f0.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True)
    for col, (label, data) in enumerate(measurements.items()):
        for j in range(3):
            ax = axes[j, col]
            ax.plot(data["time_s"], data["raw_hz"][:, j], "o-", color="0.7", markersize=3)
            ax.plot(data["time_s"], data["filtered_hz"][:, j], "o-", markersize=3)
            median = data["summary"]["formants"][j]["target_hz"]
            if median is not None:
                ax.axhline(median, linestyle="--", color="green")
            ax.set(title=f"{label} F{j + 1}: {data['summary']['formants'][j]['status']}",
                   xlabel="WAV time (s)", ylabel="Hz")
    fig.savefig(destination / "formants.png", dpi=160)
    plt.close(fig)
