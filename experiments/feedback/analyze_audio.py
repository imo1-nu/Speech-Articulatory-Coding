"""Extract and plot F0/F1/F2/F3 from audio without acoustic quality filters."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import parselmouth
import soundfile as sf
import yaml

from .acoustics import FEATURE_NAMES, feature_arrays
from .raw_acoustics import extract_raw_acoustics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main(argv=None):
    begun = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--resynthesis", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/feedback_experiment1.yaml"))
    parser.add_argument("--source-metadata", type=Path)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float)
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty; select a new --output")
    parameters = yaml.safe_load(args.config.read_text())["formants"]
    keys = ["time_step_s", "max_number_of_formants", "ceiling_hz", "window_length_s",
            "pre_emphasis_hz", "pitch_floor_hz", "pitch_ceiling_hz", "voicing_threshold"]
    settings = {k: parameters[k] for k in keys}
    inputs = {"original": args.input}
    if args.resynthesis:
        inputs["resynthesis"] = args.resynthesis
    signals, input_metadata = {}, {}
    for name, path in inputs.items():
        signal, sr = sf.read(path, always_2d=True)
        signals[name] = (signal.mean(axis=1), sr)
        input_metadata[name] = {"path": str(path), "sha256": digest(path), "sample_rate": sr,
            "channels": signal.shape[1], "frames": len(signal), "duration_s": len(signal)/sr}
    end = args.end if args.end is not None else min(m["duration_s"] for m in input_metadata.values())
    if not 0 <= args.start < end <= min(m["duration_s"] for m in input_metadata.values()):
        parser.error("Analysis interval must fit all input waveforms")
    args.output.mkdir(parents=True, exist_ok=True)
    measured, timings = {}, {}
    for name, (signal, sr) in signals.items():
        start_time = perf_counter()
        data = extract_raw_acoustics(signal, sr, args.start, end, settings)
        timings[name + "_extraction_s"] = perf_counter() - start_time
        measured[name] = feature_arrays(data)
        np.savez_compressed(args.output / f"{name}_acoustics.npz", **measured[name], bandwidth_hz=data["bandwidth_hz"])
        with (args.output / f"{name}_acoustics.csv").open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["time_s", "F0_hz", "F1_hz", "F2_hz", "F3_hz"])
            writer.writerows(np.column_stack([data["time_s"], measured[name]["raw_hz"]]))
        input_metadata[name]["summary"] = data["summary"]
        shutil.copyfile(inputs[name], args.output / f"{name}.wav")
        print(f"{name}: {len(data['time_s'])} frames, extraction {timings[name + '_extraction_s']:.3f} s", flush=True)
    start_time = perf_counter()
    fig, axes = plt.subplots(4, len(measured), figsize=(7 * len(measured), 11),
                             squeeze=False, sharex=True, sharey="row", constrained_layout=True)
    for col, (name, arrays) in enumerate(measured.items()):
        for j, feature in enumerate(FEATURE_NAMES):
            axes[j, col].plot(arrays["time_s"], arrays["raw_hz"][:, j], "o-", color="tab:blue", markersize=2, linewidth=.8)
            axes[j, col].set(title=f"{name}: {feature} (all estimates)", xlabel="WAV time (s)", ylabel="Hz")
    fig.savefig(args.output / "acoustics_raw.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True, constrained_layout=True)
    for name, arrays in measured.items():
        for j, feature in enumerate(FEATURE_NAMES):
            axes[j].plot(arrays["time_s"], arrays["raw_hz"][:, j], linewidth=.8, label=name)
            axes[j].set(ylabel=f"{feature} (Hz)")
    for ax in axes:
        ax.legend()
    axes[-1].set_xlabel("WAV time (s)")
    fig.savefig(args.output / "acoustics_overlay.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(len(signals), 1, figsize=(14, 2.5*len(signals)), squeeze=False,
                             sharex=True, constrained_layout=True)
    for ax, (name, (signal, sr)) in zip(axes[:, 0], signals.items()):
        ax.plot(np.arange(len(signal))/sr, signal, linewidth=.5)
        ax.set(title=name, xlabel="WAV time (s)", ylabel="Amplitude")
    fig.savefig(args.output / "waveforms.png", dpi=160)
    plt.close(fig)
    timings["plotting_s"] = perf_counter() - start_time
    comparison = []
    if "resynthesis" in measured:
        for j, feature in enumerate(FEATURE_NAMES):
            a, b = measured["original"]["raw_hz"][:, j], measured["resynthesis"]["raw_hz"][:, j]
            shared = np.isfinite(a) & np.isfinite(b)
            differences = b[shared] - a[shared]
            comparison.append({"feature": feature, "paired_available_frames": int(shared.sum()),
                "mean_difference_hz": float(np.mean(differences)) if len(differences) else None,
                "mae_hz": float(np.mean(np.abs(differences))) if len(differences) else None,
                "rmse_hz": float(np.sqrt(np.mean(differences**2))) if len(differences) else None})
        with (args.output / "comparison.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(comparison[0]))
            writer.writeheader(); writer.writerows(comparison)
    if args.source_metadata:
        shutil.copyfile(args.source_metadata, args.output / "source_provenance.json")
    timings["total_wall_s"] = perf_counter() - begun
    summary = {"feature_names": FEATURE_NAMES, "quality_filtering": False, "interval_s": [args.start, end],
               "inputs": input_metadata, "settings": settings, "comparison": comparison, "timing": timings,
               "comparison_note": "Shared WAV times, no delay alignment; finite pairs only; formants include unvoiced frames",
               "source_sha256": {str(p): digest(p) for p in [Path(__file__), Path(__file__).with_name("raw_acoustics.py")]},
               "environment": {"python": platform.python_version(), "parselmouth": parselmouth.__version__,
                   "praat": parselmouth.PRAAT_VERSION, "packages": {n: importlib.metadata.version(n) for n in
                       ["numpy", "matplotlib", "soundfile", "praat-parselmouth", "PyYAML"]}}}
    save_json(args.output / "summary.json", summary)
    print(f"Saved: {args.output}; total {timings['total_wall_s']:.3f} s", flush=True)


if __name__ == "__main__":
    main()
