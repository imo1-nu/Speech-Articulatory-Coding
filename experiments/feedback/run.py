"""Experiment 1: encode labelled vowels, resynthesize, measure F0/F1/F2/F3."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
from time import perf_counter

import numpy as np
import parselmouth
import soundfile as sf
import torch
import yaml

from .adapter import EMA_NAMES, encode_decode, synchronize
from .acoustics import FEATURE_NAMES, compare_acoustics, feature_arrays
from .formants import FormantExtractionError, extract_formants
from .visualization import plot_result


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def save_measurements(directory, name, data):
    np.savez_compressed(directory / f"{name}_formants.npz",
                        **{key: data[key] for key in [
                            "time_s", "f0_hz", "voiced", "f0_valid", "f0_filtered_hz", "raw_hz", "bandwidth_hz",
                            "valid", "filtered_hz"]})
    with (directory / f"{name}_formants.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["time_s", "f0_hz", "voiced"] + [
            f"F{j}_{field}" for j in (1, 2, 3)
            for field in ("raw_hz", "bandwidth_hz", "accepted_hz", "valid", "rejection")
        ])
        for i, t in enumerate(data["time_s"]):
            row = [t, data["f0_hz"][i], int(data["voiced"][i])]
            for j in range(3):
                row.extend([data["raw_hz"][i, j], data["bandwidth_hz"][i, j],
                            data["filtered_hz"][i, j], int(data["valid"][i, j]),
                            data["reasons"][i][j]])
            writer.writerow(row)
    # Explicit four-dimensional features; retain existing formant-only arrays.
    arrays = feature_arrays(data)
    np.savez_compressed(directory / f"{name}_acoustics.npz", **arrays)
    with (directory / f"{name}_acoustics.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["time_s"] + [f"{name}_{field}" for name in FEATURE_NAMES
                                       for field in ("raw_hz", "accepted_hz", "valid", "rejection")])
        for i, t in enumerate(arrays["time_s"]):
            row = [t]
            reasons = ["" if data["f0_valid"][i] else "unvoiced_or_missing", *data["reasons"][i]]
            for j in range(4):
                row.extend([arrays["raw_hz"][i, j], arrays["filtered_hz"][i, j],
                            int(arrays["valid"][i, j]), reasons[j]])
            writer.writerow(row)


def measure_and_save(directory, name, waveform, sample_rate, start, end, settings):
    try:
        data = extract_formants(waveform, sample_rate, start, end, settings)
    except FormantExtractionError as exc:
        write_json(directory / f"{name}_formants_error.json", {
            "status": "extraction_failed", "error": str(exc), "interval_s": [start, end],
        })
        raise
    save_measurements(directory, name, data)
    return data


def environment(device):
    import os
    cache = Path(os.environ["HF_HOME"]) / "hub"
    snapshots = {}
    for ref in sorted(cache.glob("models--*/refs/main")):
        snapshots[ref.parent.parent.name] = ref.read_text().strip()
    # WavLM safetensors may be cached under a different revision from config.json.
    assets = [
        {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)}
        for pattern in ("models--*/snapshots/*/*.ckpt", "models--*/snapshots/*/*.safetensors",
                        "models--*/snapshots/*/*.yaml", "models--*/snapshots/*/config.json")
        for p in sorted(cache.glob(pattern))
    ]
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "device": device, "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "praat_version": parselmouth.PRAAT_VERSION,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "model_refs": snapshots, "model_assets": assets,
        "source_sha256": {str(p): sha256(p) for p in
                          sorted([*Path("sparc").glob("*.py"), *Path("experiments/feedback").glob("*.py")])},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/feedback_experiment1.yaml"))
    parser.add_argument("--vowel", help="Run only this configured label (e.g. a)")
    parser.add_argument("--input", type=Path, help="Override inputs with a user WAV")
    parser.add_argument("--label", help="User-supplied phoneme label for --input")
    parser.add_argument("--start", type=float, help="Interval start in WAV seconds")
    parser.add_argument("--end", type=float, help="Interval end in WAV seconds")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda:0"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if args.input:
        if args.start is None or args.end is None or not args.label or args.vowel:
            parser.error("--input requires --label, --start, --end; omit --vowel")
        config["inputs"] = [{"label": args.label, "wav": str(args.input),
                             "start_s": args.start, "end_s": args.end}]
        config["manifest"] = None
    elif args.start is not None or args.end is not None or args.label:
        parser.error("--start, --end, --label require --input; otherwise edit the YAML")
    if args.vowel:
        config["inputs"] = [item for item in config["inputs"] if item["label"] == args.vowel]
    if not config["inputs"]:
        parser.error("No matching inputs")
    device = args.device or config["device"]
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("Requested CUDA is unavailable")
    config["device"] = device
    output = args.output or Path(config["output"])
    config["output"] = str(output)
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}; select a new --output")
    manifest = json.loads(Path(config["manifest"]).read_text()) if config.get("manifest") else None
    provenance = {}
    seen = set()
    for item in config["inputs"]:
        label = item["label"]
        if label in seen or Path(label).name != label or label in (".", ".."):
            parser.error("Input labels must be unique directory names")
        seen.add(label)
        path = Path(item["wav"])
        wav, sr = sf.read(path, always_2d=True)
        if len(wav) / sr < .5 or not np.isfinite(wav).all() or wav.std() < 1e-8:
            parser.error(f"Expected finite, non-silent input >= 0.5 seconds: {path}")
        # Require context for Praat's 50 ms actual window on both waveforms.
        if not .03 <= item["start_s"] < item["end_s"] <= len(wav) / sr - .04:
            parser.error(f"Interval needs valid bounds and edge context: {path}")
        digest = sha256(path)
        record = None
        if manifest:
            records = [c for c in manifest["clips"] if c["output_wav"] == str(path)]
            if len(records) != 1 or records[0]["output_sha256"] != digest:
                parser.error(f"WAV does not match the provenance manifest: {path}")
            record = records[0]["db_record"]
            if record["phoneme"] != label:
                parser.error(f"Phoneme label differs from manifest: {path}")
        provenance[label] = {"path": str(path), "sha256": digest, "database_record": record,
                             "label_source": "local manifest" if record else "user supplied"}
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["torch_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    from sparc import load_model
    print(f"Loading SPARC {config['model']} on {device}", flush=True)
    synchronize(device)
    begin = perf_counter()
    coder = load_model(config["model"], device=device, use_penn=False)
    synchronize(device)
    load_s = perf_counter() - begin
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(output / "environment.json", environment(device))
    write_json(output / "provenance.json", {
        "inputs": provenance,
        "manifest_sha256": sha256(config["manifest"]) if manifest else None,
        "config_source_sha256": sha256(args.config),
    })
    summaries = []
    for item in config["inputs"]:
        label, path = item["label"], Path(item["wav"])
        start, end = item["start_s"], item["end_s"]
        print(f"Encoding and decoding /{label}/, analysis {start:.2f}–{end:.2f} s", flush=True)
        destination = output / label
        destination.mkdir()
        original, original_sr = sf.read(path, always_2d=True)
        original = original.mean(axis=1)
        code, baseline, timing, checks = encode_decode(coder, path, device)
        np.savez_compressed(destination / "baseline_code.npz", **code)
        np.savetxt(destination / "ema.csv", np.column_stack([
            np.arange(len(code["ema"])) / coder.ft_sr, code["ema"]]), delimiter=",",
            header="time_s," + ",".join(EMA_NAMES), comments="")
        for name, wav, sr in [("original", original, original_sr),
                              ("baseline", baseline, coder.output_sr)]:
            sf.write(destination / f"{name}.wav", wav, sr, subtype="FLOAT")
            sf.write(destination / f"{name}_interval.wav",
                     wav[round(start * sr):round(end * sr)], sr, subtype="FLOAT")
        measurements = {}
        sensitivity = {}
        for name, wav, sr in [("original", original, original_sr),
                              ("baseline", baseline, coder.output_sr)]:
            data = measure_and_save(destination, name, wav, sr, start, end, config["formants"])
            measurements[name] = data
            sensitivity[name] = {}
            for ceiling in config["formants"]["sensitivity_ceilings_hz"]:
                settings = dict(config["formants"], ceiling_hz=ceiling)
                alt = measure_and_save(destination, f"{name}_ceiling{ceiling}",
                                       wav, sr, start, end, settings)
                sensitivity[name][str(ceiling)] = alt["summary"]
        plot_result(destination, original, original_sr, baseline, coder.output_sr,
                    code, coder.ft_sr, start, end, measurements)
        targets = [f["target_hz"] for f in measurements["baseline"]["summary"]["formants"]]
        acoustics = compare_acoustics(measurements["original"], measurements["baseline"],
                                     config["formants"]["min_valid_fraction"],
                                     config["formants"]["min_valid_frames"])
        with (destination / "acoustic_comparison.csv").open("w", newline="") as output_csv:
            writer = csv.DictWriter(output_csv, fieldnames=list(acoustics["comparison"][0]))
            writer.writeheader()
            writer.writerows(acoustics["comparison"])
        summary = {
            "schema_version": 2, "feature_names": FEATURE_NAMES,
            "label": label, "interval_s": [start, end], "input": provenance[label],
            "input_sr": original_sr, "output_sr": coder.output_sr, "feature_rate_hz": coder.ft_sr,
            "input_duration_s": len(original) / original_sr,
            "baseline_duration_s": len(baseline) / coder.output_sr,
            "feature_shapes": {k: list(np.shape(v)) for k, v in code.items()},
            "ft_len": int(code["ft_len"]), "timing": timing, "checks": checks,
            "formants": {name: data["summary"] for name, data in measurements.items()},
            "baseline_formant_target_hz": targets,
            "baseline_target_hz": acoustics["baseline_target_hz"],
            "baseline_target_usable": acoustics["baseline_target_usable"],
            "acoustics": acoustics,
            "ceiling_sensitivity": sensitivity,
        }
        write_json(destination / "summary.json", summary)
        summaries.append(summary)
        print(json.dumps({"label": label, "feature_names": FEATURE_NAMES,
                          "baseline_target_hz": acoustics["baseline_target_hz"],
                          "usable": summary["baseline_target_usable"], "checks": checks}), flush=True)
    write_json(output / "summary.json", {"schema_version": 2, "feature_names": FEATURE_NAMES,
                                         "model_load_s": load_s, "results": summaries})
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
