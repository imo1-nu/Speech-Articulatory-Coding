"""Extract SPARC articulatory trajectories and pitch from a JSUT corpus."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from .sparc import load_model

CHANNELS = (
    "TDX",
    "TDY",
    "TBX",
    "TBY",
    "TTX",
    "TTY",
    "LIX",
    "LIY",
    "ULX",
    "ULY",
    "LLX",
    "LLY",
)
FEATURES = {"ema": 12, "pitch": 1, "periodicity": 1, "loudness": 1}
REPO_DIR = Path(__file__).resolve().parents[1]


def extract_features(coder, source):
    """Process one utterance without batch padding or speaker/synthesis models."""
    waveform, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    if len(waveform) / sample_rate < 0.5:
        raise ValueError("Audio must be at least 0.5 seconds long")
    mono = waveform.mean(axis=1)
    if not np.isfinite(waveform).all() or mono.std() < 1e-8:
        raise ValueError("Audio must be finite and non-silent")

    with torch.inference_mode():
        # Keep the file's sampling rate: SPARC resamples JSUT's audio to 16 kHz.
        outputs = coder.inverter(source, outputs={})
        outputs = coder.source_extractor(outputs["wav"], outputs)
    features = {}
    for name, width in FEATURES.items():
        values = np.asarray(outputs[name][0], dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != width or not len(values):
            raise ValueError(f"Unexpected {name} shape: {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in {name}")
        features[name] = values

    # WavLM and the pitch tracker can differ by a frame at the utterance end.
    length = min(len(values) for values in features.values())
    features = {name: values[:length] for name, values in features.items()}
    features.update(
        time=np.arange(length, dtype=np.float64) / coder.ft_sr,
        frame_rate=np.float64(coder.ft_sr),
        ft_len=np.int64(length),
        channels=np.asarray(CHANNELS),
        source_sample_rate=np.int64(sample_rate),
        duration_seconds=np.float64(len(waveform) / sample_rate),
    )
    return features


def reusable_output(destination, source_relative, source):
    """Only skip complete outputs that still match their input audio."""
    if not destination.is_file():
        return False
    try:
        with np.load(destination, allow_pickle=False) as data:
            valid = (
                str(data["source"]) == source_relative
                and int(data["source_size"]) == source.stat().st_size
                and int(data["source_mtime_ns"]) == source.stat().st_mtime_ns
                and str(data["model"]) == "feature_extraction"
                and float(data["frame_rate"]) == 50
                and np.array_equal(data["channels"], CHANNELS)
            )
            length = int(data["ft_len"])
            return (
                valid
                and length > 0
                and all(
                    data[name].shape == (length, width)
                    and np.isfinite(data[name]).all()
                    for name, width in FEATURES.items()
                )
                and np.array_equal(data["time"], np.arange(length) / 50)
            )
    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
        return False


def save_features(destination, features):
    """Publish each NPZ only after it is completely written."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, suffix=".npz", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **features)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsut-dir",
        type=Path,
        default=Path("datasets/jsut/jsut_ver1.1"),
        help="Extracted JSUT root containing basic5000/, travel1000/, etc.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/jsut"))
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["all"],
        help="Subset directory names, or all (default).",
    )
    parser.add_argument(
        "--limit", type=int, help="Maximum utterances in total (for a trial run)."
    )
    parser.add_argument(
        "--device", default="auto", help="auto (default), cpu, cuda, or cuda:N"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute existing features."
    )
    args = parser.parse_args(argv)
    if not args.jsut_dir.is_dir():
        parser.error(
            f"JSUT directory does not exist: {args.jsut_dir}; unzip JSUT first"
        )
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    available = {
        path.parent.name: path for path in args.jsut_dir.glob("*/wav") if path.is_dir()
    }
    if not available:
        parser.error(f"No <subset>/wav directories found in {args.jsut_dir}")
    if "all" in args.subsets and args.subsets != ["all"]:
        parser.error("Use --subsets all alone, or specify subset names")
    subsets = (
        sorted(available) if args.subsets == ["all"] else sorted(set(args.subsets))
    )
    unknown = set(subsets) - available.keys()
    if unknown:
        parser.error(
            f"Unknown subsets: {', '.join(sorted(unknown))}; available: {', '.join(sorted(available))}"
        )
    sources = [
        source
        for subset in subsets
        for source in sorted(available[subset].glob("*.wav"))
    ]
    if args.limit is not None:
        sources = sources[: args.limit]
    if not sources:
        parser.error("No WAV files found in the selected subsets")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        selected_device = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        parser.error(f"Invalid --device: {exc}")
    if selected_device.type not in ("cpu", "cuda"):
        parser.error("--device must be cpu, cuda, cuda:N, or auto")
    if selected_device.type == "cuda" and (
        not torch.cuda.is_available()
        or (
            selected_device.index is not None
            and selected_device.index >= torch.cuda.device_count()
        )
    ):
        parser.error(f"CUDA device is unavailable: {device}")

    texts = {}
    for subset in subsets:
        transcript = args.jsut_dir / subset / "transcript_utf8.txt"
        if transcript.is_file():
            for line in transcript.read_text(encoding="utf-8-sig").splitlines():
                utterance, separator, text = line.partition(":")
                if separator:
                    texts[(subset, utterance)] = text

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    counts = {"saved": 0, "skipped": 0, "failed": 0}
    coder = None
    print(
        f"Selected {len(sources)} utterances; device={device}",
        file=sys.stderr,
        flush=True,
    )
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for source in tqdm(sources, desc="JSUT", unit="utterance"):
            subset = source.parent.parent.name
            source_relative = source.relative_to(args.jsut_dir).as_posix()
            destination = args.output_dir / subset / f"{source.stem}.npz"
            record = {
                "utterance_id": source.stem,
                "subset": subset,
                "source": source_relative,
                "output": destination.relative_to(args.output_dir).as_posix(),
                "text": texts.get((subset, source.stem), ""),
            }
            if not args.overwrite and reusable_output(
                destination, source_relative, source
            ):
                record["status"] = "skipped"
            else:
                if coder is None:
                    # Feature extraction loads no vocoder or trained speaker encoder.
                    print(
                        "Loading SPARC feature_extraction (CREPE tiny)...",
                        file=sys.stderr,
                        flush=True,
                    )
                    coder = load_model(
                        "feature_extraction",
                        device=device,
                        config=REPO_DIR / "configs/feature_extraction.yaml",
                    )
                try:
                    features = extract_features(coder, source)
                    features.update(
                        source=np.str_(source_relative),
                        text=np.str_(record["text"]),
                        model=np.str_("feature_extraction"),
                        source_size=np.int64(source.stat().st_size),
                        source_mtime_ns=np.int64(source.stat().st_mtime_ns),
                    )
                    save_features(destination, features)
                    record["status"] = "saved"
                except Exception as exc:
                    record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    tqdm.write(f"Failed: {source_relative}: {exc}", file=sys.stderr)
                    if selected_device.type == "cuda":
                        torch.cuda.empty_cache()
            counts[record["status"]] += 1
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()

    print(
        f"Saved: {counts['saved']}, skipped: {counts['skipped']}, failed: {counts['failed']}"
    )
    print(f"Features: {args.output_dir}\nManifest: {manifest_path}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
