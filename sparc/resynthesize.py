"""Resynthesize an audio file through SPARC's articulatory representation."""

import argparse
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch

from .sparc import load_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input audio file (WAV, FLAC, etc.).")
    parser.add_argument("-o", "--output", type=Path,
                        help="Output WAV (default: outputs/<input>_resynth.wav).")
    parser.add_argument("--features", type=Path, help="Also save encoded features as .npz.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files.")
    args = parser.parse_args(argv)

    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    output = args.output or Path("outputs") / f"{args.input.stem}_resynth.wav"
    if output.suffix.lower() != ".wav":
        parser.error("--output must have a .wav extension")
    if args.features is not None and args.features.suffix.lower() != ".npz":
        parser.error("--features must have a .npz extension")
    for destination in (output, args.features):
        if destination is None:
            continue
        if destination.resolve() == args.input.resolve():
            parser.error("Output must differ from the input file")
        if destination.exists() and not args.overwrite:
            parser.error(f"Output already exists: {destination}; use --overwrite to replace it")

    try:
        waveform, sample_rate = sf.read(args.input, dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as exc:
        parser.error(f"Cannot read input audio: {exc}")
    if len(waveform) / sample_rate < 0.5:
        parser.error("Input audio must be at least 0.5 seconds long")
    waveform = waveform.mean(axis=1)
    if not np.isfinite(waveform).all() or waveform.std() < 1e-8:
        parser.error("Input must contain finite, non-silent audio")

    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; run in an environment with GPU access")

    print("Loading SPARC (multi, cuda)...", file=sys.stderr, flush=True)
    coder = load_model("multi", device="cuda")
    # Passing the original file preserves its sample rate for the model's resampler.
    print(f"Encoding {args.input}...", file=sys.stderr, flush=True)
    with torch.inference_mode():
        code = coder.encode(args.input)
        print("Synthesizing...", file=sys.stderr, flush=True)
        synthesized = coder.decode(**code)
    if synthesized.ndim != 1 or not synthesized.size or not np.isfinite(synthesized).all():
        raise RuntimeError("The model did not produce a finite mono waveform")

    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, synthesized, coder.output_sr, subtype="PCM_16")
    if args.features is not None:
        args.features.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.features, **code)
        print(f"Features: {args.features}")
    duration = len(synthesized) / coder.output_sr
    print(f"Audio: {output} ({coder.output_sr} Hz, {duration:.2f} seconds, cuda)")


if __name__ == "__main__":
    main()
