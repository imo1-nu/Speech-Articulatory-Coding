"""Plot all saved F1/F2/F3 estimates in blue, without validity filtering."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Experiment 1 results directory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    inputs = sorted(args.root.glob("*/original_formants.npz"))
    if not inputs:
        parser.error("No saved original_formants.npz files found")
    for source in inputs:
        destination = source.parent / "formants_raw.png"
        if destination.exists() and not args.overwrite:
            parser.error(f"Output exists: {destination}; use --overwrite")
    for source in inputs:
        fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True,
                                 sharey="row", constrained_layout=True)
        fig.suptitle(f"/{source.parent.name}/: all F1–F3 estimates (no validity filtering)")
        for col, (name, label) in enumerate([
            ("original", "Original"), ("baseline", "SPARC resynthesis")
        ]):
            with np.load(source.parent / f"{name}_formants.npz", allow_pickle=False) as saved:
                times, values = saved["time_s"], saved["raw_hz"]
            for j in range(3):
                axes[j, col].plot(times, values[:, j], "o-", color="tab:blue", markersize=3)
                axes[j, col].set(title=f"{label} F{j + 1}", xlabel="WAV time (s)", ylabel="Hz")
        destination = source.parent / "formants_raw.png"
        fig.savefig(destination, dpi=160)
        plt.close(fig)
        print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
