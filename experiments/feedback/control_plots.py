"""Export figures from saved finite-difference and correction experiments."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .adapter import EMA_NAMES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    root = args.root
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    summary = json.loads((root / "jacobian/jacobian_summary.json").read_text())
    matrices = []
    for condition in summary["conditions"]:
        epsilon = condition["epsilon_sd"]
        with np.load(root / f"jacobian/jacobian_eps{epsilon}.npz") as data:
            matrices.append(data["hz_per_reference_sd"])
    limit = max(float(np.nanmax(np.abs(m))) for m in matrices)
    fig, axes = plt.subplots(2, 2, figsize=(15, 7), constrained_layout=True)
    for ax, matrix, condition in zip(axes.flat, matrices, summary["conditions"]):
        im = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
        ax.set(yticks=[0, 1], yticklabels=["F1", "F2"], xticks=range(12),
               xticklabels=EMA_NAMES, title=f"epsilon={condition['epsilon_sd']} reference SD")
        ax.tick_params(axis="x", rotation=45)
    fig.colorbar(im, ax=axes, label="Hz / reference SD (fixed temporal window)")
    fig.savefig(figures / "jacobians.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    epsilons = [c["epsilon_sd"] for c in summary["conditions"]]
    for j, ax in enumerate(axes):
        for k, name in enumerate(EMA_NAMES):
            ax.plot(epsilons, [m[j, k] for m in matrices], "o-", label=name)
        ax.set(xscale="log", xlabel="epsilon (reference SD)", ylabel="Hz / reference SD", title=f"F{j+1} sensitivity")
        ax.axhline(0, color="gray", linewidth=.5)
    axes[0].legend(ncol=6, fontsize=8)
    fig.savefig(figures / "epsilon_sensitivity.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for epsilon in epsilons:
        rows = [r for r in summary["validation"] if r["epsilon_sd"] == epsilon and r["valid"]]
        for j, ax in enumerate(axes):
            ax.scatter([r["actual_hz"][j] for r in rows], [r["predicted_hz"][j] for r in rows], label=str(epsilon))
            ax.set(xlabel="Actual change (Hz)", ylabel="Predicted change (Hz)", title=f"F{j+1}")
    for ax in axes:
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0]); hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=.8)
        ax.legend(title="epsilon (SD)")
    fig.savefig(figures / "prediction_vs_actual.png", dpi=160)
    plt.close(fig)
    recovery_path = root / "recovery/recovery_summary.json"
    if recovery_path.exists():
        recovery = json.loads(recovery_path.read_text())
        lambdas = sorted(set(r["regularization"] for r in recovery["aggregates"]))
        fig, axes = plt.subplots(1, len(lambdas), figsize=(15, 5), sharey=True, constrained_layout=True)
        for ax, regularization in zip(np.atleast_1d(axes), lambdas):
            for method in ["jacobian", "random"]:
                rows = [r for r in recovery["aggregates"] if r["regularization"] == regularization and r["method"] == method]
                ax.plot([r["gain"] for r in rows], [r["median_improvement"] for r in rows], "o-", label=method)
            ax.axhline(0, color="gray", linestyle="--", label="no correction")
            ax.set(xlabel="Gain", ylabel="Median error reduction fraction", title=f"lambda={regularization}")
            ax.legend()
        fig.savefig(figures / "recovery_errors.png", dpi=160)
        plt.close(fig)
    source_path = root / "source/source_summary.json"
    if source_path.exists():
        source = json.loads(source_path.read_text())
        accepted = [r for r in source["trajectory"] if r["status"] == "accepted"]
        target, initial = np.array(source["original_target_hz"]), np.array(source["baseline_hz"])
        features = np.vstack([initial, *[r["actual_hz"] for r in accepted]])
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for j in range(2):
            axes[j].plot(features[:, j], "o-", label="Resynthesis")
            axes[j].axhline(target[j], color="green", linestyle="--", label="Original target")
            axes[j].set(xlabel="Accepted iteration", ylabel="Hz", title=f"F{j+1}")
            axes[j].legend()
        axes[2].plot([source["before"], *[r["after"] for r in accepted]], "o-")
        axes[2].set(xlabel="Accepted iteration", ylabel="Normalized F1/F2 error norm")
        fig.savefig(figures / "original_target_correction.png", dpi=160)
        plt.close(fig)
        baseline = np.load(root / "source/baseline_code.npz")["ema"]
        corrected = baseline
        if accepted:
            corrected = np.load(root / f"source/samples/{accepted[-1]['evaluation_id']}.npz")["ema"]
        fig, axes = plt.subplots(6, 2, figsize=(13, 12), constrained_layout=True)
        for j, ax in enumerate(axes.flat):
            ax.plot(np.arange(len(baseline))/50, baseline[:, j], label="Baseline")
            ax.plot(np.arange(len(corrected))/50, corrected[:, j], label="Corrected")
            ax.set(title=EMA_NAMES[j], xlabel="Nominal time (s)")
        axes.flat[0].legend()
        fig.savefig(figures / "corrected_ema.png", dpi=160)
        plt.close(fig)
    print(f"Saved figures: {figures}")


if __name__ == "__main__":
    main()
