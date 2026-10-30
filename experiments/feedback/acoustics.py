"""Ordered acoustic features measured from audio, separate from SPARC inputs."""

import numpy as np


FEATURE_NAMES = ["F0", "F1", "F2", "F3"]


def feature_arrays(data):
    return {
        "time_s": data["time_s"], "feature_names": np.array(FEATURE_NAMES),
        "raw_hz": np.column_stack([data["f0_hz"], data["raw_hz"]]),
        "filtered_hz": np.column_stack([data["f0_filtered_hz"], data["filtered_hz"]]),
        "valid": np.column_stack([data["f0_valid"], data["valid"]]),
    }


def compare_acoustics(original, baseline, min_valid_fraction, min_valid_frames=10):
    if not np.allclose(original["time_s"], baseline["time_s"], rtol=0, atol=1e-9):
        raise ValueError("Original and baseline must use the same analysis times")
    source = feature_arrays(original)
    target = feature_arrays(baseline)
    source_summary = [original["summary"]["f0"], *original["summary"]["formants"]]
    target_summary = [baseline["summary"]["f0"], *baseline["summary"]["formants"]]
    comparisons = []
    for j, name in enumerate(FEATURE_NAMES):
        a, b = source_summary[j]["target_hz"], target_summary[j]["target_hz"]
        paired = source["valid"][:, j] & target["valid"][:, j]
        usable = a is not None and b is not None
        # Paired error metrics additionally require enough shared valid frames.
        enough_pairs = bool(usable and paired.mean() >= min_valid_fraction
                            and paired.sum() >= min_valid_frames)
        difference = target["raw_hz"][paired, j] - source["raw_hz"][paired, j]
        comparisons.append({
            "feature": name, "original_hz": a, "baseline_hz": b,
            "median_difference_hz": b - a if usable else None,
            "median_relative_difference_percent": 100 * (b - a) / a if usable else None,
            "median_difference_cents": float(1200 * np.log2(b / a)) if usable and name == "F0" else None,
            "paired_valid_frames": int(paired.sum()),
            "paired_valid_fraction": float(paired.mean()),
            "paired_mae_hz": float(np.mean(np.abs(difference))) if enough_pairs else None,
            "paired_rmse_hz": float(np.sqrt(np.mean(difference ** 2))) if enough_pairs else None,
        })
    baseline_target = [f["target_hz"] for f in target_summary]
    joint_fraction = float(target["valid"].all(axis=1).mean())
    return {
        "feature_names": FEATURE_NAMES,
        "original_hz": [f["target_hz"] for f in source_summary],
        "baseline_target_hz": baseline_target,
        "baseline_joint_valid_fraction": joint_fraction,
        "baseline_target_usable": all(v is not None for v in baseline_target)
            and joint_fraction >= min_valid_fraction,
        "comparison": comparisons,
    }
