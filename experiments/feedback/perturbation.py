"""Smooth, fixed-shape EMA perturbations in explicitly scaled coordinates."""

import numpy as np


def gaussian_window(length, feature_rate, center_s, sigma_s, radius_s):
    if sigma_s <= 0 or radius_s <= 0:
        raise ValueError("Window sigma and radius must be positive")
    distance = np.abs(np.arange(length) / feature_rate - center_s)
    edge = np.exp(-0.5 * (radius_s / sigma_s) ** 2)
    window = np.where(distance < radius_s,
                      (np.exp(-0.5 * (distance / sigma_s) ** 2) - edge) / (1 - edge), 0)
    if window.max() <= 0:
        raise ValueError("Window does not overlap EMA frames")
    return window / window.max()


def apply_perturbation(code, window, reference_sd, delta_sd):
    ema = np.asarray(code["ema"])
    reference_sd, delta_sd = np.asarray(reference_sd), np.asarray(delta_sd)
    if (window.shape != (len(ema),) or reference_sd.shape != (12,) or delta_sd.shape != (12,)
            or np.any(reference_sd <= 0) or not np.isfinite(delta_sd).all()):
        raise ValueError("Invalid perturbation dimensions or scale")
    result = {key: np.array(value, copy=True) for key, value in code.items()}
    result["ema"] = (ema + window[:, None] * reference_sd * delta_sd).astype(ema.dtype)
    return result
