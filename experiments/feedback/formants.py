"""Praat F0 and formant measurements with explicit quality decisions."""

import numpy as np
import parselmouth


class FormantExtractionError(ValueError):
    """The signal or analysis interval cannot be analysed."""


def extract_formants(waveform, sample_rate, start, end, settings):
    waveform = np.asarray(waveform, dtype=np.float64)
    if (waveform.ndim != 1 or not np.isfinite(waveform).all()
            or len(waveform) < sample_rate * 0.1 or waveform.std() < 1e-8):
        raise FormantExtractionError("Expected finite, non-silent mono audio >= 0.1 s")
    if not 0 <= start < end <= len(waveform) / sample_rate:
        raise FormantExtractionError("Analysis interval is outside the waveform")
    step = settings["time_step_s"]
    times = np.arange(start + step / 2, end, step)
    if not len(times):
        raise FormantExtractionError("No frames in the analysis interval")
    sound = parselmouth.Sound(waveform, sampling_frequency=sample_rate)
    try:
        formant = sound.to_formant_burg(
            time_step=step,
            max_number_of_formants=settings["max_number_of_formants"],
            maximum_formant=settings["ceiling_hz"],
            window_length=settings["window_length_s"],
            pre_emphasis_from=settings["pre_emphasis_hz"],
        )
        pitch = sound.to_pitch_ac(
            time_step=step, pitch_floor=settings["pitch_floor_hz"],
            pitch_ceiling=settings["pitch_ceiling_hz"],
            voicing_threshold=settings["voicing_threshold"],
        )
    except parselmouth.PraatError as exc:
        raise FormantExtractionError(str(exc)) from exc
    raw = np.array([[formant.get_value_at_time(j, t) for j in (1, 2, 3)] for t in times])
    bandwidth = np.array([
        [formant.get_bandwidth_at_time(j, t) for j in (1, 2, 3)] for t in times
    ])
    f0 = np.array([pitch.get_value_at_time(t) for t in times])
    voiced = np.isfinite(f0) & (f0 > 0)
    f0_values = f0[voiced]
    f0_median = float(np.median(f0_values)) if len(f0_values) else None
    f0_sd = float(np.std(f0_values)) if len(f0_values) else None
    f0_coverage_ok = (len(f0_values) >= settings["min_valid_frames"]
                      and voiced.mean() >= settings["min_valid_fraction"])
    f0_stable = bool(f0_coverage_ok and
                     f0_sd / f0_median <= settings["max_coefficient_of_variation"])
    f0_summary = {
        "feature": "F0", "estimator": "Praat autocorrelation (to_pitch_ac)",
        "valid_frames": int(voiced.sum()), "valid_fraction": float(voiced.mean()),
        "median_hz": f0_median, "std_hz": f0_sd,
        "iqr_hz": float(np.subtract(*np.percentile(f0_values, [75, 25]))) if len(f0_values) else None,
        "status": "ok" if f0_stable else "unstable" if f0_coverage_ok else "insufficient_valid_frames",
        "target_hz": f0_median if f0_stable else None,
    }
    reasons = [[[] for _ in range(3)] for _ in times]
    for i in range(len(times)):
        for j in range(3):
            if not voiced[i]:
                reasons[i][j].append("unvoiced")
            if not np.isfinite(raw[i, j]):
                reasons[i][j].append("missing")
            elif not settings["ranges_hz"][j][0] <= raw[i, j] <= settings["ranges_hz"][j][1]:
                reasons[i][j].append("frequency_range")
            if (not np.isfinite(bandwidth[i, j]) or not
                    0 < bandwidth[i, j] <= settings["max_bandwidth_hz"][j]):
                reasons[i][j].append("bandwidth")
        if np.isfinite(raw[i]).all() and not np.all(np.diff(raw[i]) > 0):
            for j in range(3):
                reasons[i][j].append("unordered")
    # Compare to a local median of initially acceptable neighbours. Do not
    # interpolate missing values, smooth the reported data, or renumber peaks.
    initial = np.array([[not r for r in row] for row in reasons])
    for i in range(len(times)):
        for j in range(3):
            lo, hi = max(0, i - 2), min(len(times), i + 3)
            neighbours = raw[lo:hi, j][initial[lo:hi, j]]
            if (initial[i, j] and len(neighbours) >= 3 and
                    abs(raw[i, j] - np.median(neighbours)) > settings["local_deviation_hz"][j]):
                reasons[i][j].append("local_instability")
    valid = np.array([[not r for r in row] for row in reasons])
    filtered = np.where(valid, raw, np.nan)
    summaries = []
    for j in range(3):
        values = filtered[:, j][valid[:, j]]
        coverage = float(valid[:, j].mean())
        median = float(np.median(values)) if len(values) else None
        sd = float(np.std(values)) if len(values) else None
        ok = (len(values) >= settings["min_valid_frames"]
              and coverage >= settings["min_valid_fraction"])
        stable = bool(ok and sd / median <= settings["max_coefficient_of_variation"])
        summaries.append({
            "formant": f"F{j + 1}", "valid_frames": int(len(values)),
            "valid_fraction": coverage, "median_hz": median, "std_hz": sd,
            "iqr_hz": float(np.subtract(*np.percentile(values, [75, 25]))) if len(values) else None,
            "status": "ok" if stable else "unstable" if ok else "insufficient_valid_frames",
            "target_hz": median if stable else None,
        })
    return {
        "time_s": times, "f0_hz": f0, "voiced": voiced,
        "f0_valid": voiced.copy(), "f0_filtered_hz": np.where(voiced, f0, np.nan),
        "raw_hz": raw, "bandwidth_hz": bandwidth,
        "valid": valid, "filtered_hz": filtered,
        "reasons": [[";".join(r) for r in row] for row in reasons],
        "summary": {"frames": len(times), "f0": f0_summary,
                    "voiced_fraction": float(voiced.mean()),
                    "joint_valid_fraction": float(valid.all(axis=1).mean()),
                    "formants": summaries},
    }
