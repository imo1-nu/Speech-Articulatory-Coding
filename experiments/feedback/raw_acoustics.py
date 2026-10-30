"""Praat measurements and finite-value medians, with no acoustic quality filters."""

import numpy as np
import parselmouth

from .formants import FormantExtractionError


def raw_summary(values):
    """Ignore only undefined/nonfinite numbers; do not apply acoustic thresholds."""
    values = np.asarray(values, dtype=float)
    available = values[np.isfinite(values)]
    median = float(np.median(available)) if len(available) else None
    return {"median_hz": median, "target_hz": median,
            "std_hz": float(np.std(available)) if len(available) else None,
            "available_frames": int(len(available)), "total_frames": len(values),
            "status": "raw_no_quality_filter" if len(available) else "missing"}


def extract_raw_acoustics(waveform, sample_rate, start, end, settings):
    waveform = np.asarray(waveform, dtype=np.float64)
    if (waveform.ndim != 1 or not np.isfinite(waveform).all()
            or len(waveform) < sample_rate * .1 or waveform.std() < 1e-8):
        raise FormantExtractionError("Expected finite, non-silent mono audio >= 0.1 s")
    if not 0 <= start < end <= len(waveform) / sample_rate:
        raise FormantExtractionError("Analysis interval is outside waveform")
    times = np.arange(start + settings["time_step_s"] / 2, end, settings["time_step_s"])
    if not len(times):
        raise FormantExtractionError("No analysis frames")
    sound = parselmouth.Sound(waveform, sampling_frequency=sample_rate)
    try:
        formant = sound.to_formant_burg(time_step=settings["time_step_s"],
            max_number_of_formants=settings["max_number_of_formants"], maximum_formant=settings["ceiling_hz"],
            window_length=settings["window_length_s"], pre_emphasis_from=settings["pre_emphasis_hz"])
        pitch = sound.to_pitch_ac(time_step=settings["time_step_s"], pitch_floor=settings["pitch_floor_hz"],
            pitch_ceiling=settings["pitch_ceiling_hz"], voicing_threshold=settings["voicing_threshold"])
    except parselmouth.PraatError as exc:
        raise FormantExtractionError(str(exc)) from exc
    raw = np.array([[formant.get_value_at_time(j, t) for j in (1, 2, 3)] for t in times])
    bandwidth = np.array([[formant.get_bandwidth_at_time(j, t) for j in (1, 2, 3)] for t in times])
    f0 = np.array([pitch.get_value_at_time(t) for t in times])
    summaries = [dict(raw_summary(raw[:, j]), formant=f"F{j+1}") for j in range(3)]
    return {"time_s": times, "raw_hz": raw, "filtered_hz": raw.copy(),
            "valid": np.isfinite(raw), "bandwidth_hz": bandwidth, "f0_hz": f0,
            "f0_filtered_hz": f0.copy(), "f0_valid": np.isfinite(f0), "voiced": np.isfinite(f0),
            "reasons": [["" for _ in range(3)] for _ in times],
            "summary": {"frames": len(times), "feature_policy": "raw", "quality_filtering": False,
                        "f0": dict(raw_summary(f0), feature="F0"), "formants": summaries}}
