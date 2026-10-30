"""Finite differences, perturbation recovery, then original-audio matching."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf
import torch
import yaml

from .acoustics import FEATURE_NAMES, feature_arrays
from .adapter import synchronize
from .controller import bound_excursion, damped_correction
from .formants import FormantExtractionError, extract_formants
from .jacobian import central_difference, matrix_diagnostics, prediction_metrics
from .perturbation import apply_perturbation, gaussian_window
from .raw_acoustics import extract_raw_acoustics
from .run import environment, sha256


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def primary_from(data, names):
    summaries = [data["summary"]["f0"], *data["summary"]["formants"]]
    all_features = np.array([np.nan if x["target_hz"] is None else x["target_hz"] for x in summaries])
    return all_features[[FEATURE_NAMES.index(n) for n in names]], all_features


class Evaluator:
    def __init__(self, coder, code, sd, window, item, settings, config, output, device):
        self.coder, self.code, self.sd, self.window = coder, code, sd, window
        self.item, self.settings, self.config, self.output, self.device = item, settings, config, output, device
        self.samples = output / "samples"
        self.samples.mkdir()
        self.cache, self.records, self.cache_hits = {}, [], 0
        self.start = perf_counter()
        self.baseline = None
        self.raw_mode = config.get("feature_policy", "quality") == "raw"
        self.extract = extract_raw_acoustics if self.raw_mode else extract_formants
        self.log = (output / "evaluations.jsonl").open("w", buffering=1)

    def timing(self):
        decodes = [r["decode_s"] for r in self.records]
        return {"decode_count": len(decodes), "cache_hits": self.cache_hits,
                "decode_total_s": sum(decodes), "decode_median_s": float(np.median(decodes)) if decodes else None,
                "decode_min_s": min(decodes) if decodes else None, "decode_max_s": max(decodes) if decodes else None,
                "analysis_total_s": sum(r["analysis_s"] for r in self.records),
                "storage_total_s": sum(r["storage_s"] for r in self.records),
                "evaluation_wall_s": perf_counter() - self.start}

    def evaluate(self, delta, tag, force=False):
        code = apply_perturbation(self.code, self.window, self.sd, np.asarray(delta))
        key = hashlib.sha256(code["ema"].tobytes()).hexdigest()
        if key in self.cache and not force:
            self.cache_hits += 1
            return self.cache[key]
        before = {k: np.array(v, copy=True) for k, v in code.items()}
        synchronize(self.device)
        begin = perf_counter()
        with torch.inference_mode():
            wav = self.coder.decode(**code)
        synchronize(self.device)
        decode_s = perf_counter() - begin
        if wav.ndim != 1 or not np.isfinite(wav).all() or not len(wav):
            raise RuntimeError("Invalid decoded audio")
        for name in code:
            if not np.array_equal(code[name], before[name]):
                raise RuntimeError(f"Decoder mutated {name}")
            if name != "ema" and not np.array_equal(code[name], self.code[name]):
                raise RuntimeError(f"Non-EMA condition changed: {name}")
        begin = perf_counter()
        error = None
        try:
            data = self.extract(wav, self.coder.output_sr, self.item["start_s"],
                                    self.item["end_s"], self.settings)
            primary, observed = primary_from(data, self.config["primary_features"])
        except FormantExtractionError as exc:
            data, primary, observed, error = None, np.full(2, np.nan), np.full(4, np.nan), str(exc)
        tracking_ok, max_shifts = (None if self.raw_mode else True), None
        if data is not None and self.baseline is not None and not self.raw_mode:
            shifts = np.abs(data["raw_hz"][:, :2] - self.baseline["data"]["raw_hz"][:, :2])
            max_shifts = [float(np.nanmax(shifts[:, j])) if np.isfinite(shifts[:, j]).any() else None for j in range(2)]
            tracking_ok = all(x is not None and x <= limit for x, limit in zip(
                max_shifts, self.config["validation"]["max_primary_frame_shift_hz"]))
            if not tracking_ok:
                primary[:] = np.nan
        analysis_s = perf_counter() - begin
        identifier = f"{len(self.records):05d}"
        begin = perf_counter()
        sf.write(self.samples / f"{identifier}.wav", wav, self.coder.output_sr, subtype="FLOAT")
        arrays = feature_arrays(data) if data is not None else {}
        np.savez_compressed(self.samples / f"{identifier}.npz", ema=code["ema"], delta_sd=delta,
                            **arrays, **({"bandwidth_hz": data["bandwidth_hz"]} if data is not None else {}))
        storage_s = perf_counter() - begin
        record = {"id": identifier, "tag": tag, "delta_sd": delta, "ema_sha256": key,
                  "primary_hz": primary, "observed_hz": observed,
                  "tracking_ok": tracking_ok, "max_primary_frame_shift_hz": max_shifts,
                  "feature_policy": "raw" if self.raw_mode else "quality",
                  "quality": data["summary"] if data is not None and not self.raw_mode else None,
                  "raw_summary": data["summary"] if data is not None and self.raw_mode else None,
                  "rejection_reasons": data["reasons"] if data is not None and not self.raw_mode else None,
                  "error": error, "decode_s": decode_s, "analysis_s": analysis_s,
                  "storage_s": storage_s, "conditions_unchanged": True}
        self.records.append(record)
        self.log.write(json.dumps(clean(record), allow_nan=False) + "\n")
        result = {"id": identifier, "a": primary, "observed": observed, "data": data,
                  "wav_sha256": hashlib.sha256(wav.tobytes()).hexdigest()}
        self.cache[key] = result
        if len(self.records) == 1 or len(self.records) % 25 == 0:
            print(json.dumps(clean({"progress": self.timing()})), flush=True)
        return result


def estimate(evaluator, point, epsilon, tag):
    calls = []
    def measure(z):
        result = evaluator.evaluate(z, tag)
        calls.append(result)
        return result["a"]
    matrix, plus, minus = central_difference(measure, point, epsilon)
    all_plus = np.stack([calls[2 * j]["observed"] for j in range(12)], axis=1)
    all_minus = np.stack([calls[2 * j + 1]["observed"] for j in range(12)], axis=1)
    return matrix, plus, minus, (all_plus - all_minus) / (2 * epsilon), [c["id"] for c in calls]


def random_direction(seed):
    direction = np.random.default_rng(seed).normal(size=12)
    return direction / np.linalg.norm(direction)


def run_jacobian(ev, config, baseline, scale, output):
    zero = np.zeros(12)
    matrices, conditions, validation = {}, [], []
    for epsilon in config["epsilons_sd"]:
        matrix, plus, minus, all_matrix, ids = estimate(ev, zero, epsilon, f"jacobian_eps{epsilon}")
        matrices[epsilon] = matrix
        np.savez_compressed(output / f"jacobian_eps{epsilon}.npz", hz_per_reference_sd=matrix,
                            hz_per_ema_unit=matrix / ev.sd[None, :],
                            normalized=matrix / scale[:, None], observed_hz_per_reference_sd=all_matrix,
                            plus_hz=plus, minus_hz=minus, evaluation_ids=np.array(ids))
        condition = {"epsilon_sd": epsilon, "diagnostics": matrix_diagnostics(matrix / scale[:, None]),
                     "valid_columns": int(np.isfinite(matrix).all(axis=0).sum()),
                     "central_symmetry_residual_hz": np.linalg.norm(plus + minus - 2 * baseline[:, None], axis=0)}
        for amplitude in config["validation"]["amplitudes_sd"]:
            for seed in config["validation"]["seeds"]:
                direction = random_direction(seed)
                for sign in (-1, 1):
                    delta = sign * amplitude * direction
                    result = ev.evaluate(delta, f"heldout_seed{seed}_amp{amplitude}_sign{sign}")
                    predicted = matrix @ delta
                    actual = result["a"] - baseline
                    metrics = prediction_metrics(predicted / scale, actual / scale)
                    validation.append({"epsilon_sd": epsilon, "amplitude_sd": amplitude,
                                       "seed": seed, "sign": sign, "evaluation_id": result["id"],
                                       "predicted_hz": predicted, "actual_hz": actual, **metrics})
        relevant = [r for r in validation if r["epsilon_sd"] == epsilon]
        errors = [r["relative_error"] for r in relevant if r["relative_error"] is not None]
        cosines = [r["cosine"] for r in relevant if r["cosine"] is not None]
        condition.update({"valid_validation_fraction": np.mean([r["valid"] for r in relevant]),
                          "median_relative_error": float(np.median(errors)) if errors else None,
                          "median_cosine": float(np.median(cosines)) if cosines else None})
        gate = config["validation"]
        condition["passed"] = bool(condition["diagnostics"]["rank"] == 2
            and condition["valid_validation_fraction"] >= gate["min_valid_fraction"]
            and condition["median_relative_error"] is not None
            and condition["median_relative_error"] <= gate["max_median_relative_prediction_error"]
            and condition["median_cosine"] is not None and condition["median_cosine"] >= gate["min_median_cosine"])
        conditions.append(condition)
        print(json.dumps(clean({"epsilon_result": condition, "timing": ev.timing()})), flush=True)
    enforce_gate = config.get("require_jacobian_gate", True)
    candidates = [r for r in conditions if r["passed"]] if enforce_gate else [
        r for r in conditions if r["diagnostics"]["valid"] and r["median_relative_error"] is not None]
    chosen = min(candidates, key=lambda r: r["median_relative_error"])["epsilon_sd"] if candidates else None
    summary = {"conditions": conditions, "selected_epsilon_sd": chosen,
               "validation_gate_enforced": enforce_gate,
               "selection_basis": "Lowest held-out median relative error among gate passes" if enforce_gate
                   else "Lowest held-out median relative error among numerically finite Jacobians; no quality gate",
               "validation": validation}
    save_json(output / "jacobian_summary.json", summary)
    return summary


def error_norm(target, actual, scale):
    if not np.isfinite(target).all() or not np.isfinite(actual).all():
        return None
    return float(np.linalg.norm((target - actual) / scale))


def run_recovery(ev, config, baseline, scale, epsilon, output):
    rows = []
    settings = config["recovery"]
    for amplitude in settings["amplitudes_sd"]:
        for seed in settings["seeds"]:
            point = amplitude * random_direction(seed)
            perturbed = ev.evaluate(point, f"recovery_perturbed_{amplitude}_{seed}")
            before = error_norm(baseline, perturbed["a"], scale)
            matrix, _, _, _, ids = estimate(ev, point, epsilon, f"recovery_jacobian_{amplitude}_{seed}")
            for regularization in settings["regularizations"]:
                for gain in settings["gains"]:
                    valid = before is not None and np.isfinite(matrix).all()
                    correction = damped_correction(matrix / scale[:, None], (baseline - perturbed["a"]) / scale,
                        regularization, gain, settings["max_step_norm_sd"]) if valid else None
                    for method in ("none", "jacobian", "random"):
                        if method == "none":
                            result, after, displacement = perturbed, before, np.zeros(12)
                        elif valid:
                            step = correction if method == "jacobian" else np.linalg.norm(correction) * random_direction(seed + 10000)
                            corrected = bound_excursion(point + step, settings["max_excursion_norm_sd"])
                            displacement = corrected - point
                            result = ev.evaluate(corrected, f"recovery_{method}_{amplitude}_{seed}_l{regularization}_g{gain}")
                            after = error_norm(baseline, result["a"], scale)
                        else:
                            result, after, displacement = None, None, None
                        rows.append({"amplitude_sd": amplitude, "seed": seed, "regularization": regularization,
                                     "gain": gain, "method": method, "before": before, "after": after,
                                     "improvement": (before - after) / before if after is not None and before > 1e-12 else None,
                                     "correction_norm_sd": np.linalg.norm(displacement) if displacement is not None else None,
                                     "evaluation_id": result["id"] if result is not None else None,
                                     "perturbed_id": perturbed["id"], "jacobian_evaluation_ids": ids})
            print(json.dumps(clean({"recovery_seed_complete": seed, "amplitude": amplitude, "timing": ev.timing()})), flush=True)
    aggregates = []
    for regularization in settings["regularizations"]:
        for gain in settings["gains"]:
            for method in ("none", "jacobian", "random"):
                subset = [r for r in rows if r["regularization"] == regularization and r["gain"] == gain and r["method"] == method]
                values = [r["improvement"] for r in subset if r["improvement"] is not None]
                aggregates.append({"regularization": regularization, "gain": gain, "method": method,
                                   "total": len(subset), "valid": len(values),
                                   "median_improvement": float(np.median(values)) if values else None,
                                   "improved_fraction_of_all": sum(v > 0 for v in values) / len(subset)})
    result = {"trials": rows, "aggregates": aggregates}
    save_json(output / "recovery_summary.json", result)
    return result


def run_source(ev, config, original, baseline, scale, epsilon, output):
    settings = config["source_correction"]
    point, actual, trajectory = np.zeros(12), baseline.copy(), []
    for iteration in range(settings["max_iterations"]):
        before = error_norm(original, actual, scale)
        if before is None or before <= settings["relative_error_tolerance"]:
            break
        matrix, _, _, _, ids = estimate(ev, point, epsilon, f"source_jacobian_iteration{iteration}")
        if not np.isfinite(matrix).all():
            trajectory.append({"iteration": iteration, "before": before, "status": "invalid_jacobian"})
            break
        candidates = []
        for regularization in settings["regularizations"]:
            for gain in settings["gains"]:
                step = damped_correction(matrix / scale[:, None], (original - actual) / scale,
                                         regularization, gain, settings["max_step_norm_sd"])
                candidate = bound_excursion(point + step, settings["max_excursion_norm_sd"])
                result = ev.evaluate(candidate, f"source_it{iteration}_l{regularization}_g{gain}")
                after = error_norm(original, result["a"], scale)
                candidates.append({"regularization": regularization, "gain": gain, "point_sd": candidate,
                                   "after": after, "result": result})
        valid = [c for c in candidates if c["after"] is not None]
        best = min(valid, key=lambda c: c["after"]) if valid else None
        row = {"iteration": iteration, "before": before, "jacobian_evaluation_ids": ids,
               "candidates": [{k: v for k, v in c.items() if k != "result"} | {"evaluation_id": c["result"]["id"]} for c in candidates],
               "status": "accepted" if best is not None and best["after"] < before - 1e-9 else "no_improvement"}
        if row["status"] == "accepted":
            point, actual = best["point_sd"], best["result"]["a"]
            row.update({"after": best["after"], "point_sd": point, "actual_hz": actual,
                        "evaluation_id": best["result"]["id"], "regularization": best["regularization"], "gain": best["gain"]})
        trajectory.append(row)
        print(json.dumps(clean({"source_iteration": iteration, "status": row["status"],
                               "before": before, "after": row.get("after"), "timing": ev.timing()})), flush=True)
        if row["status"] != "accepted":
            break
    before, after = error_norm(original, baseline, scale), error_norm(original, actual, scale)
    result = {"original_target_hz": original, "baseline_hz": baseline, "final_hz": actual,
              "before": before, "after": after, "improvement": (before - after) / before if before else None,
              "final_delta_sd": point, "final_excursion_norm_sd": np.linalg.norm(point), "trajectory": trajectory}
    save_json(output / "source_summary.json", result)
    return result


def main(argv=None):
    start = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/feedback_control_i.yaml"))
    parser.add_argument("--phase", choices=["jacobian", "recovery", "source"], required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label", choices=["a", "i", "u"], help="Override configured vowel")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if args.label:
        config["label"] = args.label
    config1 = yaml.safe_load(Path(config["experiment1_config"]).read_text())
    if config["primary_features"] != ["F1", "F2"]:
        parser.error("This initial study uses F1/F2 as its two primary features")
    root = args.output or Path(config["output"].format(label=config["label"]))
    config["output"] = str(root)
    output = root / args.phase
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output exists: {output}; use a new --output root")
    prior = None
    if args.phase != "jacobian":
        prior = json.loads((root / "jacobian" / "jacobian_summary.json").read_text())
        if prior["selected_epsilon_sd"] is None:
            parser.error("No Jacobian passed the predefined validation gate")
    item = next(i for i in config1["inputs"] if i["label"] == config["label"])
    if config["window"].get("center_s") is None:
        config["window"]["center_s"] = (item["start_s"] + item["end_s"]) / 2
    previous = Path(config["baseline_results"])
    reference = []
    for label in config["reference_labels"]:
        interval = next(i for i in config1["inputs"] if i["label"] == label)
        with np.load(previous / label / "baseline_code.npz", allow_pickle=False) as data:
            times = np.arange(len(data["ema"])) / 50
            reference.append(data["ema"][(times >= interval["start_s"]) & (times < interval["end_s"])])
    sd = np.std(np.concatenate(reference).astype(float), axis=0)
    if np.any(sd <= 1e-6):
        parser.error("Reference EMA SD is degenerate")
    with np.load(previous / config["label"] / "baseline_code.npz", allow_pickle=False) as data:
        code = dict(data)
    window = gaussian_window(len(code["ema"]), 50, **config["window"])
    device = config["device"]
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config["device"] = device
    torch.set_num_threads(config1["torch_threads"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    from sparc import load_model
    begin = perf_counter()
    coder = load_model(config1["model"], device=device, use_penn=False)
    synchronize(device)
    model_load_s = perf_counter() - begin
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    save_json(output / "environment.json", environment(device))
    save_json(output / "provenance.json", {"baseline_results": str(previous),
        "baseline_code_sha256": sha256(previous / config["label"] / "baseline_code.npz"),
        "config_sha256": sha256(args.config), "experiment1_config_sha256": sha256(config["experiment1_config"]),
        "reference_sd": sd, "reference_frame_count": sum(len(r) for r in reference),
        "reference_code_sha256": {v: sha256(previous / v / "baseline_code.npz") for v in config["reference_labels"]}})
    np.savez_compressed(output / "baseline_code.npz", **code)
    np.savez_compressed(output / "coordinates.npz", reference_sd=sd, window=window)
    ev = Evaluator(coder, code, sd, window, item, config1["formants"], config, output, device)
    baseline = ev.evaluate(np.zeros(12), "baseline")
    ev.baseline = baseline
    repeat = ev.evaluate(np.zeros(12), "baseline_repeat", force=True)
    if not np.isfinite(baseline["a"]).all():
        raise RuntimeError("Baseline F1/F2 unavailable")
    if baseline["wav_sha256"] != repeat["wav_sha256"]:
        raise RuntimeError("Baseline decode is not reproducible")
    scale = baseline["a"].copy()
    save_json(output / "baseline.json", {"primary_hz": baseline["a"], "observed_hz": baseline["observed"],
              "normalization_hz": scale, "repeat_exact": True, "sample_id": baseline["id"]})
    if args.phase == "jacobian":
        run_jacobian(ev, config, baseline["a"], scale, output)
    elif args.phase == "recovery":
        run_recovery(ev, config, baseline["a"], scale, prior["selected_epsilon_sd"], output)
    else:
        original, sr = sf.read(previous / config["label"] / "original.wav")
        measured = ev.extract(original, sr, item["start_s"], item["end_s"], config1["formants"])
        original_target, _ = primary_from(measured, config["primary_features"])
        if not np.isfinite(original_target).all():
            raise RuntimeError("Original F1/F2 target unavailable")
        run_source(ev, config, original_target, baseline["a"], scale, prior["selected_epsilon_sd"], output)
    timing = ev.timing() | {"model_load_s": model_load_s, "total_wall_s": perf_counter() - start}
    save_json(output / "timing.json", timing)
    ev.log.close()
    print(json.dumps(clean({"phase_complete": args.phase, "timing": timing})), flush=True)


if __name__ == "__main__":
    main()
