"""Compare context windows and uniform/triangular reconstruction on fixed JSUT splits."""

import argparse
import csv
import html
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from tqdm import tqdm

from .compare_jsut_audio import audio_player
from .model_jsut_tokens import (
    checked_feature, checkpoint_hash, configure_plots, infer_utterance, load_run,
    save_npz, write_json,
)
from .trajectory_vq import overlap_average, split_utterances, triangular_weights, usage_statistics


def evaluate_run(run_dir, batch_size=512):
    """Evaluate both aggregations at 50 Hz, preserving the model's token rate."""
    config, splits, model, mean, std = load_run(run_dir, torch.device("cpu"))
    width = model.window_frames
    weights = triangular_weights(width)
    digest = checkpoint_hash(run_dir / "best.pt")
    size = config["codebook_size"]
    with torch.inference_mode():
        code_windows = model.decoder(model.codebook.weight).numpy().reshape(size, width, 12)
    metrics = {}
    for split in ("validation", "test"):
        totals = {name: {"sse": 0.0, "delta_sse": 0.0, "velocity": 0.0, "band": 0.0}
                  for name in ("uniform", "triangular")}
        source_velocity, source_band, dimensions, delta_dimensions = 0.0, 0.0, 0, 0
        counts = np.zeros(size, dtype=np.int64)
        for relative in tqdm(splits[split], desc=f"Evaluate {width * 20}ms {split}", unit="utterance"):
            data = checked_feature(config, relative)
            original = data["ema"]
            tokens, uniform, _ = infer_utterance(model, original, mean, std, batch_size, torch.device("cpu"))
            weighted = (uniform.copy() if np.all(weights == weights[0])
                        else overlap_average(code_windows[tokens], weights, stride=model.token_stride,
                                             output_length=len(original)) * std + mean)
            auxiliary = {key: data[key] for key in ("pitch", "periodicity", "loudness", "text", "source")
                         if key in data}
            save_npz(run_dir / "tokens" / relative, tokens=tokens.astype(np.int64),
                     reconstruction=uniform, weighted_reconstruction=weighted, overlap_weights=weights,
                     time=data["time"], token_time=data["time"][::model.token_stride],
                     frame_rate=np.float64(50), token_rate=np.float64(50 / model.token_stride),
                     token_stride=np.int64(model.token_stride), token_len=np.int64(len(tokens)),
                     ft_len=np.int64(len(original)),
                     channels=data["channels"], split=np.str_(split), feature_path=np.str_(relative),
                     checkpoint_sha256=np.str_(digest), **auxiliary)
            counts += np.bincount(tokens, minlength=size)
            dimensions += original.size
            delta_dimensions += max(0, len(original) - 1) * 12
            source = (original.astype(float) - mean) / std
            source_delta = np.diff(source, axis=0)
            source_velocity += float((source_delta ** 2).sum())
            frequency = np.fft.rfftfreq(len(original), 1 / 50)
            band_mask = (frequency > 6) & (frequency <= 10)

            def band_energy(values):
                spectrum = np.abs(np.fft.rfft(values - values.mean(axis=0), axis=0)) ** 2 / len(values)
                return float(spectrum[band_mask].sum())

            source_band += band_energy(source)
            for name, values in (("uniform", uniform), ("triangular", weighted)):
                standardized = (values.astype(float) - mean) / std
                delta = np.diff(standardized, axis=0)
                totals[name]["sse"] += float(((standardized - source) ** 2).sum())
                totals[name]["delta_sse"] += float(((delta - source_delta) ** 2).sum())
                totals[name]["velocity"] += float((delta ** 2).sum())
                totals[name]["band"] += band_energy(standardized)
        metrics[split] = {}
        for name, values in totals.items():
            metrics[split][name] = {
                "normalized_mse": values["sse"] / dimensions,
                "delta_mse": values["delta_sse"] / delta_dimensions if delta_dimensions else None,
                "velocity_energy_ratio": values["velocity"] / source_velocity if source_velocity else None,
                "trajectory_6_to_10_hz_energy_ratio": values["band"] / source_band if source_band else None,
                "frames": dimensions // 12, "tokens": int(counts.sum()), **usage_statistics(counts),
            }
    result = {"window_ms": width * 20, "window_frames": width, "weights": weights.tolist(),
              "checkpoint_sha256": digest, "token_rate": 50 / model.token_stride, "splits": metrics}
    write_json(run_dir / "window_evaluation.json", result)
    return result


def write_summary(output_dir, results, expected, codebook_size=128):
    configure_plots()
    import matplotlib.pyplot as plt

    rows = []
    for result in results:
        for name in ("uniform", "triangular"):
            validation, test = result["splits"]["validation"][name], result["splits"]["test"][name]
            rows.append({"window_ms": result["window_ms"], "codebook_size": codebook_size, "weighting": name,
                         "validation_mse": validation["normalized_mse"],
                         "test_mse": test["normalized_mse"], "test_delta_mse": test["delta_mse"],
                         "test_velocity_energy_ratio": test["velocity_energy_ratio"],
                         "test_6_to_10_hz_energy_ratio": test["trajectory_6_to_10_hz_energy_ratio"],
                         "test_used_codes": test["used_codes"], "test_code_perplexity": test["code_perplexity"]})
    best = min(rows, key=lambda row: row["validation_mse"])
    write_json(output_dir / "summary.json", {"expected_windows_ms": expected,
               "completed_windows_ms": [result["window_ms"] for result in results],
               "selection_metric": "validation normalized frame MSE", "best_by_validation": best,
               "rows": rows})
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for weighting, label in (("uniform", "Uniform"), ("triangular", "Center-weighted")):
        group = [row for row in rows if row["weighting"] == weighting]
        axes[0].plot([row["window_ms"] for row in group],
                     [row["validation_mse"] for row in group], marker="o", label=label)
        axes[1].plot([row["window_ms"] for row in group],
                     [row["test_velocity_energy_ratio"] for row in group], marker="o", label=label)
    axes[0].set(xlabel="Window (ms)", ylabel="Validation normalized frame MSE")
    axes[1].set(xlabel="Window (ms)", ylabel="Test velocity energy / original")
    axes[1].axhline(1, color="gray", linestyle="--", linewidth=1)
    for axis in axes:
        axis.set_xticks(expected)
        axis.legend()
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "window_comparison.png", dpi=160)
    plt.close(fig)

    labels = {"uniform": "なし（等重み）", "triangular": "あり（中央重み）"}
    table = ''.join(f'<tr><td>{row["window_ms"]}</td><td>{labels[row["weighting"]]}</td>'
                    f'<td>{row["validation_mse"]:.5f}</td><td>{row["test_mse"]:.5f}</td>'
                    f'<td>{row["test_velocity_energy_ratio"]:.3f}</td></tr>' for row in rows)
    manifests = {}
    for result in results:
        folder = output_dir / f'window_{result["window_ms"]:03d}ms' / "audio_comparison"
        if (folder / "manifest.json").is_file():
            manifests[result["window_ms"]] = json.loads((folder / "manifest.json").read_text())
    cards = []
    if manifests:
        first_ms = min(manifests)
        for record in manifests[first_ms]["records"]:
            prefix = f"window_{first_ms:03d}ms/audio_comparison/"
            header = f'<h2>{html.escape(record["utterance_id"])}</h2><p>{html.escape(record["text"])}</p>'
            header += '<p>元音声</p>' + audio_player(prefix + record["audio"]["original"])
            header += '<p>抽出軌道からの合成</p>' + audio_player(prefix + record["audio"]["original_ema"])
            audio_rows = []
            for ms, manifest in sorted(manifests.items()):
                candidate = next(item for item in manifest["records"]
                                 if item["feature_path"] == record["feature_path"])
                prefix = f"window_{ms:03d}ms/audio_comparison/"
                players = [audio_player(prefix + candidate["audio"][key])
                           for key in ("reconstructed_ema", "weighted_ema", "weighted_ab")]
                audio_rows.append(f'<tr><td>{ms} ms</td>' + ''.join(f'<td>{p}</td>' for p in players) + '</tr>')
            cards.append('<section>' + header + '<div class="scroll"><table><tr><th>窓</th>'
                         '<th>重みなし</th><th>中央重みあり</th><th>A/B：なし → あり</th></tr>'
                         + ''.join(audio_rows) + '</table></div></section>')
    report = '''<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>JSUT 時間窓比較</title>
<style>body{font-family:system-ui,sans-serif;max-width:1300px;margin:32px auto;padding:0 16px;color:#182431}
table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}
audio{width:260px;max-width:100%}.scroll{overflow-x:auto}section{margin:32px 0;padding:18px;border:1px solid #ddd;border-radius:12px}
img{max-width:100%}p{line-height:1.7}</style><h1>JSUT 時間窓・復号重みの比較</h1>'''
    report += (f'<p>完了：{len(results)} / {len(expected)} 窓。50 Hz・{codebook_size}コード・同じ発話分割で比較。'
               '重みは左右対称の三角形（100 msなら1・2・3・2・1）。20・40 msでは重みあり／なしが同じになります。'
               '偶数窓では後方を1フレーム多く参照し、各フレームの時刻は維持します。</p>'
               '<p>各窓で1モデルを学習し、同じトークン列を2通りの重みで復号します。'
               '設定選択は検証集合のフレームMSEに基づきます。聴感と動きの保持も併せて確認してください。</p>'
               '<img src="window_comparison.png" alt="時間窓別の検証誤差と動きの保持">'
               '<table><tr><th>窓(ms)</th><th>重み</th><th>検証MSE</th><th>テストMSE</th>'
               '<th>テスト変化量比</th></tr>' + table + '</table>' + ''.join(cards))
    report += '''<script>document.querySelectorAll('audio').forEach(p=>p.addEventListener('play',()=>{
document.querySelectorAll('audio').forEach(q=>{if(q!==p)q.pause();});}));</script></html>'''
    (output_dir / "index.html").write_text(report, encoding="utf-8")


def run_command(arguments, log_path):
    with log_path.open("w", encoding="utf-8") as stream:
        subprocess.run([sys.executable, *arguments], stdout=stream, stderr=subprocess.STDOUT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("outputs/jsut_ema"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/jsut_window_sweep"))
    parser.add_argument("--jsut-dir", type=Path, default=Path("datasets/jsut/jsut_ver1.1"))
    parser.add_argument("--windows-ms", nargs="+", type=int, default=list(range(20, 181, 20)))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--codebook-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--audio-limit", type=int, default=5)
    parser.add_argument("--skip-audio", action="store_true")
    args = parser.parse_args(argv)
    windows = sorted(set(args.windows_ms))
    if any(value < 20 or value % 20 for value in windows):
        parser.error("Each window must be a positive multiple of 20 ms")
    if any(value < 1 for value in (args.epochs, args.patience, args.batch_size, args.threads,
                                   args.audio_limit, args.codebook_size)):
        parser.error("Epoch, patience, batch size, threads and audio limit must be positive")
    if args.limit is not None and args.limit < 3:
        parser.error("--limit must be at least 3")
    torch.set_num_threads(args.threads)
    expected_paths = [path.relative_to(args.features_dir).as_posix()
                      for path in sorted((args.features_dir / "basic5000").glob("*.npz"))]
    if args.limit is not None:
        expected_paths = expected_paths[:args.limit]
    expected_splits = split_utterances(expected_paths)
    experiment = {"features_dir": str(args.features_dir.resolve()), "windows_ms": windows,
                  "epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size,
                  "seed": 42, "codebook_size": args.codebook_size, "latent_dim": 16, "limit": args.limit,
                  "audio_limit": args.audio_limit, "skip_audio": args.skip_audio}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment_path = args.output_dir / "experiment.json"
    if experiment_path.exists() and json.loads(experiment_path.read_text()) != experiment:
        parser.error("Existing sweep has different settings; choose a new --output-dir")
    write_json(experiment_path, experiment)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    results = []
    for ms in windows:
        run_dir = args.output_dir / f"window_{ms:03d}ms"
        print(f"Window {ms} ms: train/evaluate both weightings", flush=True)
        if not (run_dir / "training_metrics.json").exists():
            if run_dir.exists() and any(run_dir.iterdir()):
                parser.error(f"Incomplete training run: {run_dir}; move it aside before restarting")
            arguments = ["-m", "sparc.model_jsut_tokens", "train", "--features-dir", str(args.features_dir),
                         "--run-dir", str(run_dir), "--window-ms", str(ms), "--cache-windows",
                         "--epochs", str(args.epochs), "--patience", str(args.patience),
                         "--batch-size", str(args.batch_size), "--threads", str(args.threads),
                         "--codebook-size", str(args.codebook_size),
                         "--device", args.device]
            if args.limit is not None:
                arguments += ["--limit", str(args.limit)]
            run_command(arguments, log_dir / f"window_{ms:03d}ms_train.log")
        config = json.loads((run_dir / "config.json").read_text())
        splits = json.loads((run_dir / "splits.json").read_text())
        if (splits != expected_splits or config["window_frames"] != ms // 20
                or config.get("token_rate", 50) != 50
                or config["codebook_size"] != args.codebook_size or config["seed"] != 42
                or config["max_epochs"] != args.epochs or config["patience"] != args.patience
                or config["batch_size"] != args.batch_size
                or Path(config["features_dir"]).resolve() != args.features_dir.resolve()):
            parser.error(f"Existing training run does not match sweep settings: {run_dir}")
        result = evaluate_run(run_dir, args.batch_size)
        if not args.skip_audio:
            audio_dir = run_dir / "audio_comparison"
            if not (audio_dir / "manifest.json").exists():
                run_command(["-m", "sparc.compare_jsut_audio", "--run-dir", str(run_dir),
                             "--jsut-dir", str(args.jsut_dir), "--include-weighted",
                             "--output-dir", str(audio_dir), "--limit", str(args.audio_limit),
                             "--threads", str(args.threads), "--device", args.device],
                            log_dir / f"window_{ms:03d}ms_audio.log")
            manifest = json.loads((audio_dir / "manifest.json").read_text())
            if (manifest["checkpoint_sha256"] != result["checkpoint_sha256"]
                    or len(manifest["records"]) != min(args.audio_limit, len(splits["test"]))):
                parser.error(f"Incomplete or stale comparison: {audio_dir}")
        results.append(result)
        write_summary(args.output_dir, results, windows, args.codebook_size)
        print(f"Window {ms} ms complete; report: {args.output_dir / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
