"""Compare codebook sizes for 20/40 ms windows at 50 tokens per second."""

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import soundfile as sf
import torch

from .compare_jsut_audio import audio_player
from .model_jsut_tokens import checkpoint_hash, configure_plots, write_json
from .sweep_jsut_windows import run_command
from .trajectory_vq import split_utterances


def case_key(window_ms, codebook_size):
    return f"w{window_ms:03d}_k{codebook_size:04d}"


def validate_case(run_dir, window_ms, size, features_dir, splits, epochs, patience, batch_size):
    config = json.loads((run_dir / "config.json").read_text())
    actual_splits = json.loads((run_dir / "splits.json").read_text())
    expected = {"codebook_size": size, "window_frames": window_ms // 20,
                "latent_dim": 16, "seed": 42, "max_epochs": epochs, "patience": patience,
                "batch_size": batch_size, "learning_rate": 0.001, "commitment": 0.25}
    if (any(config.get(key) != value for key, value in expected.items())
            or actual_splits != splits or config.get("token_rate", 50) != 50
            or config.get("token_stride", 1) != 1
            or Path(config["features_dir"]).resolve() != features_dir.resolve()):
        raise ValueError(f"Training settings or data splits differ: {run_dir}")
    result = json.loads((run_dir / "window_evaluation.json").read_text())
    digest = checkpoint_hash(run_dir / "best.pt")
    if (result["checkpoint_sha256"] != digest or result["window_ms"] != window_ms
            or result.get("token_rate", 50) != 50):
        raise ValueError(f"Stale or incompatible evaluation: {run_dir}")
    metrics = result["splits"]["test"]["uniform"]
    if metrics["total_codes"] != size:
        raise ValueError(f"Evaluation has the wrong codebook size: {run_dir}")
    return config, result


def assemble_audio(output_dir, cases, splits, audio_limit, windows, codebooks):
    manifests = {}
    for case in cases:
        run = Path(case["run_dir"])
        manifest = json.loads((run / "audio_comparison/manifest.json").read_text())
        if manifest["checkpoint_sha256"] != case["checkpoint_sha256"]:
            raise ValueError(f"Stale audio: {run}")
        manifests[(case["window_ms"], case["codebook_size"])] = (run, {
            item["feature_path"]: item for item in manifest["records"]})
    cards, comparisons = [], []
    for relative in splits["test"][:audio_limit]:
        selected = {}
        common_conditions = None
        for key, (run, records) in manifests.items():
            if relative not in records:
                raise ValueError(f"Missing test audio: {run}: {relative}")
            record = records[relative]
            folder = run / "audio_comparison" / Path(relative).with_suffix("")
            with np.load(folder / "conditions.npz", allow_pickle=False) as data:
                conditions = {name: data[name] for name in ("pitch", "loudness", "spk_emb", "ema", "time")}
            if common_conditions is None:
                common_conditions = conditions
            if any(not np.allclose(conditions[name], common_conditions[name], rtol=1e-5, atol=1e-6)
                   for name in conditions):
                raise ValueError(f"Synthesis conditions differ: {relative}")
            selected[key] = (run, record)
        gain = min(record["synthesis_common_gain"] for _, record in selected.values())
        first_run, first_record = next(iter(selected.values()))
        directory = output_dir / "listening" / Path(relative).with_suffix("")
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(first_run / "audio_comparison" / first_record["audio"]["original"],
                        directory / "original.wav")
        baseline, sr = sf.read(first_run / "audio_comparison" / first_record["audio"]["original_ema"])
        baseline *= gain / first_record["synthesis_common_gain"]
        sf.write(directory / "original_ema.wav", baseline, sr, subtype="PCM_16")
        waveforms, files = {}, {}
        for (ms, size), (run, record) in selected.items():
            wave, sr = sf.read(run / "audio_comparison" / record["audio"]["reconstructed_ema"])
            wave *= gain / record["synthesis_common_gain"]
            if sr != 16000 or len(wave) != len(baseline) or not np.isfinite(wave).all():
                raise ValueError(f"Invalid synthesis: {run}: {relative}")
            name = case_key(ms, size)
            sf.write(directory / f"{name}.wav", wave, sr, subtype="PCM_16")
            files[name] = (directory / f"{name}.wav").relative_to(output_dir).as_posix()
            waveforms[(ms, size)] = wave
        rows = []
        for size in codebooks:
            if not any((ms, size) in selected for ms in windows):
                continue
            players = [audio_player(files[case_key(ms, size)]) if (ms, size) in selected else "未完了"
                       for ms in windows]
            rows.append(f'<tr><th>{size}</th>' + ''.join(f'<td>{player}</td>' for player in players) + '</tr>')
        sequences = []
        for ms in windows:
            sizes = [size for size in codebooks if (ms, size) in waveforms]
            if not sizes:
                continue
            clips = []
            for size in sizes:
                if clips:
                    clips.append(np.zeros(8000))
                clips.append(waveforms[(ms, size)])
            name = f"w{ms:03d}_all_codebooks"
            sf.write(directory / f"{name}.wav", np.concatenate(clips), 16000, subtype="PCM_16")
            files[name] = (directory / f"{name}.wav").relative_to(output_dir).as_posix()
            order = " → ".join(map(str, sizes))
            sequences.append(f'<p>{ms} ms連続比較：{order} コード（間に0.5秒の無音）</p>'
                             + audio_player(files[name]))
        header = ''.join(f'<th>{ms} ms</th>' for ms in windows)
        original = (directory / "original.wav").relative_to(output_dir).as_posix()
        baseline_path = (directory / "original_ema.wav").relative_to(output_dir).as_posix()
        cards.append(f'<section><h2>{html.escape(first_record["utterance_id"])}</h2>'
                     f'<p>{html.escape(first_record["text"])}</p><div class="players">'
                     '<div><p>元音声</p>' + audio_player(original) + '</div><div><p>抽出軌道から合成</p>'
                     + audio_player(baseline_path) + '</div></div><table><tr><th>コード数</th>' + header + '</tr>'
                     + ''.join(rows) + '</table>' + ''.join(sequences) + '</section>')
        comparisons.append({"feature_path": relative, "common_gain": gain,
                            "samples": len(baseline), "original": original,
                            "original_ema": baseline_path, "files": files})
    return cards, comparisons


def write_report(output_dir, cases, splits, windows, codebooks, audio_limit=5, skip_audio=False):
    configure_plots()
    import matplotlib.pyplot as plt

    rows = []
    for case in sorted(cases, key=lambda item: (item["window_ms"], item["codebook_size"])):
        result = case["evaluation"]
        validation, test = result["splits"]["validation"]["uniform"], result["splits"]["test"]["uniform"]
        rows.append({"window_ms": case["window_ms"], "codebook_size": case["codebook_size"],
                     "token_rate": 50, "nominal_token_id_bps": 50 * math.ceil(math.log2(case["codebook_size"])),
                     "validation_mse": validation["normalized_mse"], "test_mse": test["normalized_mse"],
                     "test_delta_mse": test["delta_mse"], "test_velocity_energy_ratio": test["velocity_energy_ratio"],
                     "test_6_to_10_hz_energy_ratio": test["trajectory_6_to_10_hz_energy_ratio"],
                     "test_used_codes": test["used_codes"], "test_code_perplexity": test["code_perplexity"],
                     "run_dir": case["run_dir"], "checkpoint_sha256": case["checkpoint_sha256"]})
    best = min(rows, key=lambda row: row["validation_mse"])
    best_per_window = {str(ms): min([row for row in rows if row["window_ms"] == ms],
                                   key=lambda row: row["validation_mse"])
                       for ms in windows if any(row["window_ms"] == ms for row in rows)}
    cards, comparisons = ([], []) if skip_audio else assemble_audio(
        output_dir, cases, splits, audio_limit, windows, codebooks)
    write_json(output_dir / "summary.json", {
        "expected_cases": len(windows) * len(codebooks), "completed_cases": len(rows),
        "selection_metric": "validation normalized frame MSE", "best_by_validation": best,
        "best_per_window": best_per_window, "rows": rows, "comparison_records": comparisons,
        "note": "Bitrates count fixed-width token IDs only; source features/model overhead excluded."})
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ms in windows:
        group = [row for row in rows if row["window_ms"] == ms]
        axes[0].plot([row["codebook_size"] for row in group], [row["validation_mse"] for row in group],
                     marker="o", label=f"{ms} ms")
        axes[1].plot([row["codebook_size"] for row in group], [row["test_velocity_energy_ratio"] for row in group],
                     marker="o", label=f"{ms} ms")
    axes[0].set(ylabel="Validation normalized frame MSE")
    axes[1].set(ylabel="Test velocity energy / original")
    axes[1].axhline(1, color="gray", linestyle="--", linewidth=1)
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(codebooks, labels=[str(size) for size in codebooks])
        axis.set_xlabel("Codebook size")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "codebook_comparison.png", dpi=160)
    plt.close(fig)
    table = ''.join(f'<tr><td>{row["window_ms"]}</td><td>{row["codebook_size"]}</td>'
                    f'<td>{row["nominal_token_id_bps"]}</td><td>{row["validation_mse"]:.5f}</td>'
                    f'<td>{row["test_mse"]:.5f}</td><td>{row["test_delta_mse"]:.5f}</td>'
                    f'<td>{row["test_velocity_energy_ratio"]:.3f}</td>'
                    f'<td>{row["test_used_codes"]}/{row["codebook_size"]}</td>'
                    f'<td>{row["test_code_perplexity"]:.1f}</td></tr>' for row in rows)
    content = '''<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>JSUT コード数比較</title>
<style>body{font-family:system-ui,sans-serif;max-width:1250px;margin:32px auto;padding:0 18px;color:#182431}
section{border:1px solid #ddd;border-radius:12px;padding:18px;margin:28px 0}
td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse;width:100%}
.scroll{overflow:auto}.players{display:flex;gap:20px;flex-wrap:wrap}audio{width:300px;max-width:100%}
img{max-width:100%}p{line-height:1.7}</style><h1>50 Hz・20/40 ms：コード数比較</h1>'''
    content += (f'<p>完了：{len(rows)}/{len(windows) * len(codebooks)} 条件。'
                '同じ発話分割・正規化・潜在次元16・seed 42・学習設定で、窓長とコード数を比較します。'
                '復号は等重み平均で、この2窓では対称三角形重みと同じ結果です。</p>'
                '<p>最大epoch数・早期終了条件は全候補で共通。最良条件は検証フレームMSEで選びます。'
                'bpsはトークンIDのみの固定長換算です。変化量比は元軌道を1とした値で、大きいほど良い指標ではありません。'
                '比較音声はピッチ・音量・話者条件とゲインを共通にしています。</p>'
                '<img src="codebook_comparison.png" alt="コード数別の再構成誤差と動きの変化量">'
                '<div class="scroll"><table><tr><th>窓(ms)</th><th>コード数</th><th>ID換算bps</th>'
                '<th>検証MSE</th><th>テストMSE</th><th>差分MSE</th><th>変化量比</th>'
                '<th>使用コード</th><th>コードperplexity</th></tr>' + table + '</table></div>' + ''.join(cards))
    content += '''<script>document.querySelectorAll('audio').forEach(p=>p.addEventListener('play',()=>{
document.querySelectorAll('audio').forEach(q=>{if(q!==p)q.pause();});}));</script></html>'''
    (output_dir / "index.html").write_text(content, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("outputs/jsut_ema"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/jsut_codebook_sweep"))
    parser.add_argument("--reference-root", type=Path, default=Path("outputs/jsut_window_sweep"))
    parser.add_argument("--jsut-dir", type=Path, default=Path("datasets/jsut/jsut_ver1.1"))
    parser.add_argument("--windows-ms", nargs="+", type=int, choices=(20, 40), default=[20, 40])
    parser.add_argument("--codebooks", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--audio-limit", type=int, default=5)
    parser.add_argument("--skip-audio", action="store_true")
    args = parser.parse_args(argv)
    windows, codebooks = sorted(set(args.windows_ms)), sorted(set(args.codebooks))
    if any(size < 2 for size in codebooks):
        parser.error("Codebook sizes must be at least 2")
    if any(value < 1 for value in (args.epochs, args.patience, args.batch_size, args.threads, args.audio_limit)):
        parser.error("Training and audio parameters must be positive")
    if args.limit is not None and args.limit < 3:
        parser.error("--limit must be at least 3")
    torch.set_num_threads(args.threads)
    paths = [path.relative_to(args.features_dir).as_posix()
             for path in sorted((args.features_dir / "basic5000").glob("*.npz"))]
    if args.limit is not None:
        paths = paths[:args.limit]
    splits = split_utterances(paths)
    experiment = {"features_dir": str(args.features_dir.resolve()), "windows_ms": windows,
                  "codebooks": codebooks, "epochs": args.epochs, "patience": args.patience,
                  "batch_size": args.batch_size, "seed": 42, "latent_dim": 16,
                  "limit": args.limit, "audio_limit": args.audio_limit, "skip_audio": args.skip_audio,
                  "reference_root": str(args.reference_root.resolve())}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "experiment.json"
    if path.exists() and json.loads(path.read_text()) != experiment:
        parser.error("Existing codebook sweep has different settings; choose a new output directory")
    write_json(path, experiment)
    logs = args.output_dir / "logs"
    logs.mkdir(exist_ok=True)
    cases, reference_config, reference_normalization = [], None, None
    for size in codebooks:
        reference_available = size == 128 and all(
            (args.reference_root / f"window_{ms:03d}ms/window_evaluation.json").exists() for ms in windows)
        group = args.reference_root if reference_available else args.output_dir / f"codebook_{size:04d}"
        ready = all((group / f"window_{ms:03d}ms/training_metrics.json").exists()
                    and (group / f"window_{ms:03d}ms/window_evaluation.json").exists()
                    and (args.skip_audio or (group / f"window_{ms:03d}ms/audio_comparison/manifest.json").exists())
                    for ms in windows)
        if not ready:
            if reference_available:
                parser.error("Reference audio is missing; generate it before reusing the reference")
            arguments = ["-m", "sparc.sweep_jsut_windows", "--features-dir", str(args.features_dir),
                         "--output-dir", str(group), "--windows-ms", *map(str, windows),
                         "--codebook-size", str(size), "--epochs", str(args.epochs),
                         "--patience", str(args.patience), "--batch-size", str(args.batch_size),
                         "--threads", str(args.threads), "--device", args.device,
                         "--jsut-dir", str(args.jsut_dir), "--audio-limit", str(args.audio_limit)]
            if args.limit is not None:
                arguments += ["--limit", str(args.limit)]
            if args.skip_audio:
                arguments.append("--skip-audio")
            print(f"Training codebook {size}...", flush=True)
            run_command(arguments, logs / f"codebook_{size:04d}.log")
        for ms in windows:
            run = group / f"window_{ms:03d}ms"
            config, result = validate_case(run, ms, size, args.features_dir, splits,
                                           args.epochs, args.patience, args.batch_size)
            with np.load(run / "normalization.npz", allow_pickle=False) as data:
                norm = {key: data[key] for key in ("mean", "std")}
            if reference_config is None:
                reference_config, reference_normalization = config, norm
            if (config["files"] != reference_config["files"]
                    or any(not np.array_equal(norm[key], reference_normalization[key]) for key in norm)):
                parser.error(f"Feature files or normalization differ: {run}")
            cases.append({"window_ms": ms, "codebook_size": size, "run_dir": str(run.resolve()),
                          "checkpoint_sha256": result["checkpoint_sha256"], "evaluation": result})
        write_report(args.output_dir, cases, splits, windows, codebooks, args.audio_limit, args.skip_audio)
        print(f"Codebook {size} complete; {len(cases)}/{len(windows) * len(codebooks)} conditions", flush=True)
    print(f"Report: {args.output_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
