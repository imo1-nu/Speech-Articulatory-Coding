"""Report 40 ms trajectory coding at 25 versus 50 tokens per second."""

import argparse
import csv
import html
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch

from .compare_jsut_audio import audio_player
from .model_jsut_tokens import checkpoint_hash, configure_plots, write_json
from .sweep_jsut_windows import evaluate_run


def validate_runs(reference_run, candidate_run):
    configs = [json.loads((run / "config.json").read_text()) for run in (reference_run, candidate_run)]
    splits = [json.loads((run / "splits.json").read_text()) for run in (reference_run, candidate_run)]
    if splits[0] != splits[1]:
        raise ValueError("The two runs must use exactly the same utterance splits")
    common = ("files", "seed", "codebook_size", "latent_dim", "batch_size", "learning_rate",
              "commitment", "max_epochs", "patience", "window_frames", "window_left_frames",
              "window_right_frames", "channels")
    if any(configs[0].get(key) != configs[1].get(key) for key in common):
        raise ValueError("The two runs must use the same features and training settings")
    if (configs[0]["window_frames"] != 2 or configs[0].get("token_rate", 50) != 50
            or configs[1].get("token_rate", 50) != 25 or configs[1].get("token_stride", 1) != 2):
        raise ValueError("Expected 40 ms windows, a 50 Hz reference, and a 25 Hz candidate")
    with np.load(reference_run / "normalization.npz", allow_pickle=False) as reference:
        with np.load(candidate_run / "normalization.npz", allow_pickle=False) as candidate:
            if any(not np.array_equal(reference[key], candidate[key]) for key in ("mean", "std")):
                raise ValueError("Training normalization differs between the two runs")
    return configs[0], splits[0]


def build_report(reference_run, candidate_run, output_dir, audio_limit=5):
    configure_plots()
    config, splits = validate_runs(reference_run, candidate_run)
    runs = {50: reference_run, 25: candidate_run}
    hashes = {rate: checkpoint_hash(run / "best.pt") for rate, run in runs.items()}
    rows = []
    for rate, run in runs.items():
        result = json.loads((run / "window_evaluation.json").read_text())
        if result["checkpoint_sha256"] != hashes[rate]:
            raise ValueError(f"Stale evaluation: {run}")
        validation = result["splits"]["validation"]["uniform"]
        test = result["splits"]["test"]["uniform"]
        rows.append({"token_rate_hz": rate, "window_ms": 40,
                     "nominal_token_id_bps": rate * math.ceil(math.log2(config["codebook_size"])),
                     "validation_mse": validation["normalized_mse"], "test_mse": test["normalized_mse"],
                     "test_delta_mse": test["delta_mse"], "test_velocity_energy_ratio": test["velocity_energy_ratio"],
                     "test_6_to_10_hz_energy_ratio": test["trajectory_6_to_10_hz_energy_ratio"],
                     "test_tokens": test.get("tokens", sum(config["files"][p]["frames"] for p in splits["test"])),
                     "test_used_codes": test["used_codes"], "test_code_perplexity": test["code_perplexity"]})
    manifests = {rate: json.loads((run / "audio_comparison/manifest.json").read_text())
                 for rate, run in runs.items()}
    if any(manifests[rate]["checkpoint_sha256"] != hashes[rate] for rate in runs):
        raise ValueError("Audio comparison is from a different checkpoint")
    expected = splits["test"][:audio_limit]
    records = {rate: {record["feature_path"]: record for record in manifest["records"]}
               for rate, manifest in manifests.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    cards, comparison_records = [], []
    for relative in expected:
        if any(relative not in records[rate] for rate in runs):
            raise ValueError(f"Missing comparison audio for {relative}")
        pair = {rate: records[rate][relative] for rate in runs}
        conditions = {}
        for rate in runs:
            folder = runs[rate] / "audio_comparison" / Path(relative).with_suffix("")
            with np.load(folder / "conditions.npz", allow_pickle=False) as stored:
                conditions[rate] = dict(stored)
        for key in ("pitch", "loudness", "spk_emb", "ema", "time"):
            if not np.allclose(conditions[50][key], conditions[25][key], rtol=1e-5, atol=1e-6):
                raise ValueError(f"Synthesis conditions differ ({key}): {relative}")
        directory = output_dir / "listening" / Path(relative).with_suffix("")
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference_run / "audio_comparison" / pair[50]["audio"]["original"],
                        directory / "original.wav")
        common_gain = min(pair[rate]["synthesis_common_gain"] for rate in runs)
        waveforms = {}
        for name, rate, key in (("original_ema", 50, "original_ema"),
                                ("tokens_50hz", 50, "reconstructed_ema"),
                                ("tokens_25hz", 25, "reconstructed_ema")):
            wave, sr = sf.read(runs[rate] / "audio_comparison" / pair[rate]["audio"][key])
            if sr != 16000 or not np.isfinite(wave).all():
                raise ValueError(f"Invalid synthesis: {relative}")
            wave = wave * common_gain / pair[rate]["synthesis_common_gain"]
            waveforms[name] = wave
            sf.write(directory / f"{name}.wav", wave, sr, subtype="PCM_16")
        if len({len(wave) for wave in waveforms.values()}) != 1:
            raise ValueError(f"Synthesis lengths differ: {relative}")
        ab = np.concatenate((waveforms["tokens_50hz"], np.zeros(8000), waveforms["tokens_25hz"]))
        sf.write(directory / "ab_50_then_25.wav", ab, 16000, subtype="PCM_16")
        files = {key: (directory / f"{key}.wav").relative_to(output_dir).as_posix()
                 for key in ("original", "original_ema", "tokens_50hz", "tokens_25hz", "ab_50_then_25")}
        cells = ''.join(f'<div><h3>{label}</h3>{audio_player(files[key])}</div>' for key, label in (
            ("original", "元音声"), ("original_ema", "抽出軌道から合成"),
            ("tokens_50hz", "50 Hzトークン"), ("tokens_25hz", "25 Hzトークン")))
        cards.append(f'<section><h2>{html.escape(pair[50]["utterance_id"])}</h2>'
                     f'<p>{html.escape(pair[50]["text"])}</p><div class="players">{cells}</div>'
                     '<p>A/B比較：50 Hz → 25 Hz（間に0.5秒の無音）</p>'
                     + audio_player(files["ab_50_then_25"]) + '</section>')
        comparison_records.append({"feature_path": relative, "files": files, "common_gain": common_gain,
                                   "samples": len(waveforms["tokens_25hz"])})

        if relative == expected[0]:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(6, 2, figsize=(12, 12), sharex=True)
            time = conditions[50]["time"]
            for channel, axis in enumerate(axes.flat):
                axis.plot(time, conditions[50]["ema"][:, channel], label="Original EMA", linewidth=1)
                axis.plot(time, conditions[50]["reconstruction"][:, channel], label="50 Hz tokens", linewidth=1)
                axis.plot(time, conditions[25]["reconstruction"][:, channel], label="25 Hz tokens", linewidth=1)
                axis.set_ylabel(config["channels"][channel])
            axes[0, 0].legend(fontsize=8)
            axes[-1, 0].set_xlabel("Time (s)")
            axes[-1, 1].set_xlabel("Time (s)")
            fig.tight_layout()
            fig.savefig(output_dir / "trajectory_comparison.png", dpi=140)
            plt.close(fig)

    summary = {"window_ms": 40, "rows": rows, "checkpoint_sha256": hashes,
               "same_utterance_splits_and_normalization": True,
               "note": "Bitrate is fixed-width token IDs only; pitch/loudness/speaker/model overhead excluded.",
               "comparison_records": comparison_records}
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = ''.join(f'<tr><td>{r["token_rate_hz"]}</td><td>{r["nominal_token_id_bps"]}</td>'
                    f'<td>{r["validation_mse"]:.5f}</td><td>{r["test_mse"]:.5f}</td>'
                    f'<td>{r["test_velocity_energy_ratio"]:.3f}</td><td>{r["test_tokens"]}</td></tr>' for r in rows)
    content = '''<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>40 ms・25 Hz対50 Hz</title>
<style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#182431}
section{border:1px solid #ddd;border-radius:12px;margin:24px 0;padding:18px}
.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}
audio{width:100%;max-width:360px}p{line-height:1.7}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}
table{border-collapse:collapse;width:100%}img{max-width:100%}</style><h1>40 ms窓：25 Hzと50 Hzの比較</h1>
<p>同じ5,000発話・分割・正規化・128コード・潜在次元16・学習設定を使用。
25 Hzモデルは40 msごとに1トークンを出し、1トークンから20 ms刻みの軌道を2フレーム復元します。
窓は重ならず、発話末尾の補完フレームは評価から除外します。50 Hzモデルは20 msごとに
トークンを出し、重なった予測を平均します。両モデルの出力軌道は50 Hzです。</p>
<p>合成では同じピッチ・音量・話者条件を使い、比較音声のゲインも揃えています。
下記bpsはトークンIDの固定長換算だけで、ピッチ・話者・モデル等の情報量は含みません。</p>
<table><tr><th>トークンHz</th><th>ID換算bps</th><th>検証MSE</th><th>テストMSE</th>
<th>テスト変化量比</th><th>テストトークン数</th></tr>'''
    content += (table + '</table><p><a href="trajectory_comparison.png">先頭発話の12次元軌道比較図</a></p>'
                + ''.join(cards))
    content += '''<script>document.querySelectorAll('audio').forEach(p=>p.addEventListener('play',()=>{
document.querySelectorAll('audio').forEach(q=>{if(q!==p)q.pause();});}));</script></html>'''
    (output_dir / "index.html").write_text(content, encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, default=Path("outputs/jsut_window_sweep/window_040ms"))
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/jsut_40ms_25hz_comparison/model_25hz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/jsut_40ms_25hz_comparison"))
    parser.add_argument("--jsut-dir", type=Path, default=Path("datasets/jsut/jsut_ver1.1"))
    parser.add_argument("--audio-limit", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.audio_limit < 1 or args.threads < 1:
        parser.error("Audio limit and thread count must be positive")
    torch.set_num_threads(args.threads)
    try:
        validate_runs(args.reference_run, args.run_dir)
        evaluate_run(args.run_dir)
        audio_dir = args.run_dir / "audio_comparison"
        if not (audio_dir / "manifest.json").is_file():
            subprocess.run([sys.executable, "-m", "sparc.compare_jsut_audio", "--run-dir", str(args.run_dir),
                            "--jsut-dir", str(args.jsut_dir), "--limit", str(args.audio_limit),
                            "--device", args.device, "--threads", str(args.threads)], check=True)
        summary = build_report(args.reference_run, args.run_dir, args.output_dir, args.audio_limit)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    for row in summary["rows"]:
        print(f"{row['token_rate_hz']} Hz: test MSE={row['test_mse']:.6f}, "
              f"velocity ratio={row['test_velocity_energy_ratio']:.3f}")
    print(f"Report: {args.output_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
