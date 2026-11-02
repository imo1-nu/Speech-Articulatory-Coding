"""Generate and compare Japanese PPGs for original, EMA-synthesized and token-synthesized JSUT audio."""

import argparse
import csv
import html
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from tqdm import tqdm

from .compare_jsut_audio import audio_player
from .japanese_ppg import (
    FFT_SIZE, FRAME_RATE, HOP, MAX_FRAMES, MODEL_ID, MODEL_REVISION, PHONEMES,
    SAMPLE_RATE, SOURCE_COMMIT, JapanesePPGExtractor, compare_posteriors,
    frame_times, load_audio, sha256_file,
)
from .model_jsut_tokens import configure_plots, save_npz, write_json


VARIANTS = ("original", "original_ema", "reconstructed_ema")
LABELS = {"original": "元音声", "original_ema": "オリジナル軌道から合成",
          "reconstructed_ema": "トークン再構成軌道から合成"}
PLOT_LABELS = {"original": "Original audio", "original_ema": "Original EMA synthesis",
               "reconstructed_ema": "Token EMA synthesis"}
PAIRS = (("original", "original_ema"), ("original", "reconstructed_ema"),
         ("original_ema", "reconstructed_ema"))


def select_inputs(run_dir, limit):
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if (config["window_frames"] != 2 or config["codebook_size"] != 128
            or config.get("token_rate", 50) != 50):
        raise ValueError("Expected the primary 40 ms / 128-code / 50 Hz run")
    comparison_dir = (run_dir / "audio_comparison").resolve()
    manifest = json.loads((comparison_dir / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(run_dir / "best.pt") != manifest["checkpoint_sha256"]:
        raise ValueError("Audio comparison is from a different VQ-VAE checkpoint")
    test = set(json.loads((run_dir / "splits.json").read_text())["test"])
    records = manifest["records"][:limit]
    if not records:
        raise ValueError("No comparison utterances were found")
    seen = set()
    prepared = []
    for record in records:
        relative = Path(record["feature_path"])
        if (relative.is_absolute() or ".." in relative.parts or relative.as_posix() not in test
                or record["split"] != "test" or relative in seen):
            raise ValueError(f"Invalid or duplicate test utterance: {relative}")
        seen.add(relative)
        audio = {}
        for variant in VARIANTS:
            source = (comparison_dir / record["audio"][variant]).resolve()
            if not source.is_relative_to(comparison_dir) or not source.is_file():
                raise ValueError(f"Missing comparison audio inside {comparison_dir}: {source}")
            waveform, info = load_audio(source)
            audio[variant] = {"path": source, "waveform": waveform, "info": info,
                              "sha256": sha256_file(source)}
        prepared.append((record, audio))
    return manifest, prepared


def statistics(values):
    return {"frames": len(values), "mean_js_nats": float(np.mean(values)),
            "median_js_nats": float(np.median(values)), "p95_js_nats": float(np.quantile(values, 0.95)),
            "max_js_nats": float(np.max(values))}


def plot_comparison(path, utterance_id, posteriors, comparisons):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 3, 1.5]})
    end = max(len(values) for values in posteriors.values()) / FRAME_RATE
    for axis, variant in zip(axes[:3], VARIANTS):
        values = posteriors[variant]
        image = axis.imshow(values.T, aspect="auto", origin="upper", vmin=0, vmax=1,
                            extent=(0, len(values) / FRAME_RATE, len(PHONEMES) - 0.5, -0.5),
                            interpolation="nearest", cmap="viridis")
        axis.set_yticks(np.arange(len(PHONEMES)), labels=PHONEMES, fontsize=7)
        axis.set_title(f"{utterance_id} — {PLOT_LABELS[variant]}", fontsize=10)
        axis.set_ylabel("Japanese phone")
    for key, item in comparisons.items():
        first, second = key.split("__")
        axes[3].plot(item["time"], item["js"], label=f"{PLOT_LABELS[first]} vs {PLOT_LABELS[second]}", linewidth=1)
    axes[3].set(xlabel="Time (s)", ylabel="JS divergence (nats)", xlim=(0, end), ylim=(0, np.log(2)))
    axes[3].legend(fontsize=7)
    axes[3].grid(alpha=0.2)
    fig.tight_layout(rect=(0, 0, 0.94, 1))
    colorbar_axis = fig.add_axes((0.95, 0.30, 0.012, 0.60))
    fig.colorbar(image, cax=colorbar_axis, label="Posterior probability")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(output_dir, records, aggregates):
    rows = []
    for key, values in aggregates.items():
        first, second = key.split("__")
        rows.append(f'<tr><td>{LABELS[first]} ↔ {LABELS[second]}</td>'
                    f'<td>{values["mean_js_nats"]:.5f}</td><td>{values["p95_js_nats"]:.5f}</td>'
                    f'<td>{values["frames"]}</td></tr>')
    cards = []
    for record in records:
        players = ''.join(f'<div><h3>{LABELS[variant]}</h3>{audio_player(record["variants"][variant]["audio"])}'
                          f'<p>PPG：{record["variants"][variant]["frames"]}フレーム</p></div>' for variant in VARIANTS)
        details = []
        for key, values in record["comparisons"].items():
            first, second = key.split("__")
            details.append(f'<li>{LABELS[first]} ↔ {LABELS[second]}：平均JS {values["mean_js_nats"]:.5f}'
                           f'（共通{values["frames"]}フレーム）</li>')
        cards.append(f'<section><h2>{html.escape(record["utterance_id"])}</h2><p>{html.escape(record["text"])}</p>'
                     f'<div class="players">{players}</div><ul>' + ''.join(details) + '</ul>'
                     f'<a href="{html.escape(record["figure"], quote=True)}">図を拡大</a>'
                     f'<img loading="lazy" src="{html.escape(record["figure"], quote=True)}" alt="3音声のPPGとJS divergence"></section>')
    content = '''<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>JSUT 日本語PPG比較</title>
<style>body{font-family:system-ui,sans-serif;max-width:1250px;margin:30px auto;padding:0 18px;color:#182431}
section{border:1px solid #ddd;border-radius:12px;padding:18px;margin:26px 0}
.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:18px}
audio{width:100%;max-width:360px}img{max-width:100%;display:block;margin:16px auto}
p,li{line-height:1.7}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}
table{border-collapse:collapse;width:100%}h3{font-size:1rem}</style>
<h1>40 ms・128コード：日本語PPGの比較</h1>
<p>元音声・オリジナル調音軌道からの合成・トークン再構成軌道からの合成を、同じ日本語PPGモデルで比較しています。
PPGは無音を含む35音素の推定確率で、100 Hz（10 ms刻み）です。音声やPPGの時間伸縮・位置合わせは行いません。</p>
<p>JS divergenceは同じ時刻の分布差を表し、自然対数で0〜log(2)の値を取ります。
音声の長さが異なる場合は共通時間区間のみを比較します。PPGは正解音素ラベルや、発音の正しさそのものではありません。</p>
<table><tr><th>比較する音声</th><th>平均JS（nats）</th><th>95%点</th><th>比較フレーム数</th></tr>'''
    content += ''.join(rows) + '</table><p><a href="summary.csv">発話ごとの数値CSV</a></p>' + ''.join(cards)
    content += '''<script>document.querySelectorAll('audio').forEach(p=>p.addEventListener('play',()=>{
document.querySelectorAll('audio').forEach(q=>{if(q!==p)q.pause();});}));</script></html>'''
    (output_dir / "index.html").write_text(content, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/jsut_window_sweep/window_040ms"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/jsut_ppg_40ms128"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    if args.limit < 1 or args.threads < 1:
        parser.error("--limit and --threads must be positive")
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error(f"Output directory is not empty: {args.output_dir}; choose a new --output-dir")
    try:
        device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                              if args.device == "auto" else args.device)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    if device.type not in ("cpu", "cuda"):
        parser.error("--device must be auto, cpu, cuda, or cuda:N")
    if device.type == "cuda" and (not torch.cuda.is_available() or (
            device.index is not None and device.index >= torch.cuda.device_count())):
        parser.error(f"CUDA device is unavailable: {device}")
    torch.set_num_threads(args.threads)
    try:
        source_manifest, prepared = select_inputs(args.run_dir, args.limit)
        print(f"Loading Japanese PPG model on {device}; {len(prepared)} utterances / 3 variants", file=sys.stderr, flush=True)
        extractor = JapanesePPGExtractor(device=device)
        configure_plots()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "source_commit": SOURCE_COMMIT,
                    "checkpoint_sha256": extractor.checkpoint_sha256, "phonemes": list(PHONEMES),
                    "frame_rate": FRAME_RATE, "sample_rate": SAMPLE_RATE, "is_causal": False,
                    "frontend": {"fft_size": FFT_SIZE, "hop_samples": HOP, "mel_channels": 80,
                                 "padding": "reflect", "padding_samples": (FFT_SIZE - HOP) // 2,
                                 "frame_center_offset_samples": HOP // 2, "amplitude_normalization": "none",
                                 "rounding": "FP16 magnitude and log-Mel; FP32 Mel matmul and Transformer",
                                 "autocast": False, "max_frames": MAX_FRAMES}}
        records, csv_rows = [], []
        all_distances = {f"{first}__{second}": [] for first, second in PAIRS}
        for record, inputs in tqdm(prepared, desc="Japanese PPG", unit="utterance"):
            relative = Path(record["feature_path"]).with_suffix("")
            directory = args.output_dir / relative
            directory.mkdir(parents=True, exist_ok=True)
            posteriors, times, variants = {}, {}, {}
            for variant in VARIANTS:
                item = inputs[variant]
                posterior = extractor.extract(item["waveform"])
                time = frame_times(len(posterior))
                posteriors[variant], times[variant] = posterior, time
                destination = directory / f"{variant}.npz"
                save_npz(destination, ppg=posterior, time=time, phonemes=np.asarray(PHONEMES),
                         frame_rate=np.float64(FRAME_RATE), sample_rate=np.int64(SAMPLE_RATE),
                         **{key: np.asarray(value) for key, value in item["info"].items()},
                         source=np.str_(str(item["path"])), audio_sha256=np.str_(item["sha256"]),
                         variant=np.str_(variant), utterance_id=np.str_(record["utterance_id"]),
                         model_id=np.str_(MODEL_ID), model_revision=np.str_(MODEL_REVISION),
                         source_commit=np.str_(SOURCE_COMMIT), model_sha256=np.str_(extractor.checkpoint_sha256))
                audio_path = directory / f"{variant}.wav"
                shutil.copyfile(item["path"], audio_path)
                variants[variant] = {**item["info"], "frames": len(posterior),
                                     "source": str(item["path"]), "audio_sha256": item["sha256"],
                                     "ppg": destination.relative_to(args.output_dir).as_posix(),
                                     "audio": audio_path.relative_to(args.output_dir).as_posix()}
            comparisons, comparison_arrays, summaries = {}, {}, {}
            for first, second in PAIRS:
                key = f"{first}__{second}"
                time, distances = compare_posteriors(posteriors[first], times[first], posteriors[second], times[second])
                comparisons[key] = {"time": time, "js": distances}
                comparison_arrays[f"{key}_time"] = time
                comparison_arrays[f"{key}_js_nats"] = distances
                summaries[key] = statistics(distances)
                all_distances[key].append(distances)
                csv_rows.append({"utterance_id": record["utterance_id"], "pair": key, **summaries[key]})
            save_npz(directory / "comparisons.npz", **comparison_arrays)
            figure = directory / "ppg_comparison.png"
            plot_comparison(figure, record["utterance_id"], posteriors, comparisons)
            records.append({"utterance_id": record["utterance_id"], "text": record["text"],
                            "feature_path": record["feature_path"], "split": "test", "variants": variants,
                            "figure": figure.relative_to(args.output_dir).as_posix(), "comparisons": summaries,
                            "comparison_npz": (directory / "comparisons.npz").relative_to(args.output_dir).as_posix()})
        aggregates = {key: statistics(np.concatenate(values)) for key, values in all_distances.items()}
        write_json(args.output_dir / "summary.json", {"utterances": len(records), "variants": list(VARIANTS),
                                                     "aggregates": aggregates, "records": csv_rows})
        with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        write_json(args.output_dir / "manifest.json", {
            "run_dir": str(args.run_dir.resolve()), "vq_checkpoint_sha256": source_manifest["checkpoint_sha256"],
            "device": str(device), "threads": args.threads, "ppg_model": metadata, "records": records,
            "comparison": {"metric": "standard Jensen-Shannon divergence", "units": "nats",
                           "alignment": "common absolute-time frames only", "time_warping": False}})
        write_report(args.output_dir, records, aggregates)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"PPGs: {args.output_dir}\nReport: {args.output_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
