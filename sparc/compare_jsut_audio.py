"""Listen to test audio, SPARC EMA synthesis, and VQ-reconstructed EMA synthesis."""

import argparse
import html
import json
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import quote

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
import yaml

from .model_jsut_tokens import checked_feature, checkpoint_hash, load_run, save_npz, write_json
from .sparc import load_model
from .trajectory_vq import overlap_average, triangular_weights


REPO_DIR = Path(__file__).resolve().parents[1]
TRIANGULAR_WEIGHTS = np.asarray([1, 2, 3, 2, 1], dtype=np.float64)


def select_test(splits, utterances, limit):
    test = splits["test"]
    if utterances:
        selected = []
        for requested in utterances:
            matches = [relative for relative in test
                       if relative == requested or Path(relative).stem == requested]
            if len(matches) != 1:
                raise ValueError(f"Expected one test utterance matching {requested}, found {len(matches)}")
            if matches[0] not in selected:
                selected.append(matches[0])
    else:
        selected = test
    if not selected:
        raise ValueError("The run has no test utterances")
    return selected[:limit]


def prepare_inputs(run_dir, jsut_dir, selected):
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    digest = checkpoint_hash(run_dir / "best.pt")
    items = []
    for relative in selected:
        original = checked_feature(config, relative)
        for name in ("pitch", "loudness", "periodicity", "source"):
            if name not in original:
                raise ValueError(f"Missing {name} in original features: {relative}")
        with np.load(run_dir / "tokens" / relative, allow_pickle=False) as data:
            reconstruction = data["reconstruction"].astype(np.float32)
            if (str(data["checkpoint_sha256"]) != digest or str(data["split"]) != "test"
                    or str(data["feature_path"]) != relative
                    or reconstruction.shape != original["ema"].shape
                    or not np.isfinite(reconstruction).all()
                    or not np.array_equal(data["time"], original["time"])):
                raise ValueError(f"Stale or unaligned test reconstruction: {relative}; rerun tokenize")
        source = (jsut_dir / str(original["source"])).resolve()
        if not source.is_relative_to(jsut_dir.resolve()) or not source.is_file():
            raise ValueError(f"Original audio is missing or outside --jsut-dir: {source}")
        items.append((relative, original, reconstruction, source))
    return digest, items


def weighted_reconstructions(run_dir, items):
    """Decode existing tokens with triangular weights; keep saved outputs intact."""
    config, _, model, mean, std = load_run(run_dir, torch.device("cpu"))
    with torch.inference_mode():
        decoded = model.decoder(model.codebook.weight).numpy().reshape(-1, model.window_frames, 12)
    result = {}
    for relative, _, reconstruction, _ in items:
        with np.load(run_dir / "tokens" / relative, allow_pickle=False) as data:
            tokens = data["tokens"]
        if (tokens.shape != ((len(reconstruction) + model.token_stride - 1) // model.token_stride,)
                or not np.issubdtype(tokens.dtype, np.integer)
                or tokens.min() < 0 or tokens.max() >= config["codebook_size"]):
            raise ValueError(f"Invalid tokens for weighted reconstruction: {relative}")
        windows = decoded[tokens]
        uniform = overlap_average(windows, stride=model.token_stride,
                                  output_length=len(reconstruction)) * std + mean
        if not np.allclose(uniform, reconstruction, rtol=1e-5, atol=1e-5):
            raise ValueError(f"Saved reconstruction does not match uniform averaging: {relative}")
        weights = triangular_weights(model.window_frames)
        result[relative] = (reconstruction.copy() if np.all(weights == weights[0])
                            else overlap_average(windows, weights, stride=model.token_stride,
                                                 output_length=len(reconstruction)) * std + mean)
    return result


def synthesize_trajectories(coder, source, original, trajectories):
    """Use one set of source features and one common gain for all EMA variants."""
    if float(original["periodicity"].sum()) <= 0:
        raise ValueError("Cannot estimate speaker embedding with zero periodicity weights")
    with torch.inference_mode():
        outputs = coder.inverter(source, outputs={}, include_acoustics=True)
        outputs["periodicity"] = original["periodicity"][None]
        outputs = coder.speaker_encoder(outputs["wav"], outputs)
        speaker = np.asarray(outputs["spk_emb"][0], dtype=np.float32)
        if speaker.ndim != 1 or not np.isfinite(speaker).all():
            raise ValueError("Invalid speaker embedding")
        common = {"pitch": original["pitch"], "loudness": original["loudness"], "spk_emb": speaker}
        synthesized = {name: coder.decode(ema=ema, **common) for name, ema in trajectories.items()}
    for name, waveform in synthesized.items():
        if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
            raise ValueError(f"Invalid synthesized waveform: {name}")
    if len({len(waveform) for waveform in synthesized.values()}) != 1:
        raise ValueError("The synthesized waveforms have different lengths")
    # Preserve the relative levels of the syntheses; prevent PCM clipping together.
    peak = max(float(np.max(np.abs(waveform))) for waveform in synthesized.values())
    gain = min(1.0, 0.99 / peak) if peak else 1.0
    return {name: waveform * gain for name, waveform in synthesized.items()}, speaker, gain


def audio_player(relative):
    return f'<audio controls preload="none" src="{html.escape(quote(relative), quote=True)}"></audio>'


def write_report(output_dir, records):
    cards = []
    for record in records:
        players = []
        labels = [("1. 元音声", "original"), ("2. 抽出した軌道から合成", "original_ema"),
                  ("3. VQ再構成・等重み平均", "reconstructed_ema")]
        order = "元音声 → 抽出軌道 → 等重みVQ再構成"
        if "weighted_ema" in record["audio"]:
            labels.append(("4. VQ再構成・中央重み付き平均", "weighted_ema"))
            order += " → 中央重み付きVQ再構成"
        for label, key in labels:
            players.append(f'<div><h3>{label}</h3>{audio_player(record["audio"][key])}</div>')
        cards.append(f'<section><h2>{html.escape(record["utterance_id"])}</h2>'
                     f'<p>{html.escape(record["text"])}</p><div class="players">'
                     + "".join(players) + f'</div><p>連続比較：{order}（間に0.5秒の無音）</p>'
                     + audio_player(record["audio"]["comparison"]) + '</section>')
        if "weighted_ab" in record["audio"]:
            cards[-1] = cards[-1].removesuffix('</section>') + (
                '<p>A/B比較：等重み平均 → 中央重み付き平均（間に0.5秒の無音）</p>'
                + audio_player(record["audio"]["weighted_ab"]) + '</section>')
    content = '''<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>JSUT テスト音声の聴き比べ</title><style>
body{font-family:system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;background:#f5f7fa;color:#182431}
section{background:white;border:1px solid #dce2e9;border-radius:12px;padding:20px;margin:20px 0}
.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px}
audio{width:100%;max-width:380px}h3{font-size:1rem}p{line-height:1.7}
@media(max-width:850px){.players{grid-template-columns:1fr}}
</style><h1>JSUT テスト音声の聴き比べ</h1>
<p>本学習モデルのテスト分割から選んだ発話です。各合成は同じピッチ・音量・話者埋め込みを使用し、
調音軌道だけを変更しています。合成音声は16 kHz、元音声のWAVはそのままコピーしています。
合成にはクリッピング防止の共通ゲインだけを適用し、個別の音量正規化は行っていません。</p>
'''
    if records and "weighted_ema" in records[0]["audio"]:
        weights = "・".join(str(int(value)) for value in records[0]["overlap_weights"])
        content += ('<p>3・4は同じ学習済みモデル・トークン列を使い、窓の平均方法だけを変更しています。'
                    f'4の重みは、窓の先頭から {weights}。発話端では有効な予測の重みで正規化します。</p>')
    content += "\n".join(cards)
    content += '''<script>document.querySelectorAll('audio').forEach(player=>{
player.addEventListener('play',()=>document.querySelectorAll('audio').forEach(other=>{
if(other!==player)other.pause();}));});</script></html>'''
    (output_dir / "index.html").write_text(content, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/jsut_tokens"))
    parser.add_argument("--jsut-dir", type=Path, default=Path("datasets/jsut/jsut_ver1.1"))
    parser.add_argument("--output-dir", type=Path,
                        help="Default: <run-dir>/audio_comparison")
    parser.add_argument("--limit", type=int, default=5, help="Maximum test utterances (default: 5)")
    parser.add_argument("--utterances", nargs="+", help="Test utterance IDs or relative NPZ paths")
    parser.add_argument("--include-weighted", action="store_true",
                        help="Also compare symmetric triangular overlap weights, without retraining")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    if args.limit < 1 or args.threads < 1:
        parser.error("--limit and --threads must be positive")
    output_dir = args.output_dir or args.run_dir / "audio_comparison"
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        parser.error(f"Output directory is not empty: {output_dir}; choose a new --output-dir")
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
        splits = json.loads((args.run_dir / "splits.json").read_text(encoding="utf-8"))
        selected = select_test(splits, args.utterances, args.limit)
        digest, items = prepare_inputs(args.run_dir, args.jsut_dir, selected)
        weighted = weighted_reconstructions(args.run_dir, items) if args.include_weighted else {}
        run_config = json.loads((args.run_dir / "config.json").read_text())
        weights = triangular_weights(run_config.get("window_frames", 5))
        # Match the zero padding used by the stored feature-extraction model.
        model_config = yaml.safe_load((REPO_DIR / "configs/model_multiling.yaml").read_text())
        model_config["zero_pad"] = True
        print(f"Loading SPARC multi on {device}; {len(items)} test utterances", file=sys.stderr, flush=True)
        coder = load_model("multi", config=model_config, device=str(device))
        if coder.ft_sr != 50 or coder.output_sr != 16000:
            raise ValueError("Expected a 50 Hz, 16 kHz SPARC synthesis model")
        output_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for relative, original, reconstruction, source in tqdm(items, desc="Synthesize", unit="utterance"):
            trajectories = {"original_ema": original["ema"], "reconstructed_ema": reconstruction}
            if args.include_weighted:
                trajectories["weighted_ema"] = weighted[relative]
            synthesized, speaker, gain = synthesize_trajectories(coder, source, original, trajectories)
            directory = output_dir / Path(relative).with_suffix("")
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, directory / "original.wav")
            audio_paths = {"original": (directory / "original.wav").relative_to(output_dir).as_posix()}
            for name, waveform in synthesized.items():
                filename = directory / f"{name}.wav"
                sf.write(filename, waveform, coder.output_sr, subtype="PCM_16")
                audio_paths[name] = filename.relative_to(output_dir).as_posix()
            wave, sr = sf.read(source, dtype="float32", always_2d=True)
            wave = wave.mean(axis=1)
            if sr != coder.output_sr:
                wave = librosa.resample(wave, orig_sr=sr, target_sr=coder.output_sr)
            gap = np.zeros(coder.output_sr // 2, dtype=np.float32)
            clips = [wave]
            for waveform in synthesized.values():
                clips.extend((gap, waveform))
            combined = np.concatenate(clips)
            if not np.isfinite(combined).all():
                raise ValueError(f"Non-finite comparison audio: {source}")
            comparison_gain = min(1.0, 0.99 / max(float(np.max(np.abs(combined))), 1e-8))
            sf.write(directory / "comparison.wav", combined * comparison_gain,
                     coder.output_sr, subtype="PCM_16")
            audio_paths["comparison"] = (directory / "comparison.wav").relative_to(output_dir).as_posix()
            extra_conditions = {}
            if args.include_weighted:
                ab = np.concatenate((synthesized["reconstructed_ema"], gap, synthesized["weighted_ema"]))
                sf.write(directory / "weighted_ab.wav", ab, coder.output_sr, subtype="PCM_16")
                audio_paths["weighted_ab"] = (directory / "weighted_ab.wav").relative_to(output_dir).as_posix()
                extra_conditions = {"weighted_reconstruction": weighted[relative],
                                    "overlap_weights": weights}
            save_npz(directory / "conditions.npz", pitch=original["pitch"],
                     loudness=original["loudness"], periodicity=original["periodicity"],
                     spk_emb=speaker, ema=original["ema"], reconstruction=reconstruction,
                     time=original["time"], checkpoint_sha256=np.str_(digest), **extra_conditions)
            record = {"utterance_id": Path(relative).stem, "feature_path": relative,
                      "text": str(original.get("text", "")), "source": str(source), "split": "test",
                      "frames": len(original["ema"]), "synthesis_sample_rate": coder.output_sr,
                      "synthesis_samples": len(synthesized["original_ema"]), "synthesis_common_gain": gain,
                      "comparison_common_gain": comparison_gain,
                      "overlap_weights": weights.tolist() if args.include_weighted else None,
                      "audio": audio_paths}
            records.append(record)
            write_json(output_dir / "manifest.json", {
                "run_dir": str(args.run_dir.resolve()), "checkpoint_sha256": digest,
                "synthesis_model": "multi", "records": records,
                "common_conditions": ["pitch", "loudness", "speaker_embedding"],
                "weighted_overlap_weights": weights.tolist() if args.include_weighted else None,
            })
            write_report(output_dir, records)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Listening report: {output_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
