# Speech Articulatory Coding (SPARC)
[Paper](https://arxiv.org/abs/2406.12998) | [Audio Samples](https://berkeley-speech-group.github.io/sparc-demo) | [Colab Demo](https://colab.research.google.com/drive/1TVGJJpOzPiesLPo46gZNCQLMl-y_QIKe#scrollTo=uBemLVlk-s7W)

<div align="center">
    <img src="images/articulatory_coding.png" alt="drawing" width="600"/>
</div>

This is the official code base for [Coding Speech through Vocal Tract Kinematics](https://arxiv.org/abs/2406.12998).

## このリポジトリでの研究運用

- `upstream-original`: 著者版を保持し、著者リポジトリの `upstream/main` を追跡する。
- `main`: 研究で使用する、動作確認済みの安定版。自分の fork の `origin/main` と同期する。
- 新しい研究・実験・修正は、必ず `main` から新しい作業ブランチを作成して進める。検証した変更を `main` に取り込む。

```sh
git switch main
git switch -c research/<topic>
```

詳細な手順と著者版の更新方法は [AGENTS.md](AGENTS.md) を参照。

## 音声ファイルからの再合成

入力音声を SPARC の調音・音源・話者特徴に変換し、その特徴から WAV を再合成する。
`multi` モデル・CREPE・GPU（CUDA）を使用する。
リポジトリのルートで、[uv](https://docs.astral.sh/uv/) による環境構築を行う。

```sh
bash scripts/setup_resynthesis.sh
```

Python と依存ライブラリは共有ドライブを避け、ローカルの
`~/.local/share/sparc/venv/` に配置する（`XDG_DATA_HOME` 設定時はその配下）。
環境の配置先は `SPARC_ENV_DIR` で変更できる。
Python 3.12、PyTorch 2.8.0 / CUDA 12.8 を導入し、実行にも `uv run` を使用する。

```sh
# 付属音声で動作確認
bash scripts/resynthesize.sh sample_audio/sample1.wav

# 自分の音声を再合成し、解析した特徴量も保存
bash scripts/resynthesize.sh /path/to/input.wav \
    --output outputs/resynthesized.wav \
    --features outputs/features.npz
```

出力先を省略すると `outputs/<入力ファイル名>_resynth.wav` に保存する。
出力は 16 kHz・モノラル・16 bit PCM の WAV。
特徴量のフレーム長に合わせて合成するため、出力時間は入力とわずかに異なる場合がある。
ステレオ入力はモノラル化し、入力のサンプリング周波数は内部で 16 kHz に変換する。
0.5 秒以上の、無音ではない音声を指定する。
既存の出力を置き換える場合は `--overwrite` を付ける。
初回実行時はモデルのダウンロードにインターネット接続が必要。
モデルは `.cache/huggingface/` に保存する（`HF_HOME` 設定時はその保存先）。

`--features` で保存した `.npz` は、以下のように読み込める。

```python
import numpy as np

with np.load("outputs/features.npz", allow_pickle=False) as features:
    code = dict(features)
# ema, pitch, loudness, periodicity, pitch_stats, spk_emb, ft_len
```

## フィードバック基礎実験1

実験1の実装は `experiments/feedback/` に分離し、SPARC本体は変更していない。
実測結果・品質上の制約は [reports/initial_experiment.md](reports/initial_experiment.md) を参照。
音響特徴の順序は **`[F0, F1, F2, F3]`**（すべてHz）。F0は原音と再合成音を
同じPraat自己相関法・同じWAV時刻で測定し、有声フレームの中央値を使う。
SPARCの `code['pitch']`（CREPEによるDecoderの入力条件）とは区別する。
F0にも有効10フレーム以上・有効率80%以上・SD/中央値≤0.15の暫定基準を適用する。
無声・欠測は補完せず `NaN`、採用できない区間代表値は `null` とする。
`*_acoustics.csv/npz` に4次元の軌跡・有効判定、`acoustic_comparison.csv` に
元音声との差・共有有効フレームのMAE/RMSE、`f0.png` にF0軌跡とpitch入力条件を保存する。
`summary.json` はschema version 2で、`feature_names` と4次元の `baseline_target_hz` を保存する。
既存の3次元フォルマント配列は `*_formants.npz` と `baseline_formant_target_hz` に残す。
F0追加後の実測結果は `outputs/feedback_experiment1_f0/` に保存した。
既存の再合成環境に、追加依存をインストールする。共有ドライブではコピー方式を使用する。

```sh
uv pip install --python ~/.local/share/sparc/venv/bin/python \
  --target .cache/feedback-deps --cache-dir /tmp/sparc-feedback-uv \
  --link-mode copy --no-deps -r requirements-feedback.txt

# 最初に /a/ のみ。出力先は空のディレクトリを指定する。
bash scripts/experiment1.sh --vowel a --output outputs/experiment1_a_new
# /a/, /i/, /u/。モデル・区間・品質基準はYAMLで指定。
bash scripts/experiment1.sh --output outputs/experiment1_new
# ユーザー指定WAV。区間の秒数はそのWAVの先頭を0とする。
bash scripts/experiment1.sh --input /path/to/a.wav --label a \
  --start 0.5 --end 1.0 --output outputs/custom_a
# CPUでも同じコマンドを実行できる。
bash scripts/experiment1.sh --device cpu --vowel a --output outputs/experiment1_cpu_new
```

`configs/feedback_experiment1.yaml` に入力、母音ラベル、解析区間、モデル、デバイス、
乱数seed、Praat設定を指定する。パスはリポジトリルートからの相対パス。
既存モデルを使うため、スクリプトは既定でオフライン実行する。
重みが未取得の場合は `HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0` で取得を許可する。
`--device auto` はCUDAを検出した場合GPUを使い、それ以外はCPUを使う。
モデル不要の検証は次で実行できる。

```sh
PYTHONPATH="$PWD/.cache/feedback-deps" MPLCONFIGDIR="$PWD/.cache/matplotlib" \
  ~/.local/share/sparc/venv/bin/python -m unittest discover -s tests -v
```

実装で確認したAPI・制約：

- `load_model("multi", device=..., use_penn=False)` と `encode(Path)` / `decode(**code)` を使用。
  日本語を含む学習データで、既存キャッシュのある `multi` を選んだ。
- 入力はモデル内でモノラル化・16 kHzへ変換され、波形の平均0・標準偏差1に正規化される。
  EMAへの追加正規化は行わない。EMAの数値をmmや今回の母音内SDと同一視しない。
- EMAは約50 Hzの `[T, 12]`、順序は
  `TDX,TDY,TBX,TBY,TTX,TTY,LIX,LIY,ULX,ULY,LLX,LLY`。
  実測ではEMA・pitch・loudnessの長さが異なる。`ft_len`をEMA長と仮定せず、
  Decoder内部の `_match_and_cat` が最短長に揃える動作をそのまま使う。
- Decoderの実入力はEMA・pitch・loudness・spk_emb。
  periodicity・pitch_stats・ft_lenは保存するが、`decode` の `**kwargs` に入り直接使われない。
  pitchはGenerator内で `(pitch - 50) * 0.01` に変換される。
- EncoderはWavLMと10 Hz低域フィルタ、Decoderは時間方向の畳み込み・転置畳み込み・
  dilated residual blocksを使用する。前後文脈に依存するのでWAV全体をencode/decodeし、
  指定区間だけの音響特徴を集計する。逐次的な閉ループモデルとしては扱わない。
- EMA時刻は `frame_index / 50` の公称時刻。音声とEMAの厳密な遅延補正は行っていない。
  原音と再合成音は共通のWAV時刻区間で解析する。
- 同一codeを2回decodeし、全条件の不変性と音声の一致を検証する。
  目標特徴は再合成音声の品質判定を通過した中央値とし、不採用値はJSONでは `null`、
  CSV/NPZでは `NaN` と理由を保存する。原音の値で補完しない。

## ヤコビアン推定とオフライン補正

実験2以降の /i/・F1/F2 の予備実験は
[reports/feedback_control_i.md](reports/feedback_control_i.md) にまとめた。
F0・F3も観測・保存し、EMAのみを操作する。基準再合成への摂動復元を検証してから、
元音声への補正を実施する。新しい出力先を全段階で共通に指定する。

```sh
bash scripts/feedback_control.sh --phase jacobian --output outputs/control_new
bash scripts/feedback_control.sh --phase recovery --output outputs/control_new
bash scripts/feedback_control.sh --phase source --output outputs/control_new
PYTHONPATH="$PWD/.cache/feedback-deps" MPLCONFIGDIR="$PWD/.cache/matplotlib" \
  ~/.local/share/sparc/venv/bin/python -m experiments.feedback.control_plots outputs/control_new
PYTHONPATH="$PWD/.cache/feedback-deps" MPLCONFIGDIR="$PWD/.cache/matplotlib" \
  ~/.local/share/sparc/venv/bin/python -m experiments.feedback.control_report outputs/control_new
```

設定は `configs/feedback_control_i.yaml`。有限差分の検証基準を通過しなければ後段の実行を拒否する。
各実生成のDecoder時間・音響解析時間・保存時間を `evaluations.jsonl` に記録し、
25回ごとに累積時間を表示する。`timing.json` は段階全体の集計。
結果は `outputs/feedback_control_i/` に保存し、元音声への補正後の音声は
`source/corrected.wav`、条件を含む特徴は `source/corrected_code.npz` にある。
ヤコビアンの主行列は2×12で、単位はHz/参照SD。参照SDは実験1の母音区間27フレームから求め、
訓練データのSDとは区別する。正規化座標・EMA座標の行列も保存する。

有効判定なしで /a/・/i/・/u/ を比較する設定は `configs/feedback_control_raw.yaml`。
周波数範囲・帯域幅・有声判定・時間変動などによる音響推定値の除外をせず、
指定区間で数値が得られた全フレームの中央値を使用する。
F1/F2を補正し、F0/F3も保存する。未定義値は補完しない。
以前の判定ありの /i/ 結果と区別し、`outputs/feedback_control_{a,i,u}_raw/` に保存する。

```sh
for vowel in a u i; do
  for phase in jacobian recovery source; do
    bash scripts/feedback_control.sh --config configs/feedback_control_raw.yaml \
      --label "$vowel" --phase "$phase"
  done
done
for vowel in a u i; do
  PYTHONPATH="$PWD/.cache/feedback-deps" MPLCONFIGDIR="$PWD/.cache/matplotlib" \
    ~/.local/share/sparc/venv/bin/python -m experiments.feedback.control_plots \
    "outputs/feedback_control_${vowel}_raw"
done
PYTHONPATH="$PWD/.cache/feedback-deps" MPLCONFIGDIR="$PWD/.cache/matplotlib" \
  ~/.local/share/sparc/venv/bin/python -m experiments.feedback.raw_control_report
```

比較結果と図は [reports/feedback_control_raw.md](reports/feedback_control_raw.md) にまとめる。
数値比較は `outputs/feedback_control_raw_summary.json` と `outputs/feedback_control_raw_features.csv`。
ヤコビアンの予測誤差の基準も試行の除外には使用せず、診断値として記録する。

デモページの日本語文章音声についても、有効判定なしのF0・F1–F3を抽出した。
[結果・比較図・再現コマンド](reports/japanese_demo_raw.md) と、
`outputs/japanese_demo_raw/` の時系列CSV/NPZを参照。
任意のWAVには `python -m experiments.feedback.analyze_audio --input INPUT.wav --output OUTPUT_DIR`
を使用でき、`--resynthesis` で比較用音声、`--start`・`--end` で解析区間を指定できる。

## Installation

```
pip install speech-articulatory-coding
```

## Usage

#### Load Model

```python
from sparc import load_model
coder = load_model("en", device= "cpu")     # For using CPU
coder = load_model("en", device= "cuda:0")  # For using GPU
```

For pitch tracker, we found [PENN](https://github.com/interactiveaudiolab/penn) is fast at inference. You can activate that with `use_penn=True`. The default is using [torchcrepe](https://github.com/maxrmorrison/torchcrepe).

```python
coder = load_model("en", device= "cpu", use_penn=True)    # Use PENN for pitch tracker

```

For inversion only, you can use the following,

```python
coder = load_model("feature_extraction") 
coder_from_config = load_model(config="configs/feature_extraction.yaml")
```

The following model checkpoints are offered. You can replace `en` with other models (`multi` or `en+`) in `load_model`.

| Model  | Language |     Training Dataset    | 
|--------|:--------:|:--------------:|
| en     |  English |LibriTTS-R|
| multi  |   Multi  |LibriTTS-R, Multilignual LibriSpeech, AISHELL, JVS, KSS    |
| en+     |  English |LibriTTS-R, LibriTTS, EXPRESSO|


#### Articulatory Analysis

```python
code = coder.encode(WAV_FILE)          # Single inference
codes = coder.encode([WAV_FILE1, WAV_FILE2, ...]) # Batched processing
```


The articulatory code outputs have the following format.

```python
# All features are in 50 Hz except speaker encoding
{"ema": (L, 12) array, #'TDX','TDY','TBX','TBY','TTX','TTY','LIX','LIY','ULX','ULY','LLX','LLY'
 "loudness": (L, 1) array, 
 "pitch": (L, 1) array, 
 "periodicity": (L, 1) array, # auxiliary output of pitch tracker
 "pitch_stats": (pitch mean, pitch std),
 "spk_emb": (spk_emb_dim,) array, # all shared models use spk_emb_dim=64
 "ft_len": Length of features, # usefull when batched processing with padding
}
```

#### Articulatory Synthesis

```python
wav = coder.decode(**code)
sr = coder.sr
```

#### Voice Conversion

```python
wav = coder.convert(SOURCE_WAV_FILE, TARGET_WAV_FILE)
sr = coder.sr
```
#### Demo

Please check `notebooks/demo.ipynb` for a demonstration of the functions.


### Training

#### Feature extraction

Check `scripts/encode_audio.py` and an example script for extracting LibriTTS, `scripts/extract_libritts.sh`


## TODO

- Add training codes.

