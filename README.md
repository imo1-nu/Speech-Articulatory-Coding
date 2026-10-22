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

