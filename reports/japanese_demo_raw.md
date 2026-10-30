# デモページの日本語音声：F0・F1–F3の抽出（有効判定なし）

対象：`gt_japanese_049_nonpara30_UT-PARAPHRASE-sent257-phrase2.wav`。
[SPARC公式デモページ](https://berkeley-speech-group.github.io/sparc-demo/) のMultilingual欄・JapaneseのGround Truthを使用した。
以前の原音のローカルコピーは見つからなかったため、同じファイル名の原音をデモページから再取得した。
以前保存したSPARC再合成音声も同じ条件で解析した。SPARCによる再生成は行っていない。

原音URL：[WAV](https://berkeley-speech-group.github.io/sparc-demo/static/multi/gt_japanese_049_nonpara30_UT-PARAPHRASE-sent257-phrase2.wav)。
原音SHA-256：`bd342a9c31f449a50eb45ada997fddfcd923b5f7eea3a149edf657eab3f1938b`。
取得日時とページのハッシュは [出典記録](../outputs/japanese_demo_raw/source_provenance.json) に保存した。

## 条件

原音は16 kHz・モノラル・7.0274375秒、既存再合成は16 kHz・モノラル・7.02秒。
共通区間0–7.02秒、10 ms刻みで702点を解析した。
フォルマントはPraat Burg法、最大5共鳴、上限5000 Hz、窓長25 ms、pre-emphasis 50 Hz。
F0はPraat自己相関法、60–400 Hz、抽出器のvoicing threshold=0.45。

有声判定・周波数範囲・帯域幅・時間変動・フレーム数などによる追加の有効判定は行っていない。
F1–F3は無声部分も含め、得られた全推定値を保存・描画する。
F0は抽出器が無声区間などで数値を返さない箇所があり、未定義値はNaNとして保存する。
欠測補完・0埋め・平滑化は行っていない。図の点はすべて青色。
文章音声全体の集計であり、母音別の代表値ではない。

## 数値結果

中央値は数値が得られた全フレームから計算した。得られなかった値は集計不能として扱う。

| 特徴 | 原音の中央値 (Hz) | 再合成の中央値 (Hz) | 原音の取得数 | 再合成の取得数 |
|---|---:|---:|---:|---:|
| F0 | 146.41 | 146.34 | 290 / 702 | 290 / 702 |
| F1 | 767.01 | 629.13 | 698 / 702 | 698 / 702 |
| F2 | 1868.06 | 1828.65 | 698 / 702 | 698 / 702 |
| F3 | 2873.71 | 2842.25 | 698 / 702 | 698 / 702 |

同じWAV時刻で両方の数値が存在するフレームについて、差を計算した。
音声間の遅延補正・音素単位の対応付けは行っていない。
フォルマント誤差には無声部分の推定値や共鳴番号の変化も含まれる。

| 特徴 | 比較できたフレーム数 | MAE (Hz) | RMSE (Hz) |
|---|---:|---:|---:|
| F0 | 279 | 2.54 | 3.71 |
| F1 | 698 | 216.04 | 355.08 |
| F2 | 698 | 247.83 | 360.32 |
| F3 | 698 | 238.05 | 324.99 |

## 図とデータ

![原音と再合成のF0・フォルマント](../outputs/japanese_demo_raw/acoustics_raw.png)

![重ね合わせ比較](../outputs/japanese_demo_raw/acoustics_overlay.png)

- [元音声](../outputs/japanese_demo_raw/original.wav) / [以前の再合成音声](../outputs/japanese_demo_raw/resynthesis.wav)
- [原音の時系列CSV](../outputs/japanese_demo_raw/original_acoustics.csv) / [再合成の時系列CSV](../outputs/japanese_demo_raw/resynthesis_acoustics.csv)
- [原音NPZ](../outputs/japanese_demo_raw/original_acoustics.npz) / [再合成NPZ](../outputs/japanese_demo_raw/resynthesis_acoustics.npz)
- [数値・条件・環境・時間のJSON](../outputs/japanese_demo_raw/summary.json) / [誤差CSV](../outputs/japanese_demo_raw/comparison.csv)

CSVの列は `time_s,F0_hz,F1_hz,F2_hz,F3_hz`。
NPZの `valid` は有限値の存在だけを表し、品質判定ではない。`filtered_hz` は `raw_hz` と同じ内容。
入力ファイルと保存コピーのハッシュ、全702フレームの保持、CSV/NPZの列順と欠測処理を確認した。

## 実行時間と再現

原音の抽出：0.0330秒。
再合成の抽出：0.0315秒。
作図：0.7429秒。
処理全体：0.8593秒（Python起動/import、ダウンロード、実装・本レポート作成を除く）。
今回はDecoderの生成は0回。

```sh
PYTHONPATH="$PWD/.cache/feedback-deps" MPLCONFIGDIR="$PWD/.cache/matplotlib" \
  ~/.local/share/sparc/venv/bin/python -m experiments.feedback.analyze_audio \
  --input outputs/source_audio/gt_japanese_049_nonpara30_UT-PARAPHRASE-sent257-phrase2.wav \
  --resynthesis outputs/gt_japanese_049_nonpara30_UT-PARAPHRASE-sent257-phrase2_resynth.wav \
  --source-metadata outputs/source_audio/japanese_demo_provenance.json \
  --output outputs/japanese_demo_raw_new
```

既存結果は上書きしない。必要に応じて `--start` と `--end` でWAV時刻の区間を指定できる。

## 誤差の平均と分散

同じWAV時刻の「再合成−原音」を符号付き誤差とし、両方の数値が存在するフレームで集計した。
有効判定による除外は行わず、遅延補正も行っていない。
分散は平均誤差からの二乗偏差をNで割った値（ddof=0）。

| 特徴 | 比較フレーム数 | 平均誤差 (Hz) | 分散 (Hz²) | 標準偏差 (Hz) |
|---|---:|---:|---:|---:|
| F0 | 279 | -1.558 | 11.345 | 3.368 |
| F1 | 698 | -41.133 | 124390.608 | 352.691 |
| F2 | 698 | -12.148 | 129682.855 | 360.115 |
| F3 | 698 | -3.849 | 105606.192 | 324.971 |

[集計CSV](../outputs/japanese_demo_raw/error_statistics.csv) / [定義・不偏分散も含むJSON](../outputs/japanese_demo_raw/error_statistics.json)
