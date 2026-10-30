# 実験1：調音軌道とF0・フォルマント特徴の取得

実施日：2026-10-30（JST）。作業ブランチ：`research/feedback-experiment1`。

後続の実験2・摂動復元・元音声への補正の結果は
[feedback_control_i.md](feedback_control_i.md) を参照。以下は実験1の記録である。

## F0追加版

ユーザーの追加指定に従い、音響特徴を **`[F0, F1, F2, F3]`**（Hz）に拡張し、
同じモデル・入力・母音区間・seedで3母音をGPU上で再実行した。
追加版は [outputs/feedback_experiment1_f0/](../outputs/feedback_experiment1_f0/) に保存した。
初回の音声・数値は `outputs/feedback_experiment1/` に保持している。

F0は元音声と再合成音声から、同じPraat自己相関法で測定した。
SPARC Encoderが返す `code['pitch']` はCREPEによるDecoderの入力条件であり、
今回の目標F0には**再合成音声から測ったF0**を用いる。
pitch・EMA・loudness・spk_emb等の操作は加えていない。

F0推定は60–400 Hz、voicing threshold=0.45、刻み10 ms。
区間代表値は有声フレームの中央値とし、有効10フレーム以上・有効率80%以上・
SD/中央値≤0.15という従来と共通の**暫定的な集計基準**を適用した。
F0にはフォルマント用の帯域幅条件を適用せず、F1–F3の成否から独立して採否を決める。
無声・欠測はNaN、採用不能な区間代表値とその比較指標はnullとし、0で埋めない。
この基準はSPARC公式の基準ではなく、閾値の妥当性は未検証である。

| 母音 | 元音声F0中央値 (Hz) | 再合成F0中央値 (Hz) | 中央値差 (Hz) | 相対差 | 共有有声フレーム数 | 同時刻F0 RMSE (Hz) |
|---|---:|---:|---:|---:|---:|---:|
| /a/ | 147.69 | 151.35 | +3.66 | +2.48% | 17/18 | 1.85 |
| /i/ | 148.71 | 146.55 | −2.16 | −1.45% | 16/16 | 6.93 |
| /u/ | 145.83 | 148.16 | +2.32 | +1.59% | 20/20 | 3.06 |

差の符号は再合成−元音声。中央値差は各音声の有効フレーム別に求めた中央値同士の差、
RMSEは両方で有声だった**同じ時刻**のフレーム差から求めた。
特に /a/ は有効フレーム集合が異なるため、中央値差とRMSEは同じ誤差指標ではない。
共有有声フレームのMAEは /a/ 1.54 Hz、/i/ 5.31 Hz、/u/ 2.45 Hzだった。
F0の有効数・SDは、原音が /a/ 18・4.20 Hz、/i/ 16・9.42 Hz、/u/ 20・3.14 Hz、
再合成が /a/ 17・4.51 Hz、/i/ 16・11.97 Hz、/u/ 20・3.60 Hzで、全母音のF0が基準を通過した。

F0は採用できたが、従来のF2/F3の欠測・不安定性は残る。
したがって**完全な4次元の目標ベクトルは、主設定では依然として3母音とも未確定**である。
今後EMAのみを操作する場合、F0への応答を測ることとF0を制御できることは区別する。
Decoderのpitch条件を固定したままのF0可制御性は未評価である。

追加保存物：

- `*_acoustics.csv/npz`：明示的な `[F0,F1,F2,F3]` のraw値・採用値・有効判定。CSVには棄却理由も保存。
- `acoustic_comparison.csv`：特徴ごとの中央値差・相対差・共有フレームMAE/RMSE。F0はcents差も保存。
- `summary.json`：schema version 2。`feature_names` と4要素の `baseline_target_hz` を保存。
  従来の3要素の代表値は `baseline_formant_target_hz` として保持。
- `f0.png`：原音・再合成のF0軌跡と、固定したSPARC pitch入力を別パネルで表示。
- `environment.json`、`provenance.json`、`config.yaml`：追加版の環境・ソースハッシュ・実行設定。

[/a/ F0図](../outputs/feedback_experiment1_f0/a/f0.png) /
[/i/ F0図](../outputs/feedback_experiment1_f0/i/f0.png) /
[/u/ F0図](../outputs/feedback_experiment1_f0/u/f0.png) /
[追加版の数値結果](../outputs/feedback_experiment1_f0/summary.json)

有効判定による色分けをせず、保存済みのF1–F3推定値をすべて青で示す図も追加した。
左列が元音声、右列が再合成音声で、同じフォルマントの左右は共通の縦軸にしている。
品質基準で棄却された点も表示する。数値自体が欠測（NaN）の箇所は描画せず、補完しない。
既存の `formants.png` と品質判定・数値結果は変更していない。

![全推定値を青で表示した /a/ のフォルマント](../outputs/feedback_experiment1_f0/a/formants_raw.png)

![全推定値を青で表示した /i/ のフォルマント](../outputs/feedback_experiment1_f0/i/formants_raw.png)

![全推定値を青で表示した /u/ のフォルマント](../outputs/feedback_experiment1_f0/u/formants_raw.png)

この追加図だけを再生成するコマンド（SPARCの再実行は不要）：

```sh
MPLCONFIGDIR="$PWD/.cache/matplotlib" ~/.local/share/sparc/venv/bin/python \
  -m experiments.feedback.plot_raw_formants outputs/feedback_experiment1_f0 --overwrite
```

再実行コマンド（既存ディレクトリへの上書きは拒否するため、新しい出力先を指定する）：

```sh
bash scripts/experiment1.sh --output outputs/feedback_experiment1_f0_new
```

自動テストは**13件成功**。既知F0=125 Hzの合成音による推定確認に加え、
F0とフォルマントの採否の独立性、共有有声時刻での誤差、4次元の列順、無声時の欠測処理を検証した。
保存物を読み直し、3母音すべてのSPARC条件と再合成波形のサンプル値が初回実行と完全一致すること、
4次元配列と欠測処理、ソースハッシュの一致も確認した（`validation.json`）。

## 初回のF1–F3解析記録

**SPARCによる /a/・/i/・/u/ の12次元EMA抽出と再合成は成功した。
同じcodeから2回生成した音声は、各母音でビット単位で一致した。
ただし、標準のPraat設定では3母音ともF1・F2・F3を揃えた目標特徴の品質基準を満たさなかった。**
実験2・3は実施していない。ヤコビアンの安定性、調音修正、誤差改善率は未評価である。

## 目的と範囲

`docs/feedback_test.md` の実験1に従い、音声→EMA→再合成→F1/F2/F3抽出を実データで検証する。
将来の目標音響特徴には、原音ではなく**基準EMAからの再合成音声**を使用する。
SPARCの再学習、EMAへの摂動、ヤコビアン推定、閉ループ制御は今回の範囲に含めない。

## モデル・実行環境

| 項目 | 使用値 |
|---|---|
| SPARC | `speech-articulatory-coding 0.1.0`、`multi`、CREPE full |
| リポジトリの基準コミット | `3df75e4266466ff4bd1c1502d86e65570ed4a178` |
| SPARC重みrevision | `2e6a07d4b35022d4366a2896a6293fe16aaed78a` |
| SPARC重みSHA-256 | `15c468c2703172b2423efa0459ff40695ef1f08f8312ad3577ee20d277b0485b` |
| WavLM config revision | `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c` |
| キャッシュ内WavLM safetensors revision | `07d9d3d8576fd3d718ee7b16b2b6242e9610d9af` |
| GPU | NVIDIA GeForce RTX 5090 |
| Python / PyTorch | 3.12.13 / 2.8.0+cu128 |
| CUDA runtime / cuDNN | 12.8 / 91002 |
| Transformers / NumPy / SciPy | 4.49.0 / 1.26.4 / 1.17.1 |
| Parselmouth / 内蔵Praat | 0.4.7 / 6.1.38 |
| seed / CPU threads | 20261030 / 4 |

既存環境とキャッシュ済み重みを利用し、オフラインで推論した。
Parselmouthのみ `.cache/feedback-deps/` に追加し、既存仮想環境は変更していない。
CUDAでは決定的アルゴリズムを有効にし、`CUBLAS_WORKSPACE_CONFIG=:4096:8` を指定した。
全依存バージョン、重みのパス・ハッシュ、実行ソースのハッシュは
[environment.json](../outputs/feedback_experiment1/environment.json) に保存した。
未コミットの研究実装は、基準コミットとソースハッシュの両方で識別する。

## 入力と解析区間

既存の `datasets/rtmridb_vowels/s1_mu1_wav/{a,i,u}.wav` を使用した。
ローカルの `manifest.json` にある `subject=s1`、`file=s1_4_mu1`、母音ラベルを確認し、
各WAVのSHA-256がmanifestと一致することを実行時に検証した。
公式付属の文章音声を持続母音とみなすことはしていない。

| 母音 | 元WAV長 (s) | 指定解析区間 (WAV秒) | 解析フレーム数 | 原音の有声率 | 再合成の有声率 |
|---|---:|---|---:|---:|---:|
| /a/ | 1.160998 | 0.48–0.66 | 18 | 100% | 94.4% |
| /i/ | 0.998458 | 0.48–0.64 | 16 | 100% | 100% |
| /u/ | 0.928798 | 0.36–0.56 | 20 | 100% | 100% |

全入力は44.1 kHz・モノラル。AAC付きMP4からPCM16へ変換したWAVであり、未圧縮の原収録ではない。
秒数の原点は**各WAVの先頭**で、データベースの元動画上の開始時刻は再適用していない。
原音の有声判定とフォルマント軌跡を調べ、無声部分を避けて上記区間を選んだ。
長い持続母音ではなく、有効な有声区間は短い。音響特徴の変動も残っている。
母音ラベルは既存manifestに基づき、話者の健常性を独立に検証したわけではない。

## 実装と品質基準

- `experiments/feedback/adapter.py`：encode/decode、時間計測、code全項目の不変性、2回decodeの一致確認。
- `experiments/feedback/formants.py`：Praat Burg法、有声判定、周波数・帯域幅・局所変動の品質管理。
- `experiments/feedback/run.py`：YAML・ユーザーWAV指定、出典照合、数値・音声・環境の保存。
- `experiments/feedback/visualization.py`：波形・スペクトログラム・12次元EMA・フォルマント軌跡。
- `configs/feedback_experiment1.yaml`、`scripts/experiment1.sh`、`requirements-feedback.txt`：再実行設定。
- `tests/test_feedback_experiment1.py`：重み不要の8件の検証。

モデルの前後文脈を保持するためWAV全体をencode/decodeし、音響統計だけを指定区間に限定した。
EMA・pitch・loudness・spk_emb等には変更を加えていない。
実際のAPI、波形正規化、Decoderの条件入力と時間依存性は [README](../README.md) に記録した。

主解析はBurg法の上限5000 Hz、最大5フォルマント、時間刻み10 ms、窓長25 ms、
pre-emphasis 50 Hz。Praatの窓の実長はこの設定で50 msとなる。
設定の意味は [Praat公式マニュアル](https://www.fon.hum.uva.nl/praat/manual/Sound__To_Formant__burg____.html) と
[Parselmouth API](https://parselmouth.readthedocs.io/en/stable/api_reference.html) を確認した。

有声判定には自己相関pitch（60–400 Hz、voicing threshold 0.45）を用いた。
F1/F2/F3の許容範囲はそれぞれ150–1300 / 400–3500 / 1200–4800 Hz、
最大帯域幅は400 / 600 / 800 Hz。順序も確認する。
前後2フレームを含む局所中央値から150 / 250 / 300 Hzを超えて外れる値を除外した。
これらは今回設定した研究用の基準であり、医学的な正常範囲を表すものではない。

各フォルマントは有効10フレーム以上、有効率80%以上、変動係数SD/中央値≤0.15を満たす場合に採用する。
3次元目標には、さらに3フォルマントが同時に有効なフレームが80%以上あることを要求する。
統計は各フォルマントの有効フレーム別に集計するため、中央値の使用フレーム集合は必ずしも同一ではない。
不採用値は補完せず、raw値・帯域幅・有効マスク・棄却理由を保存した。
目標値はJSONで `null`、フレームごとの不採用値はCSV/NPZで `NaN` とする。
解析そのものが失敗した場合も専用のエラーJSONを保存して例外を返す。

## 実測結果

### EMAと再合成

| 母音 | EMA形状 | pitch長 | loudness長 | ft_len | 再合成長 (s) | encode (s) | 初回decode (s) | 2回目decode (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| /a/ | 57×12 | 58 | 59 | 58 | 1.14 | 0.9721 | 0.0465 | 0.0113 |
| /i/ | 49×12 | 50 | 50 | 50 | 0.98 | 0.0731 | 0.0213 | 0.0114 |
| /u/ | 46×12 | 46 | 47 | 46 | 0.92 | 0.0493 | 0.0203 | 0.0112 |

モデルロードは8.4745秒。上記はCUDA同期を含む実測で、ファイル保存・Praat解析・作図は含まない。
初回の遅延やGPU上の他プロセスの影響を含み、性能ベンチマークとしては扱わない。
再合成は16 kHz・モノラル・float32 WAVとして保存した。
`ft_len`とEMA長は一致しない例があり、Decoderが条件を最短長に揃えるため出力は入力より短い。

3母音すべてで、decode前後のcode全項目が不変だった。
各母音の2回の出力差は最大絶対差=0、RMS差=0で、ビット単位で一致した。
これは同じプロセス・同じデバイス内での再現性の確認である。

CPUでも /a/ を実行し、encode 10.0613秒、初回decode 0.1104秒、2回目decode 0.0888秒で完了した。
CPU内の2回のdecodeも完全一致した。CPUとGPU間の同一性は保証しない。
CPUのF1/F3は553.06 / 1592.47 Hzで、GPUと小さな数値差があった。
CPU結果は `outputs/feedback_experiment1_cpu/` に保存した。

### フォルマント（主解析：上限5000 Hz）

下表は**品質基準を満たしたフォルマントの中央値**（Hz）。`—` は目標値として不採用。
原音の値は診断用で、目標値には用いていない。

| 母音 | 音声 | F1 | F2 | F3 | F1/F2/F3有効フレーム | 同時有効率 |
|---|---|---:|---:|---:|---|---:|
| /a/ | 原音 | 863.1 | — | 2674.6 | 18 / 1 / 18（全18） | 5.6% |
| /a/ | 再合成 | 553.0 | — | 1592.8 | 16 / 7 / 17（全18） | 38.9% |
| /i/ | 原音 | 292.2 | 2212.0 | 2858.1 | 16 / 16 / 16（全16） | 100% |
| /i/ | 再合成 | 410.1 | 2024.3 | — | 16 / 16 / 10（全16） | 62.5% |
| /u/ | 原音 | 360.6 | — | 2511.0 | 20 / 0 / 20（全20） | 0% |
| /u/ | 再合成 | 376.3 | — | 2120.2 | 20 / 3 / 17（全20） | 5.0% |

不採用の主因は /a/・/u/ のF2、/i/ の再合成F3の有効フレーム不足だった。
再合成の有効部分だけの中央値は /a/ F2=1430.2 Hz（7/18）、/i/ F3=2668.5 Hz（10/16）、
/u/ F2=1886.8 Hz（3/20）だったが、**これらを目標値として採用していない**。
再合成の採用部分のSDは /a/ F1/F3=29.7/39.0 Hz、/i/ F1/F2=15.3/51.3 Hz、
/u/ F1/F3=49.8/90.7 Hzだった。

品質基準の通過は、正しい共鳴の番号が同定されたことを保証しない。
とくに /a/ の原音F3と再合成F3には大きな違いがあり、合成の変化とLPCのピーク番号の変化を
この結果だけで分離できない。/u/ F1にも区間内の傾きが残る。
**3母音とも、主設定で完全な3次元目標ベクトルは未確定**という結論である。

### フォルマント上限への感度

同じ音声・同じ区間・同じ品質基準で、上限4500/5500 Hzも解析した。
再合成音声について、採用可能な中央値のみを示す。

| 母音 | 上限 (Hz) | F1 | F2 | F3 | 3次元目標の品質基準 |
|---|---:|---:|---:|---:|---|
| /a/ | 4500 | 524.0 | 1308.9 | 1610.2 | 通過（同時有効88.9%） |
| /a/ | 5000 | 553.0 | — | 1592.8 | 不通過 |
| /a/ | 5500 | — | — | — | 不通過 |
| /i/ | 4500 | 414.5 | 2019.1 | — | 不通過 |
| /i/ | 5000 | 410.1 | 2024.3 | — | 不通過 |
| /i/ | 5500 | 410.5 | 2025.4 | 2764.8 | 通過（同時有効100%） |
| /u/ | 4500 | 367.7 | — | 2119.2 | 不通過 |
| /u/ | 5000 | 376.3 | — | 2120.2 | 不通過 |
| /u/ | 5500 | 400.5 | — | 2949.9 | 不通過 |

/a/・/i/ には設定によって基準を通過する候補がある一方、/u/ はどの上限でも不通過だった。
/u/ のF3などは設定によって大きく変わるため、時間方向の安定性だけでは不十分である。
この比較は探索的な品質調査であり、成功した設定へ主解析を置き換えてはいない。

## 保存物と図

主結果は [outputs/feedback_experiment1/](../outputs/feedback_experiment1/) にある。

- `config.yaml`、`environment.json`、`provenance.json`、`summary.json`：設定・環境・出典・数値集計。
- 各母音の `baseline_code.npz`：EMAとpitch/loudness/periodicity/pitch_stats/spk_emb/ft_lenを保存。
- `ema.csv`：公称時刻と12変数。長さは実際のEMA長を使用。
- `original.wav`、`baseline.wav` と `*_interval.wav`：全音声と指定区間。
- `*_formants.csv/npz`：raw F1–F3、帯域幅、F0、有効判定、棄却理由。上限比較のファイルも保存。
- `audio.png`：原音と再合成の波形・スペクトログラム。帯は解析区間。
- `ema.png`：12次元の基準EMA。時間は `frame_index/50` の公称値で、厳密な遅延補正はしていない。
- `formants.png`：灰色がraw値、青色が有効値、緑の破線が採用中央値。欠測・棄却は補間しない。
- `run.log`、`tests.log`、`validation.json`：実行記録と保存物の整合性検証。

[/a/ 波形・スペクトログラム](../outputs/feedback_experiment1/a/audio.png) /
[/a/ EMA](../outputs/feedback_experiment1/a/ema.png) /
[/a/ フォルマント](../outputs/feedback_experiment1/a/formants.png) /
[/i/ フォルマント](../outputs/feedback_experiment1/i/formants.png) /
[/u/ フォルマント](../outputs/feedback_experiment1/u/formants.png)

`outputs/` は既存の `.gitignore` 対象であり、結果はこの作業環境内に保存されている。

## 実行コマンドと検証

リポジトリルートから、まず /a/ を確認してから3母音を実行した。
最終保存物は、作図とエラー記録を整えた最終コードで再実行した結果である。

```sh
# 追加依存（既存SPARC環境を利用）
uv pip install --python ~/.local/share/sparc/venv/bin/python \
  --target .cache/feedback-deps --cache-dir /tmp/sparc-feedback-uv \
  --link-mode copy --no-deps -r requirements-feedback.txt

bash scripts/experiment1.sh --vowel a --output outputs/feedback_experiment1_a
bash scripts/experiment1.sh
bash scripts/experiment1.sh --device cpu --vowel a --output outputs/feedback_experiment1_cpu

PYTHONPATH="$PWD/.cache/feedback-deps" MPLCONFIGDIR="$PWD/.cache/matplotlib" \
  ~/.local/share/sparc/venv/bin/python -m unittest discover -s tests -v
```

再実行時は `--output outputs/experiment1_new` 等で空の出力先を指定する。
既存結果の上書きは拒否する。ユーザーWAVへの変更例はREADMEにある。

自動テストは **11件成功**（新規8件、既存3件）。
重みをロードせず、既知共鳴の合成音からのフォルマント推定、無声雑音・欠測・帯域幅・局所外れ値、
無音・非有限値・区間外のエラー、エラーJSON保存、decode再現性・条件変更検出を検証した。
これらは実SPARC実験とは区別する。

実SPARCでは、GPUで3母音、CPUで /a/ を実行した。
さらに保存したNPZを `allow_pickle=False` で読み直し、EMA形状、WAVの長さ・サンプリング周波数、
原音コピーの一致、不採用値のNaN、図の生成、実行ソースハッシュの一致を検証した。

## 技術的課題と次の実験

1. **母音データの長さ・音質**：現データの有声区間は短く、元データがAAC経由でもある。
   より長く定常的な /a/・/i/・/u/ のWAVを用意し、現在の素材との違いを評価する。
2. **共鳴同定と設定依存性**：特にF2/F3の帯域幅とピーク番号が不安定。
   スペクトル包絡と軌跡を確認し、話者に適した解析条件を基準音声で決め、摂動後も同じ条件を固定する。
   今回の不採用値を原音や任意値で補わない。
3. **文脈・時間対応**：EMAと他条件の長さが異なり、全体的な再合成差も存在する。
   今後も全体のcodeを保存し、同じデコード長・同じ解析区間・固定した非EMA条件で比較する。
4. **実験2への移行条件**：まず再合成基準音声から安定した目標特徴を取得する。
   その後に有限差分を実装し、音響抽出の揺らぎとEMA摂動への応答を区別する。
   今回はヤコビアンの安定性や調音修正による誤差改善について結論を出していない。
