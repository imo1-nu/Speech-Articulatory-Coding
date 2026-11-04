# JSUT抽出・トークン化・合成の実行手順

[概要](../README.md) / [実験一覧と結果](../reports/experiments.md) / [PPG制御](ppg-control.md)

コマンドはリポジトリのルートで実行する。以下の`outputs/`はローカル成果物で、Gitには含めない。
主な研究条件は40 ms・128コード・50 Hzだが、CLIの既存の既定値は変更していない。

## WavLM特徴量を目標とするRVQ制御

```sh
# 既存5発話：オフラインと60ms遅延FB、PPGとWavLMの両方
bash scripts/wavlm_feedback_experiment.sh
# 個別の方式・発話を実行する例
bash scripts/wavlm_feedback_experiment.sh --objective wavlm --mode delayed --horizon-ms 400 \
  --utterances BASIC5000_0050 --output-dir outputs/jsut_wavlm_feedback_single
# 同じコード・引数・入力で未完了の比較を再開する場合
bash scripts/wavlm_feedback_experiment.sh --resume
```

WavLM-largeのrevision `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c`、`hidden_states[9]`の1024次元を使う。
SPARCと同じ左右160サンプルのゼロ埋めと、float64母平均・母標準偏差による波形標準化後のfloat32入力。
10 Hz平滑化と線形射影は適用しない。主距離は各フレームL2正規化後のcosine距離、raw MSEも診断用に保存する。
候補と目標は同じ16 kHz波形読み込みを用い、フレーム時刻はCNNの400サンプル受容野と320サンプルstrideから
`(320*i+39.5)/16000`で対応付ける。PPGの100 Hz・5 msオフセットと混同しない。

元音声を目標とし、既存RVQ8の同じ5発話で比較する。全8段を編集し、罰則は1段目の遷移NLLだけ。
オフラインはPPG/WavLMそれぞれλ=0/0.03/0.1、最大2パス×8段×各段24時刻。
遅延制御は両目標について編集なし・補正なし・FBあり、λ=0.03、既定400 msの探索窓、20 ms更新・60 ms観測遅延。
各段で現在位置と誤差最大の未来1位置を1パス探索し、全方式で12近傍＋4広域候補、バッチ4を使う。
探索窓は `--horizon-ms 200|400|800` で選べる。既定比較は60発話条件になる。

観測波形をt−60 msで切り、未観測60 msは発行済みコマンドの予測音声を文脈に使う。
WavLMは利用可能な接頭辞にだけ適用し、到着した特徴量は保持する。
観測−発行時予測の正規化特徴量差をEMA更新し、L2ノルム上限1・ゲイン0.5で予測を補正して再正規化する。
更新率は100 msあたり0.25（50 Hzで新規1フレームなら `1-0.75**0.2`）。PPGは既存のlog確率補正を使う。
WavLM観測は20 ms波形から、PPG観測は40 ms波形から可能で、最初の到着時刻はそれぞれt=80/100 ms。
終端の観測補完は解析専用。非因果モデルを接頭辞へ適用する近似で、発話全体の特徴量と同じではない。

特徴量は波形ハッシュでキャッシュするが、候補採用前の単独検証と保存WAVの検証は再抽出する。
WavLMやPPGの過去の内部状態を、変更された候補の特徴量として再利用しない。
出力は `outputs/jsut_wavlm_feedback/`。試聴HTML、音声指標CSV、特徴量CSV、各条件のトークン・軌道・特徴量時刻・
遅延FBバイアス・探索履歴・計算時間と評価数を保存する。
MCD/STOI/F0/VUVは元軌道合成参照、CERはWhisper large-v3とJSUT正解文。
目標距離の尺度が異なるため、同じλを両方式の同じ制約強度とは解釈しない。
既存出力は上書き保護し、`--resume` はソース・引数・依存成果物が一致する場合だけ完了条件を再利用する。
WavLMとPPGのモデルは検索中に保持し、Whisperは探索終了後にロードする。

## RVQの1段目に遷移ソフト制約を加えた局所探索

```sh
bash scripts/rvq_transition_search.sh
# 重みや保存先を変更する場合
bash scripts/rvq_transition_search.sh --weights 0 0.01 0.03 0.1 --output-dir outputs/jsut_rvq_first_stage_transition_v2
```

既存の8段RVQの全段を編集対象とし、元音声PPGとの平均JSに、**1段目だけ**の平均遷移NLLを加える。
目的関数は `mean_JS + λ * mean(-log P(q[t,1] | q[t-1,1])))`。
2〜8段には遷移罰則を加えない。初期値は保存済みの8段RVQトークンで、各候補は1つの時刻・段のIDだけを変更する。
1段目を変更しても他段は自動で再量子化しない。全段を1→8の順に編集する。
これは全発話の元音声PPGを参照するオフライン実験で、観測遅延を設けたオンラインFBではない。

RVQ専用の1段目遷移表を訓練4500発話だけから推定し、加算平滑化α=1、自己遷移あり、発話間遷移なし。
検証・テストの遷移は学習に使わず、別途NLL・未観測率を診断する。
旧VQの遷移表は使用しない。コードブック・AE・生成器・音声・PPG・ASR版を照合する。

λ=0/0.01/0.03/0.1で同じ探索予算規則を使う。
最大2パス×8段×各段24時刻、各時刻の候補は段ごとの潜在コードベクトルによる12近傍＋4広域候補。
PPG誤差で優先時刻を選び、候補はバッチ4で評価し、採用前に単独再評価する。
各候補は目的関数を1e-5超改善する場合だけ採用し、1パスで採用がなければ終了。
2〜8段では遷移NLLが定数なのでJSのみの採用判定と等価。
同じ探索予算規則でも、状態・候補・再検証回数は採用経路によって異なる。

出力は `outputs/jsut_rvq_first_stage_transition/`。
既存のテスト5発話で元音声・元軌道合成・RVQ8編集前・各λの編集後を試聴できる。
F0・音量・話者・ゲインは固定。音響指標は元軌道合成参照、CERはWhisper large-v3とJSUT正解文。
`prior_diagnostics.json` に遷移の汎化診断、`summary.csv` と `search_aggregates.csv` に探索結果、
`audio_metrics.csv` と `audio_aggregates.csv` に音声指標、manifestに軌道MSE・差分MSE等を保存する。
速度・加速度RMSは訓練SDで標準化した軌道の有限差分で、物理単位や生理学的上限ではない。
発話ごとの `lambda_*_trace.jsonl` に探索経過、`lambda_*_result.npz` に初期／編集トークン・PPG・編集軌道を保存する。
既存出力先は上書きしないため、再実行時は新しい `--output-dir` を指定する。

## 固定した連続AEへのRVQ

```sh
uv pip install --python "$HOME/.local/share/sparc/venv/bin/python" -r requirements-rvq.txt
bash scripts/train_trajectory_rvq.sh
bash scripts/compare_trajectory_rvq.sh --output-dir outputs/jsut_rvq/comparison_verified
```

学習済み `outputs/jsut_reconstruction_ablation/continuous_ae` のエンコーダとデコーダを固定し、
訓練4500発話の潜在表現に各段128中心のMiniBatchKMeansを順番に適用する。
次段は前段までの量子化残差を学習する。seed=42+段番号、batch_size=4096、max_iter=100、n_init=3。
検証・テストデータはコードブックの学習に使わず、同じ8段モデルの1/2/4/8段の接頭部を比較する。
今回はAEとの同時学習、速度損失、音声損失、遷移制約は追加しない。

モデルは `outputs/jsut_rvq/model`、最終レポートは `outputs/jsut_rvq/comparison_verified/index.html`。
最初の比較 `outputs/jsut_rvq/comparison` も保存している。
最終比較はテスト250発話のトークンIDを `tokens/` に保存し、各段のコードベクトルを足し合わせて
デコーダへ渡す。連続潜在表現を合成へ直接渡す経路は使わない。
音声は従来と同じ5発話、元音声・元軌道合成・VQ128・VQ512・連続AE・RVQ4条件を試聴できる。
既存音声の指標は波形ハッシュ・AE・生成器・Whisper版と設定を照合して再利用する。
F0・音量・話者・ゲインと音響指標の参照は以前の比較と同一。
元軌道合成PPGから選んだ既存のk/ky候補9区間についても全条件の音声クリップを作る。

RVQ1/2/4/8段の公称ビットレートは350/700/1400/2800 bit/s（50 Hz・各段7 bit）。
VQ128は350、VQ512は450 bit/sなので、段数の違いは同一ビットレートの比較ではない。
VQはエンコーダ・デコーダも量子化と同時学習しており、固定AEとの学習方法の差も含む。
コード使用数・perplexity・各種MSE・速度／帯域エネルギー比はmanifest、
CER・MCD・STOI・F0/VUVと認識文はCSVに保存する。
既存出力先は上書きしない。再学習には新しい `--output-dir`、比較時は対応する `--model-dir` と新しい `--output-dir` を指定する。

## 再構成の切り分け：VQ128・VQ512・連続AE

```sh
# 初回は連続AEを学習し、その後に比較する
bash scripts/compare_trajectory_reconstruction.sh

# AEの学習だけを実行する場合
bash scripts/train_continuous_ae.sh
```

既存の40 ms・128コードと512コードを再学習せず、量子化なしの連続AEを別途学習する。
AEは24→128→64→16→64→128→24の同じMLP、同じ4500/250/250分割と訓練正規化、
seed・Adam・学習率・バッチサイズ・最大epoch・検証MSEによる早期停止を使う。
VQの量子化を推論時だけ外す方法は使わない。AEは連続潜在表現なので、同じビットレートでの比較ではない。

出力は `outputs/jsut_reconstruction_ablation/continuous_ae/` と `comparison/`。
学習済みAEは参照チェックポイント・分割・正規化を確認して再利用する。
未完了の学習先や既存の比較先は上書きしない。再評価は新しい `--output-dir`、
新規学習は新しい `--ae-dir` を指定する。学習専用スクリプトのモデル保存先オプションは `--output-dir`。

軌道評価はテスト250発話の重複平均後の標準化MSE・差分MSE・速度エネルギー比・6〜10 Hz帯域比。
音声評価は既存の試聴5発話を使い、元軌道合成・VQ128・VQ512・連続AEで同じF0・音量・話者情報・ゲインを使う。
元軌道合成とVQ128音声が以前の保存音声を再現することも確認する。
MCD・STOI・F0/VUVの参照は元軌道からの合成音声で、CERはWhisper large-v3とJSUT正解文による。
従来の元音声参照の音響指標とは、参照が異なることに注意する。

元軌道合成PPGで `k` / `ky` が最大の連続区間から、各発話最大3個を長い順に選び、
音声クリップ・共通スケールのスペクトログラム・訓練SDで標準化したTDX/TDYと速度を保存する。
この候補区間は正解音素境界ではなく、PPG確率はカ行の正解率そのものではない。
`comparison/index.html` で全体と候補区間を試聴でき、CSV・NPZ・manifestに数値とモデルハッシュを保存する。
学習目標はまず軌道MSEのみとし、この比較では速度損失・音声損失・RVQはまだ導入しない。

## JSUT からの調音軌道・ピッチ抽出

展開済みの JSUT コーパスから、発話ごとに調音軌道とピッチを抽出する。
環境はREADMEで説明する `bash scripts/setup_resynthesis.sh` と共通。
入力ディレクトリには `basic5000/wav/*.wav` などを含む
`datasets/jsut/jsut_ver1.1` を指定する。ZIP の展開・ダウンロードは行わない。
JSUT の構成については [公式配布ページ](https://sites.google.com/site/shinnosuketakamichi/publication/jsut/) も参照。

保存済みの研究パイプラインと同じ入力先を用意する場合は、
`bash scripts/extract_jsut.sh --subsets basic5000 --output-dir outputs/jsut_ema` を使用する。
抽出CLI単体の既定出力先は `outputs/jsut` のままで、後段の既定入力 `outputs/jsut_ema` とは異なる。

```sh
# まず basic5000 の先頭3発話で確認
bash scripts/extract_jsut.sh --subsets basic5000 --limit 3

# 展開済みの全サブセットを処理
bash scripts/extract_jsut.sh

# 入出力先・サブセット・GPUを指定（CPUは --device cpu）
bash scripts/extract_jsut.sh \
    --jsut-dir /path/to/jsut_ver1.1 \
    --output-dir outputs/jsut \
    --subsets basic5000 travel1000 --device cuda:0
```

`--device auto` が既定値で、CUDA が利用可能なら GPU、それ以外は CPU を使用する。
SPARC の `feature_extraction` 設定（WavLM による調音推定と CREPE tiny による
ピッチ抽出）を使い、再合成や話者埋め込みの計算は行わない。
初回は Hugging Face からモデルを取得するためインターネット接続が必要。
48 kHz 等の入力音声は内部で 16 kHz にリサンプリングし、ステレオはモノラル化する。
発話は一つずつ処理し、長さの異なる発話のバッチパディングによる影響を避ける。

出力は `outputs/jsut/<subset>/<発話ID>.npz`。全ての時系列を末尾の最短フレーム数に
揃え、50 Hz（20 ms 間隔）で保存する。元の音声と特徴量のフレーム数の関係から、
末尾の長さは入力時間とわずかに異なることがある。

| キー | 形状・内容 |
|---|---|
| `ema` | `(T, 12)` float32 の推定調音軌道 |
| `pitch` | `(T, 1)` float32、Hz 単位のピッチ |
| `periodicity` | `(T, 1)` float32、ピッチ推定の有声性の目安 |
| `loudness` | `(T, 1)` float32、SPARC が正規化した波形の振幅に基づく音量 |
| `time` | `(T,)`、先頭フレームを 0 秒とした公称時刻 |
| `frame_rate`, `ft_len` | 50 Hz、保存したフレーム数 `T` |
| `channels` | `TDX, TDY, TBX, TBY, TTX, TTY, LIX, LIY, ULX, ULY, LLX, LLY` |
| `source`, `text` | JSUT ルートからの音声相対パス、対応する書き起こし（存在する場合） |
| `source_sample_rate`, `duration_seconds` | 元の音声のサンプリング周波数・秒数 |
| `model`, `source_size`, `source_mtime_ns` | 抽出モデル名、再開時の入力変更検出用情報 |

`ema` は音声から推定したモデル座標であり、実測 EMA や mm 単位の位置ではない。
無声区間でも `pitch` は値を持つため、必要に応じて `periodicity` でマスクする。
`time` は音声に対する厳密なイベント境界や音素アラインメントを表さない。

```python
import numpy as np

with np.load("outputs/jsut/basic5000/BASIC5000_0001.npz", allow_pickle=False) as data:
    ema = data["ema"]              # (T, 12)
    pitch_hz = data["pitch"][:, 0] # (T,)
    time_seconds = data["time"]
    voiced = data["periodicity"][:, 0] >= 0.5  # 用途に応じて閾値を調整
    pitch_hz = np.where(voiced, pitch_hz, np.nan)
```

再実行時は、入力ファイルのサイズ・更新日時が一致し、特徴量が正常に読み込める発話を
スキップする。破損・未完了の出力は再処理し、強制的に全発話を再計算する場合は
`--overwrite` を付ける。処理対象の結果・書き起こし・エラーは実行ごとに
`manifest.jsonl` に書き出す。0.5 秒未満・無音・破損音声などで失敗しても残りを続行し、
失敗があれば終了コード 1 を返す。全件が既存出力で揃っていればモデルをロードしない。

## 調音軌道の VQ-VAE トークン化・遷移分析

抽出済みの `outputs/jsut_ema/basic5000/*.npz` を使用し、調音軌道12次元だけを
VQ-VAE でトークン化する。ピッチ・有声性・音量は補助情報として出力に引き継ぐ。
入力元が `outputs/jsut` の場合は `train --features-dir outputs/jsut` を指定する。
上記の再合成・抽出と同じ Python 環境で実行でき、追加のモデルダウンロードは不要。

```sh
# 学習・検証・テスト（既定は basic5000 全発話）
bash scripts/model_jsut_tokens.sh train \
    --features-dir outputs/jsut_ema --run-dir outputs/jsut_tokens

# 最良モデルで全対象発話をトークン化
bash scripts/model_jsut_tokens.sh tokenize --run-dir outputs/jsut_tokens

# 学習発話から直接遷移を推定し、検証・テスト発話で評価
bash scripts/model_jsut_tokens.sh transitions --run-dir outputs/jsut_tokens
```

少量の実音声で一連の処理を試す場合は、別の出力先を使う。

```sh
bash scripts/model_jsut_tokens.sh train \
    --features-dir outputs/jsut_ema --run-dir outputs/jsut_tokens_trial \
    --limit 30 --epochs 10 --codebook-size 32 --device cpu
bash scripts/model_jsut_tokens.sh tokenize --run-dir outputs/jsut_tokens_trial
bash scripts/model_jsut_tokens.sh transitions --run-dir outputs/jsut_tokens_trial
```

### 学習・時間窓

- 各フレームを中心に前後2フレーム、計5フレーム（100 ms相当）を入力し、
  20 ms ごとに1トークンを出力する。端は同じ発話の端の値で補い、発話をまたがない。
  将来2フレームを参照するオフライン分析であり、因果的なオンライン推定ではない。
- 発話単位で seed 42 により90%・5%・5%に分割する。小規模実験では検証・テストを
  それぞれ最低1発話確保する。正規化は学習集合のチャンネル別平均・標準偏差を使う。
- エンコーダは `60 → 128 → 64 → 16`、デコーダは逆構成の MLP。
  コードブックは128個で、初期値には学習集合のエンコーダ出力をサンプリングして使う。
  再構成 MSE、コードブック損失、重み0.25の commitment 損失で学習する。
- 既定は Adam、学習率0.001、バッチ512、最大100 epoch。
  検証の正規化再構成 MSE が10 epoch改善しなければ終了し、最良モデルを採用する。
- `--subsets basic5000 travel1000`、`--codebook-size`、`--latent-dim`、`--epochs`、
  `--patience`、`--learning-rate`、`--commitment`、`--batch-size`、`--seed` を変更できる。
  `train`・`tokenize` は `--device auto` が既定で、CPU／CUDAを自動選択する。
  各コマンドの `--threads` はCPUのスレッド数（既定は最大4）を指定する。

### 保存物・評価

`--run-dir` は学習実行ごとに空のディレクトリを指定する。異なるモデルの出力が
混ざるのを防ぐため、既存の学習結果の上書き・学習再開は行わない。

| 保存物 | 内容 |
|---|---|
| `config.json`, `splits.json` | 設定、対象特徴ファイルと変更検出情報、発話の分割表 |
| `best.pt`, `normalization.npz` | 最良モデル、学習集合の平均・標準偏差 |
| `history.jsonl`, `training_metrics.json` | epoch別の損失・コード使用率、最良epochのテスト評価 |
| `tokens/<subset>/<発話ID>.npz` | 整数 `tokens: (T,)`、元の `time: (T,)`、50 Hz、分割名、補助特徴、モデルのハッシュ |
| `reconstruction_metrics.json` | 分割別の再構成誤差、平均軌道のベースライン、コード使用率・perplexity |
| `representatives.json`, `codebook_trajectories.npz` | 各コードに最も近い学習入力窓の所在、実際の窓、デコードした窓、使用頻度 |
| `transitions.npz`, `transition_metrics.json` | 学習集合の遷移カウント・確率、未使用コード、評価結果 |
| `plots/` | 各分割の再構成例、代表軌道、遷移行列・コード使用頻度のPNG |

トークンファイルの `reconstruction: (T, 12)` は、各トークンから復号した5フレーム窓を
重ね合わせて平均し、元のモデル座標に戻した軌道。人工的なパディング位置は平均に含めない。
`rmse_per_channel` は元座標での RMSE、`normalized_mse` は学習集合の標準偏差で
標準化したフレーム再構成 MSE。学習時の窓 MSE と区別して出力する。

```python
import numpy as np

with np.load("outputs/jsut_tokens/tokens/basic5000/BASIC5000_0001.npz",
             allow_pickle=False) as data:
    tokens = data["tokens"]  # 整数ID、元の音声と同じ50 Hz
    time_seconds = data["time"]
    reconstructed_ema = data["reconstruction"]

with np.load("outputs/jsut_tokens/transitions.npz", allow_pickle=False) as model:
    probability = model["transition_probs"]  # [現在のトークン, 次のトークン]
    counts = model["transition_counts"]
```

遷移モデルはトークンを直接状態とする一次マルコフモデル。
学習発話内の `z[t] → z[t+1]` だけを数え、自己遷移も残す。
経験確率 `transition_probs_empirical` は出遷移がない行を0として保存する。
平滑化確率 `transition_probs` は `(counts[i,j] + alpha) / (sum_j counts[i,j] + K*alpha)`。
`transitions --alpha 1.0` が既定で、出遷移がない行は一様分布となる。
未使用コードと出遷移がないコードはJSONにも明記する。

検証・テストでは遷移あたりの負の対数尤度と perplexity を計算し、同じ学習集合から
同じ係数で平滑化した単純なトークン頻度モデルと比較する。自然対数を使い、発話先頭の
確率は両モデルとも評価に含めない。自己遷移と入力窓の重なりの影響も含む評価となる。
トークンが音素や調音ジェスチャーに対応するかは、代表軌道などで別途検討する。

トークン化・遷移分析時には特徴ファイルのサイズ・更新日時を確認し、遷移分析では
トークンに保存したモデルのSHA-256も照合する。入力を変更した場合は新しい学習実行を作る。

## テスト発話の音声を聴き比べる

VQ-VAE のテスト分割から発話を選び、元音声、抽出した調音軌道からの合成、
VQ再構成軌道からの合成を比較する。先に `tokenize` を実行しておく。

```sh
# 本学習モデルのテスト分割の先頭5発話を生成（CPU／CUDA自動選択）
bash scripts/compare_jsut_audio.sh --run-dir outputs/jsut_tokens

# 発話・出力先を指定（指定IDはテスト分割に含まれる必要がある）
bash scripts/compare_jsut_audio.sh --run-dir outputs/jsut_tokens \
    --utterances BASIC5000_0006 BASIC5000_0050 \
    --output-dir outputs/jsut_listening_selected --device cpu

# テスト全250発話を生成
bash scripts/compare_jsut_audio.sh --run-dir outputs/jsut_tokens \
    --limit 250 --output-dir outputs/jsut_listening_all
```

既定の出力先は `<run-dir>/audio_comparison`。
`index.html` をブラウザで開くと、書き起こし付きで3種類を個別に再生できる。
`comparison.wav` は「元音声 → 抽出軌道からの合成 → 再構成軌道からの合成」の順で、
間に0.5秒の無音を入れた連続比較用。
元音声は `--jsut-dir datasets/jsut/jsut_ver1.1`（既定）から取得する。

各発話のディレクトリには `original.wav`（元ファイルのコピー）、`original_ema.wav`、
`reconstructed_ema.wav`、`comparison.wav`、`conditions.npz` を保存する。
合成モデルは SPARC `multi`。2種類の合成には保存済みの同じピッチ・音量と、
元音声から保存済みperiodicityを使って求めた同じ話者埋め込みを与え、調音軌道だけを変える。
ピッチの再抽出は行わない。話者埋め込みの計算では抽出時と同じゼロパディングを用いる。

元音声のサンプリング周波数・長さ・音量は保存したまま、合成は16 kHzのPCM16とする。
合成2種類にはクリッピング防止の共通ゲインだけを適用し、個別の音量正規化は行わない。
連続比較では元音声を16 kHz・モノラルに変換し、必要なら連続音声全体に共通ゲインを適用する。
合成の末尾は特徴量のフレーム数に依存するため、元音声とわずかに長さが異なる場合がある。
条件・ゲイン・使用したモデルのハッシュ・ファイル一覧は `manifest.json` に保存する。
既存の試聴結果を保護するため、再実行時には空の `--output-dir` を指定する。

### 中央を重くした復号との比較

同じモデルとトークン列のまま、復号窓の平均を等重みから中央重み付きに変更して比較できる。

```sh
bash scripts/compare_jsut_audio.sh --run-dir outputs/jsut_tokens \
    --include-weighted --output-dir outputs/jsut_tokens/audio_comparison_weighted
```

従来の3種類に `weighted_ema.wav` を追加し、4種類を同じHTMLで再生する。
`weighted_ab.wav` は「等重み平均 → 中央重み付き平均」の順で0.5秒の無音を挟んだ比較用音声。
中央重み付きでは、復号窓の5フレームに `1, 2, 3, 2, 1` の重みを付ける。
各時刻で利用できる予測の重みの合計で割り、発話端も正規化する。
ピッチ・音量・話者埋め込み・合成時のゲインは3種類の合成すべてで共通にする。
既存モデル・トークン・保存済み再構成は変更せず、追加した軌道と重みを新しい出力先の
`conditions.npz` に保存する。通常の `tokenize` は引き続き等重み平均を使う。

### 時間窓20〜180 ms・重みあり／なしの比較

```sh
bash scripts/sweep_jsut_windows.sh --device cuda
```

20 ms刻みの9窓について、basic5000の同じ5,000発話を学習4,500・検証250・テスト250に
分けて比較する。各窓でVQ-VAEを1つ学習し、その同じトークン列を等重み平均と三角形重みで
復号するため、計18条件となる。コード数128、潜在次元16、seed 42、バッチ512、
最大100 epoch・patience 10は共通。モデル選択は窓ごとの検証再構成損失、条件の比較は
検証集合のフレーム再構成MSEを使う。モデルの入出力層だけが窓長に応じて変わる。

トークン出力は全条件で50 Hz。1〜9フレームの時間窓を使い、偶数窓では
前方 `(W-1)//2` フレーム、後方 `W//2` フレームとする。例えば40 msは現在・次フレーム、
80 msは前1・現在・後2フレーム。三角形重みは100 msで `1,2,3,2,1`、120 msで
`1,2,3,3,2,1`。20 msと40 msでは対称重みが等重みとなるため、2方式の出力は同一。

既定の結果は `outputs/jsut_window_sweep/` に保存する。
`index.html` では同じテスト5発話について9窓×2方式を聴き比べられる。
`summary.csv`・`summary.json` に検証／テストのフレームMSE、差分MSE、元軌道に対する
変化量比、6〜10 Hzの軌道成分エネルギー比、コード使用状況を保存する。
`window_comparison.png` は検証誤差と動きの保持量を表示する。
各モデル・ログ・2種類の再構成・試聴音声は窓ごとのサブディレクトリに残す。
比較用のトークン保存は検証・テスト発話が対象で、学習発話も必要な場合は各runで
`scripts/model_jsut_tokens.sh tokenize --run-dir <run>` を実行できる。

同じ設定で再実行すると学習完了モデルを再利用する。途中で中断した学習ディレクトリは
再開できないので、別の場所へ移してから再実行する。既存の `outputs/jsut_tokens` は変更しない。
GPU学習では窓をデバイス上にキャッシュする。単体実験でも
`model_jsut_tokens.sh train --window-ms 180 --cache-windows --run-dir <new-run>` を使える。
従来の100 msモデルも引き続き読み込める。

### 40 ms窓でトークンを25 Hzにする

入力の調音軌道は50 Hzのまま、40 msごとに1トークンを出すモデルを別に学習できる。

```sh
bash scripts/model_jsut_tokens.sh train \
    --features-dir outputs/jsut_ema \
    --run-dir outputs/jsut_40ms_25hz_comparison/model_25hz \
    --window-ms 40 --token-rate 25 --cache-windows --device cuda

# 既存の40 ms・50 Hzモデルと評価・試聴比較
bash scripts/compare_jsut_token_rates.sh --device cuda
```

`--token-rate` は25または50（従来の既定値）。25 Hzでは元の2フレームごとに窓を取り、
1トークンから50 Hzの軌道を2フレーム復元する。40 ms窓同士は重ならず、補間は行わない。
発話末尾が奇数フレームなら最後の入力を複製して窓を満たし、復元結果を元の長さで切る。
比較時のMSE・変化量は人工的な末尾フレームを除いた50 Hzの軌道で計算する。
40 msでは三角形重みが `1,1` で、25 Hzでは窓の重なりもないため、重みあり／なしは同じ。

出力NPZの `tokens` と `token_time` は25 Hz（長さ `ceil(T/2)`）、`reconstruction` と
`time` は50 Hz（長さ `T`）。`token_rate`、`token_stride`、`token_len`、元の `ft_len` も保存する。
従来の50 Hzトークンモデルは変更せず、そのまま読み込める。
遷移モデルの学習も25 Hzのトークン列を扱える。全発話のトークンが必要な場合は
`model_jsut_tokens.sh tokenize --run-dir outputs/jsut_40ms_25hz_comparison/model_25hz` を実行する。

比較スクリプトは既定で `outputs/jsut_window_sweep/window_040ms` を参照し、
学習設定・発話分割・特徴ファイル・正規化が一致することを確認する。
テスト250発話の評価と、同じ5発話の元音声・抽出軌道合成・50 Hzトークン合成・25 Hzトークン合成を
`outputs/jsut_40ms_25hz_comparison/index.html` にまとめる。
数値は `summary.csv`・`summary.json`、50 Hz→25 Hzの連続比較WAVは `listening/` に保存する。
`--reference-run`、`--run-dir`、`--output-dir`、`--audio-limit` で対象や保存先を指定できる。
表示するbpsはコードIDだけの固定長換算であり、ピッチ・音量・話者・モデル等の情報量は含まない。

### 50 Hz・20/40 ms窓でコード数を比較する

```sh
bash scripts/sweep_jsut_codebooks.sh --device cuda
```

64・128・256・512コードを20 ms・40 msの両方で比較する（計8条件）。
トークン出力は全て50 Hzで、同じbasic5000の5,000発話を学習4,500・検証250・テスト250に
分ける。潜在次元16、seed 42、バッチ512、Adamの学習率0.001、commitment重み0.25、
最大100 epoch・patience 10を固定し、入力窓とコード数だけを変える。
各モデルは検証の窓MSEで最良epochを採用し、条件の選択には検証フレームMSEを使う。
この2種類の窓では、等重み平均と左右対称の三角形重みは同じになる。

結果は `outputs/jsut_codebook_sweep/index.html` に保存する。
同じテスト5発話について、元音声・抽出軌道からの合成・8条件の再構成軌道からの合成を
聴き比べられる。窓ごとの連続比較WAVは64→128→256→512コードの順。
ピッチ・音量・話者条件を共通にし、音声ゲインも全候補で揃える。
`summary.csv`・`summary.json` に軌道MSE、差分MSE、変化量比、コード使用数・perplexityを保存し、
`codebook_comparison.png` に再構成誤差と変化量比を表示する。
表のbpsはコードIDだけの固定長換算で、ピッチ・音量・話者・モデル等は含めない。

128コードの基準モデルは、設定と分割が一致する
`outputs/jsut_window_sweep/window_020ms`・`window_040ms` を再利用し、変更しない。
その他のモデルは `codebook_<コード数>/window_<窓長>ms/` に保存する。
同じ設定で再実行すれば、完了済みモデルを再学習せずに集計できる。
候補は `--codebooks 64 128 256 512`、窓は `--windows-ms 20 40`、
出力先は `--output-dir`、基準モデルの保存先は `--reference-root` で指定する。
単一コード数の窓比較には `sweep_jsut_windows.sh --codebook-size 256 --windows-ms 20 40` も使える。
