# 実験一覧と結果

[概要](../README.md) / [JSUT手順](../docs/jsut-pipeline.md) / [PPG・FB手順](../docs/ppg-control.md) / [JSON索引](experiments.json) / [ブランチ対応表](branch-history.md)

音声・重み・CSV等はローカルの`outputs/`に保持し、移動・削除していない。以下の成果物リンクは同じ作業環境で利用する。
Gitだけを取得した環境では、この文書に収録した結果概要と手順を参照できる。

初回の整理以前に実行したJSUTコードは未コミットだった。今回の機能別コミットはコードの保存・整理であり、過去の実行時点を復元した履歴ではない。
モデル版・設定・出力の検証記録は各manifestとJSON索引を参照。データがない結果を成功扱いにはしない。

## 主条件と読み方

- 主条件は40 ms・128コード・50 Hz。トークンは20 ms間隔で、40 ms窓を重ねて復号する。
- この主条件は研究方針として選択したもので、全スイープでMSE最小という意味ではない。コード数比較の検証MSE最小は40 ms・512コード。
- トークン遷移は観測された一次マルコフ連鎖であり、HMMの隠れ状態学習は行っていない。
- 日本語PPGは35カテゴリ・100 Hz。Whisper評価はlarge-v3で、turboではない。
- PPG JSの低下だけではCER・STOIの改善を示さない。CERは表記差やASRの挿入に影響され、100%を超える場合もある。
- 100 ms確定実験と、20 ms確定・観測入力t−60 msの実験は、情報制限・更新幅・探索予算が異なる。
- A→B実験はAの軌道・F0・音量をB長に合わせ、A話者情報を固定する。BのPPGは伸縮しない。CER(A)の増加だけをBへの変換成功とは扱わない。

## 実験の対応

| 実験 | 状態 | 出力・レポート | 実装・手順 |
|---|---|---|---|
| JSUTのEMA・F0抽出 | 完了 | [outputs/jsut_ema](../outputs/jsut_ema) | [scripts/extract_jsut.sh](../scripts/extract_jsut.sh) |
| VQ-VAEと観測トークン遷移 | 完了 | [outputs/jsut_tokens](../outputs/jsut_tokens) | [scripts/model_jsut_tokens.sh](../scripts/model_jsut_tokens.sh) |
| 元音声・元軌道・再構成軌道の試聴 | 完了 | [outputs/jsut_tokens/audio_comparison](../outputs/jsut_tokens/audio_comparison/index.html) | [scripts/compare_jsut_audio.sh](../scripts/compare_jsut_audio.sh) |
| 重み付き重複復号の比較 | 完了 | [outputs/jsut_tokens/audio_comparison_weighted](../outputs/jsut_tokens/audio_comparison_weighted/index.html) | [scripts/compare_jsut_audio.sh](../scripts/compare_jsut_audio.sh) |
| 20〜180 ms・重みあり／なし | 完了 | [outputs/jsut_window_sweep](../outputs/jsut_window_sweep/index.html) | [scripts/sweep_jsut_windows.sh](../scripts/sweep_jsut_windows.sh) |
| 40 ms・50 Hzと25 Hz | 完了 | [outputs/jsut_40ms_25hz_comparison](../outputs/jsut_40ms_25hz_comparison/index.html) | [scripts/compare_jsut_token_rates.sh](../scripts/compare_jsut_token_rates.sh) |
| 20/40 ms・64/128/256/512コード | 完了 | [outputs/jsut_codebook_sweep](../outputs/jsut_codebook_sweep/index.html) | [scripts/sweep_jsut_codebooks.sh](../scripts/sweep_jsut_codebooks.sh) |
| 5発話×3音声の日本語PPG | 完了 | [outputs/jsut_ppg_40ms128](../outputs/jsut_ppg_40ms128/index.html) | [scripts/extract_jsut_ppg.sh](../scripts/extract_jsut_ppg.sh) |
| 編集前波形の照合で中断した初回探索 | 未完了 | [outputs/jsut_ppg_local_search](../outputs/jsut_ppg_local_search) | [scripts/local_jsut_ppg_search.sh](../scripts/local_jsut_ppg_search.sh) |
| 遷移制約なしの局所探索 | 完了 | [outputs/jsut_ppg_local_search_v1](../outputs/jsut_ppg_local_search_v1/index.html) | [scripts/local_jsut_ppg_search.sh](../scripts/local_jsut_ppg_search.sh) |
| 遷移NLLのソフト制約 | 完了 | [outputs/jsut_ppg_soft_transition](../outputs/jsut_ppg_soft_transition/index.html) | [scripts/sweep_ppg_transition.sh](../scripts/sweep_ppg_transition.sh) |
| 100 ms確定・固定遅延PPG | 完了 | [outputs/jsut_ppg_sequential](../outputs/jsut_ppg_sequential/index.html) | [scripts/compare_sequential_ppg.sh](../scripts/compare_sequential_ppg.sh) |
| 20 ms確定・観測波形t−60 ms | 完了 | [outputs/jsut_ppg_delay060_step020](../outputs/jsut_ppg_delay060_step020/index.html) | [scripts/compare_delayed_ppg.sh](../scripts/compare_delayed_ppg.sh) |
| 同一発話制御のWhisper large-v3・音響評価 | 完了 | [outputs/jsut_ppg_delay060_step020/audio_metrics](../outputs/jsut_ppg_delay060_step020/audio_metrics/index.html) | [scripts/evaluate_generated_audio.sh](../scripts/evaluate_generated_audio.sh) |
| A軌道からBのPPGへの制御 | 完了 | [outputs/jsut_ppg_A_to_B](../outputs/jsut_ppg_A_to_B/index.html) | [scripts/cross_utterance_ppg.sh](../scripts/cross_utterance_ppg.sh) |
| A→BのCER(A/B)と音響評価 | 完了 | [outputs/jsut_ppg_A_to_B/audio_metrics](../outputs/jsut_ppg_A_to_B/audio_metrics/index.html) | [scripts/evaluate_cross_utterance.sh](../scripts/evaluate_cross_utterance.sh) |
| 量子化なし連続AEの学習 | 完了 | [学習記録](../outputs/jsut_reconstruction_ablation/continuous_ae/) | [学習スクリプト](../scripts/train_continuous_ae.sh) |
| VQ128・VQ512・連続AEの再構成比較 | 完了 | [音声・カ行候補の比較](../outputs/jsut_reconstruction_ablation/comparison/index.html) | [比較スクリプト](../scripts/compare_trajectory_reconstruction.sh) |
| 固定連続AEのRVQ・128コード×1/2/4/8段 | 完了 | [音声・カ行候補の比較](../outputs/jsut_rvq/comparison_verified/index.html) | [RVQ手順](../docs/jsut-pipeline.md#固定した連続aeへのrvq) |
| RVQ8・1段目のみ遷移ソフト制約 | 完了 | [音声と探索結果](../outputs/jsut_rvq_first_stage_transition/index.html) | [実行手順](../docs/jsut-pipeline.md#rvqの1段目に遷移ソフト制約を加えた局所探索) |
| JSUT抽出スモークテスト | スモーク | [outputs/jsut_smoke](../outputs/jsut_smoke) | [scripts/extract_jsut.sh](../scripts/extract_jsut.sh) |
| VQスモークテスト | スモーク | [outputs/jsut_tokens_smoke](../outputs/jsut_tokens_smoke) | [scripts/model_jsut_tokens.sh](../scripts/model_jsut_tokens.sh) |
| 復号条件修正前の試聴出力 | 置換済み | [outputs/jsut_window_sweep/window_040ms/audio_comparison_before_exact_weights](../outputs/jsut_window_sweep/window_040ms/audio_comparison_before_exact_weights) | [scripts/compare_jsut_audio.sh](../scripts/compare_jsut_audio.sh) |
| feedback_control_a_raw | 旧実験 | [outputs/feedback_control_a_raw](../outputs/feedback_control_a_raw) | `research/japanese-demo-acoustics` |
| feedback_control_i | 旧実験 | [outputs/feedback_control_i](../outputs/feedback_control_i) | `research/japanese-demo-acoustics` |
| feedback_control_i_raw | 旧実験 | [outputs/feedback_control_i_raw](../outputs/feedback_control_i_raw) | `research/japanese-demo-acoustics` |
| feedback_control_u_raw | 旧実験 | [outputs/feedback_control_u_raw](../outputs/feedback_control_u_raw) | `research/japanese-demo-acoustics` |
| feedback_experiment1 | 旧実験 | [outputs/feedback_experiment1](../outputs/feedback_experiment1) | `research/japanese-demo-acoustics` |
| feedback_experiment1_a | 旧実験 | [outputs/feedback_experiment1_a](../outputs/feedback_experiment1_a) | `research/japanese-demo-acoustics` |
| feedback_experiment1_cpu | 旧実験 | [outputs/feedback_experiment1_cpu](../outputs/feedback_experiment1_cpu) | `research/japanese-demo-acoustics` |
| feedback_experiment1_f0 | 旧実験 | [outputs/feedback_experiment1_f0](../outputs/feedback_experiment1_f0) | `research/japanese-demo-acoustics` |
| feedback_experiment1_preliminary | 旧実験 | [outputs/feedback_experiment1_preliminary](../outputs/feedback_experiment1_preliminary) | `research/japanese-demo-acoustics` |
| japanese_demo_raw | 旧実験 | [outputs/japanese_demo_raw](../outputs/japanese_demo_raw) | `research/japanese-demo-acoustics` |

`jsut_ppg_local_search`は波形照合で中断した出力で、完了した`jsut_ppg_local_search_v1`とは別物。
ソフト制約スイープのトップmanifestは`runs[]`を持ち、`records[]`がないことは失敗を意味しない。

## 記録された主要結果

以下は既存CSVから転記した数値。指標定義・母集団は実験ごとに異なるため、条件をまたいで順位だけを比較しない。
空欄は未計測・非該当であり、ゼロを意味しない。すべての数値行はJSON索引にも保存した。

### 時間窓・重み

| window_ms | weighting | validation_mse | test_mse | test_used_codes |
|---|---|---|---|---|
| 20 | uniform | 0.12937 | 0.12976 | 128 |
| 20 | triangular | 0.12937 | 0.12976 | 128 |
| 40 | uniform | 0.10645 | 0.10679 | 128 |
| 40 | triangular | 0.10645 | 0.10679 | 128 |
| 60 | uniform | 0.11332 | 0.11350 | 128 |
| 60 | triangular | 0.11284 | 0.11311 | 128 |
| 80 | uniform | 0.12430 | 0.12438 | 128 |
| 80 | triangular | 0.12183 | 0.12193 | 128 |
| 100 | uniform | 0.13376 | 0.13400 | 128 |
| 100 | triangular | 0.13108 | 0.13154 | 128 |
| 120 | uniform | 0.14718 | 0.14747 | 128 |
| 120 | triangular | 0.14303 | 0.14344 | 128 |
| 140 | uniform | 0.16309 | 0.16333 | 128 |
| 140 | triangular | 0.15886 | 0.15904 | 128 |
| 160 | uniform | 0.17845 | 0.17859 | 128 |
| 160 | triangular | 0.17312 | 0.17300 | 128 |
| 180 | uniform | 0.19472 | 0.19568 | 128 |
| 180 | triangular | 0.18760 | 0.18840 | 128 |

### 40 ms・トークン周期

| test_mse | test_tokens | test_used_codes |
|---|---|---|
| 0.10679 | 58629 | 128 |
| 0.15024 | 29380 | 128 |

### 時間窓・コード数

| window_ms | codebook_size | validation_mse | test_mse | test_used_codes |
|---|---|---|---|---|
| 20 | 64 | 0.16045 | 0.16103 | 64 |
| 20 | 128 | 0.12937 | 0.12976 | 128 |
| 20 | 256 | 0.10529 | 0.10528 | 253 |
| 20 | 512 | 0.11925 | 0.11931 | 168 |
| 40 | 64 | 0.13628 | 0.13710 | 63 |
| 40 | 128 | 0.10645 | 0.10679 | 128 |
| 40 | 256 | 0.11166 | 0.11209 | 120 |
| 40 | 512 | 0.08617 | 0.08621 | 276 |

### 遷移ソフト制約・同一発話

| transition_weight | final_js | final_transition_nll | final_unseen_transition_fraction |
|---|---|---|---|
| 0.00000 | 0.12805 | 3.22465 | 0.10739 |
| 0.01000 | 0.13206 | 3.00031 | 0.07394 |
| 0.03000 | 0.13316 | 2.87720 | 0.05634 |
| 0.10000 | 0.15218 | 2.43496 | 0.01585 |

### 100 ms確定・固定遅延PPG

| horizon_ms | mode | mean_js | transition_nll |
|---|---|---|---|
| 200 | unedited | 0.23544 | 2.24581 |
| 200 | no_feedback | 0.13728 | 3.45487 |
| 200 | feedback | 0.12863 | 3.61585 |
| 400 | unedited | 0.23954 | 2.24581 |
| 400 | no_feedback | 0.11304 | 3.74246 |
| 400 | feedback | 0.11135 | 3.67151 |
| 800 | unedited | 0.23955 | 2.24581 |
| 800 | no_feedback | 0.11304 | 3.48670 |
| 800 | feedback | 0.10220 | 3.65152 |

### 20 ms確定・観測波形t−60 ms

| horizon_ms | mode | mean_js | transition_nll |
|---|---|---|---|
| 200 | unedited | 0.23998 | 2.24581 |
| 200 | no_feedback | 0.14897 | 4.47567 |
| 200 | feedback | 0.14179 | 4.25961 |
| 400 | unedited | 0.23953 | 2.24581 |
| 400 | no_feedback | 0.12778 | 3.84150 |
| 400 | feedback | 0.14953 | 3.84698 |
| 800 | unedited | 0.23950 | 2.24581 |
| 800 | no_feedback | 0.11974 | 3.90481 |
| 800 | feedback | 0.12387 | 3.84577 |

### 同一発話制御・音声評価（CERは比率）

| condition | cer | mcd_db | stoi |
|---|---|---|---|
| original | 0.11940 | 0.00000 | 1.00000 |
| original_ema | 0.19403 | 6.77678 | 0.84211 |
| before | 0.50746 | 7.50681 | 0.78733 |
| offline | 0.50746 | 7.42905 | 0.79428 |
| horizon_200ms_unedited | 0.53731 | 7.50131 | 0.78750 |
| horizon_200ms_no_feedback | 4.37313 | 7.88745 | 0.75022 |
| horizon_200ms_feedback | 0.83582 | 7.79043 | 0.74325 |
| horizon_400ms_unedited | 0.50746 | 7.50690 | 0.78735 |
| horizon_400ms_no_feedback | 0.80597 | 7.83231 | 0.73875 |
| horizon_400ms_feedback | 0.92537 | 7.71296 | 0.74665 |
| horizon_800ms_unedited | 0.50746 | 7.50634 | 0.78736 |
| horizon_800ms_no_feedback | 0.76119 | 7.62759 | 0.75632 |
| horizon_800ms_feedback | 0.98507 | 7.56887 | 0.76499 |

### A→B・音声評価（CERは比率）

| condition | cer_B | cer_A | ppg_js_to_B | mcd_db | stoi |
|---|---|---|---|---|---|
| target_original | 0.11940 | 0.94030 | 0.00000 | 0.00000 | 1.00000 |
| source_initial | 0.95522 | 0.44776 | 0.53426 | 11.51746 | 0.23653 |
| target_tokens_oracle | 8.28358 | 8.35821 | 0.39378 | 9.20580 | 0.39786 |
| target_ema_oracle | 1.02985 | 1.07463 | 0.36043 | 8.71850 | 0.44951 |
| horizon_200ms_unedited | 0.95522 | 0.44776 | 0.53427 | 11.52447 | 0.23621 |
| horizon_200ms_no_feedback | 1.04478 | 1.04478 | 0.40668 | 10.78129 | 0.29660 |
| horizon_200ms_feedback | 1.29851 | 1.26866 | 0.40305 | 10.52967 | 0.29260 |
| horizon_400ms_unedited | 0.95522 | 0.44776 | 0.53428 | 11.51514 | 0.23655 |
| horizon_400ms_no_feedback | 2.97015 | 2.85075 | 0.44772 | 11.13185 | 0.28138 |
| horizon_400ms_feedback | 1.40299 | 1.40299 | 0.42852 | 11.02258 | 0.29944 |
| horizon_800ms_unedited | 0.95522 | 0.44776 | 0.53426 | 11.51429 | 0.23663 |
| horizon_800ms_no_feedback | 1.29851 | 1.26866 | 0.41931 | 10.96730 | 0.27083 |
| horizon_800ms_feedback | 0.98507 | 0.82090 | 0.42314 | 11.08143 | 0.28131 |

### 量子化あり／なしの再構成比較

40 ms・50 Hz、VQ128/512の既存重みを固定し、同じMLPの連続AEを独立学習した。
軌道はテスト250発話・58,629フレーム、音声は既存の5試聴発話。
音響指標の参照は元軌道合成で、F0・音量・話者・ゲインは全再構成で同一。CERはJSUT正解文を参照する。
比較実行ソースは `80498a79f5bdec367040226b4d4d0c93ff538903`。学習AEは量子化なしの参考で、同じビットレートではない。

| 条件 | 重複平均後MSE | 差分MSE | 速度エネルギー比 |
|---|---|---|---|
| VQ128 | 0.1067924 | 0.0474726 | 0.9094 |
| VQ512 | 0.0862074 | 0.0402363 | 0.9093 |
| 連続AE | 0.0004723 | 0.0003422 | 0.9879 |

| 条件 | CER（比率） | MCD（dB） | STOI |
|---|---|---|---|
| 元音声 | 0.09091 | 6.7636 | 0.8401 |
| 元軌道合成 | 0.14141 | 0.0000 | 1.0000 |
| VQ128 | 0.42424 | 5.0591 | 0.8757 |
| VQ512 | 0.41414 | 4.7737 | 0.8919 |
| 連続AE | 0.15152 | 1.8455 | 0.9900 |

連続AEは元軌道合成に近いSTOI・CERを得た一方、512コード化によるCER改善は小さかった。
この結果は量子化を含む学習・表現の制約が大きいことを示唆するが、離散ボトルネックの瞬間的な誤差だけを分離した因果推定ではない。
元軌道合成PPGから選んだk/ky候補9区間の音声・共通スケールのスペクトル・TDX/TDY位置と速度をレポートに掲載した。
候補は正解音素境界ではなく、音声評価は5発話に限られる。速度損失・音声損失・RVQはこの比較では未導入。

### 固定連続AEへのRVQ

作業ブランチは `research/trajectory-rvq`、実装は `231874f`。
同じ40 ms窓・50 Hz・4500/250/250分割・訓練正規化を使用する。
学習済み連続AEを固定し、訓練発話の潜在表現と残差に128中心のMiniBatchKMeansを順に適用した。
1/2/4/8段は共通8段モデルの接頭部。テスト発話のトークンIDだけから復号して合成し、連続潜在表現の迂回路はない。
既存VQは同時学習、今回は固定AEへの事後量子化なので学習方法も異なる。

軌道はテスト250発話・58,629フレーム、音声は従来と同じ5発話・正解99文字。
音声条件は固定し、MCD/STOI/F0/VUVの参照は元軌道合成、CERはWhisper large-v3とJSUT正解文。
既存5条件の音声指標はモデル・生成器・波形ハッシュと認識設定を照合して再利用した。

| 条件 | 公称bit/s | 標準化MSE ↓ | 差分MSE ↓ | MCD dB ↓ | STOI ↑ | CER % ↓ |
|---|---:|---:|---:|---:|---:|---:|
| VQ128 | 350 | 0.1067924 | 0.0474726 | 5.0591 | 0.8757 | 42.42 |
| VQ512 | 450 | 0.0862074 | 0.0402363 | 4.7737 | 0.8919 | 41.41 |
| RVQ128×1 | 350 | 0.1081399 | 0.0490674 | 4.9520 | 0.8853 | 41.41 |
| RVQ128×2 | 700 | 0.0439438 | 0.0282975 | 4.2315 | 0.9145 | 19.19 |
| RVQ128×4 | 1400 | 0.0150450 | 0.0146060 | 3.2740 | 0.9569 | 10.10 |
| RVQ128×8 | 2800 | 0.0028655 | 0.0036157 | 2.3156 | 0.9840 | 15.15 |
| 連続AE | — | 0.0004723 | 0.0003422 | 1.8455 | 0.9900 | 15.15 |

公称bit/sはトークンのみ・エントロピー符号化なし。連続AEは同一情報量の比較ではない。
元音声CERは9.09%、元軌道合成は14.14%。段数を増やすと軌道誤差・MCD・STOIが改善し、
RVQ8段は連続AEに近づく。テストで全8段それぞれ128/128コードが使用され、perplexityは122.70〜127.47。
6〜10 Hzエネルギー比はRVQ1/2/4/8で0.6800/0.7369/0.8599/0.9501、連続AEは0.9758。

CERは4段10/99文字、8段15/99文字で単調改善しない。
F0 RMSEもRVQ1/2/4/8で5.83/8.03/9.21/5.47 Hzと単調改善せず、4段はVQ128の5.58 Hzより大きい。
入力F0が同じでも、生成波形から測定するF0は調音軌道や抽出誤差の影響を受ける。
4段のCERが元軌道合成より低くても、元軌道の再現精度やカ行の正解率が上回ったとは解釈しない。
再現優先の候補は8段、情報量との折り合いを見る候補は4段。音声5発話の予備評価であり、
同一ビットレートでの方式優劣や、主観的明瞭度の確定には使わない。

試聴ページには元音声・元軌道合成・VQ128/512・連続AE・RVQ全4条件と、参照PPGによるk/ky候補9区間のクリップを掲載。
元データ、既存モデル・音声を変更していない。最初の比較出力も保持し、最終比較では保存トークン復号を明示し、
最初の比較と全20本のRVQ音声が一致することを確認した。詳細は最終比較のmanifest・verificationを参照。
保存トークンのCPU再復号と評価時GPU復号の軌道差は最大1.20×10⁻⁶未満。
RVQの残差・段数接頭部・トークン復号・訓練分割限定のテストを追加し、全84件のテストとシェル構文検証が通った。

### RVQ8・1段目の遷移ソフト制約

作業ブランチ `research/rvq-first-stage-transition`、実装 `592ba2d`。
全8段のトークンを編集し、1段目だけに平均遷移NLLを加えたオフラインの全発話PPG局所探索。
ターゲットは同じ発話の元音声PPG。既存5発話、正解99文字、40 ms・50 Hz・各段128コード。
F0・音量・話者・ゲインは固定し、音響指標の参照は元軌道合成、CERはWhisper large-v3とJSUT正解文。
段1→8の順に最大2パス、各段24時刻、12近傍＋4広域候補、バッチ4。
全条件で同じ探索予算規則を用い、1つのIDだけを変えて他段は再量子化しない。
候補生成は段ごとの潜在コードベクトル距離による。採用前に単独再評価し、目的関数の改善を確認する。

訓練4500発話・1,096,065遷移で専用の1段目遷移表を学習した（α=1、自己遷移あり、発話間遷移なし）。
観測された遷移種類は5,005/16,384。訓練・検証・テストのNLLは2.16686/2.17322/2.17896 nats/edge、
未観測遷移率は検証0.0573%、テスト0.0531%。自然な初期トークンに対しては、128コードの推定が極端に疎という結果ではない。

| 条件 | PPG JS ↓ | 段1 NLL ↓ | 軌道MSE ↓ | MCD dB ↓ | STOI ↑ | CER % ↓ |
|---|---:|---:|---:|---:|---:|---:|
| 編集前RVQ8 | 0.15926 | 2.19837 | 0.002734 | 2.3156 | 0.9840 | 15.15 |
| λ=0 | 0.05699 | 2.81175 | 0.080900 | 3.3514 | 0.9370 | 13.13 |
| λ=0.01 | 0.05126 | 2.67435 | 0.076747 | 3.3780 | 0.9346 | 26.26 |
| λ=0.03 | 0.05316 | 2.53725 | 0.057289 | 3.4080 | 0.9365 | 14.14 |
| λ=0.1 | 0.06430 | 2.32458 | 0.037680 | 3.1608 | 0.9494 | 17.17 |

JS・NLLは5発話の単純平均、CERは全99文字のmicro平均、軌道MSEは同じ5発話857フレームの重複平均後。
前節のRVQ250発話の軌道MSEと分母が異なる。

| 条件 | 段1の未観測遷移 % | 段1編集数 | 他段編集数 | 速度RMS SD/s | 加速度RMS SD/s² |
|---|---:|---:|---:|---:|---:|
| 編集前RVQ8 | — | 0 | 0 | 17.075 | 618.169 |
| λ=0 | 5.994 | 100 | 434 | 20.936 | 1101.672 |
| λ=0.01 | 4.969 | 88 | 531 | 20.782 | 1089.354 |
| λ=0.03 | 3.319 | 83 | 438 | 19.973 | 1010.225 |
| λ=0.1 | 1.287 | 63 | 463 | 18.938 | 899.727 |

λ=0.1はλ=0より未観測遷移を約79%、加速度RMSを約18%減らし、MCD/STOIも改善した。
ただし編集前より軌道MSE・差分MSE・MCD・STOIは悪化し、加速度RMSも約46%高い。
CERはλ=0が13/99文字、λ=0.01が26/99文字、λ=0.03が14/99文字、λ=0.1が17/99文字で、
遷移制約による認識改善は確認できない。λ=0.01の悪化にはBASIC5000_0066の17/23文字誤りが大きく寄与した。
PPG JSが最小でも音声指標が最良とは限らず、強いλも全発話で同じ効果ではない。

1段目の遷移を学習分布へ寄せる効果は確認できたが、それだけで全8段を合成した軌道や音声の再現性を保つには不十分。
復号軌道の変化量や音響的な再現を目的関数に含める比較が次の候補になる。
速度・加速度RMSの低下は物理的妥当性の保証ではない。音声5発話の予備評価で、λの一般的な最適値は決めない。

実行は約517秒。全20条件で初期トークンと編集前音声を照合し、全7,680座標ステップの目的関数が非増加であることを確認。
保存トークンのCPU再復号と保存軌道との差は最大1.20×10⁻⁶未満。保存WAVのPPGも採用結果と一致した。
全86件のテストが通り、詳細はレポート・manifest・verification・comparison_summary.csvに保持した。

## 旧実験と保全した履歴

母音別フォルマントFB・ヤコビアン解析・日本語デモのコードと詳細な報告は、ローカルの`research/japanese-demo-acoustics`に保持する。
可視化・再合成試行は`research/resynthesis-trial`に保持する。これらの枝は今回mainへ混ぜず、履歴を保存する。
ローカルで旧報告を読む例：

```sh
git show research/japanese-demo-acoustics:reports/initial_experiment.md
git show research/japanese-demo-acoustics:reports/feedback_control_raw.md
git show research/japanese-demo-acoustics:docs/feedback_test.md
```

この2枝はローカル保全対象で、今回のpush対象はmainのみ。新規cloneで旧枝が取得できるとは限らない。
音声可視化MP4・画像、`source_audio`・`validation`等の補助出力も元の場所に保持する。

## 再実行と検証

詳細コマンドはdocsを参照。記載した既存出力先への再実行は上書き保護により拒否されるため、新しい出力先を指定する。
整理時は既存の75件のテスト、全シェル構文、CLIヘルプ、文書・索引のローカルパスを検証する。
重みの再学習や実験の再生成は整理の一部として実施しない。
