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
| WavLM特徴量／PPG目標の一括・遅延FB | 完了 | [60条件の試聴と評価](../outputs/jsut_wavlm_feedback/index.html) | [実行手順](../docs/jsut-pipeline.md#wavlm特徴量を目標とするrvq制御) |
| 保存音声へのUTMOS22 strong追加評価 | 完了 | [UTMOS集計と試聴](../outputs/jsut_wavlm_feedback/index.html#utmos-evaluation) | [実行手順](../docs/jsut-pipeline.md#utmosによる追加評価) |
| 初期調音位置固定からのWavLM逐次制御 | 完了 | [固定・FBなし・FBありの試聴](../outputs/jsut_held_start_wavlm_feedback/index.html) | [実行手順](../docs/jsut-pipeline.md#初期調音位置を保持した状態からのwavlm制御) |
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

### WavLM特徴量を目標とするRVQ制御

作業ブランチ `research/wavlm-feature-feedback` は、ユーザーの指定により `research/rvq-first-stage-transition` の
`aecede4` から派生した。実装は `cbff807`、CSV保存の修正は `75d934b`。
WavLM-largeの固定revisionの `hidden_states[9]`（1024次元・平滑化前）を使い、フレームごとのcosine距離を最小化する。
SPARCと同じ正規化済み16 kHz波形・左右160サンプルのゼロ埋めで、同一入力に対する平滑化直前特徴量の最大差は0だった。
目標音声の読み込みはPPG側と共通の既存torchaudioリサンプリングを使う。

既存5発話、RVQ8・40 ms窓・50 Hz。全8段を編集し、遷移罰則は段1だけ。
オフラインはPPG/WavLM×λ=0/0.03/0.1、遅延制御は両目標×無編集/補正なし/FBありで、計60発話条件。
オフラインの予算は2パス×8段×24時刻、遅延制御は20 ms更新・60 ms波形観測遅延・400 ms探索窓・各段2時刻を1パス。
各座標12近傍＋4広域候補、バッチ4、採用前にキャッシュを使わず単独再抽出する。
PPGの100 HzとWavLMの50 Hzは、それぞれの実フレーム時刻で評価範囲・観測・発行時予測を対応付ける。
観測音声はSPARCで逐次生成した出力で、同じ生成器の発行時予測との誤差を補正するシミュレーション。
WavLMでは正規化特徴量の観測−予測をEMA更新し、ノルム上限1・ゲイン0.5で補正する。
目標特徴量と初期軌道・F0・音量・話者情報は事前に利用可能。終端の観測補完は解析専用。

| 条件 | WavLM距離 ↓ | PPG JS ↓ | CER % ↓ | MCD dB ↓ | STOI ↑ |
|---|---:|---:|---:|---:|---:|
| 編集前RVQ8 | 0.20234 | 0.15926 | 15.15 | 2.3156 | 0.9840 |
| 一括・PPG λ=0 | 0.22257 | 0.05699 | 13.13 | 3.3514 | 0.9370 |
| 一括・PPG λ=0.03 | 0.22477 | 0.05316 | 14.14 | 3.4080 | 0.9365 |
| 一括・PPG λ=0.1 | 0.21787 | 0.06430 | 17.17 | 3.1608 | 0.9494 |
| 一括・WavLM λ=0 | 0.15908 | 0.14416 | 10.10 | 3.0663 | 0.9680 |
| 一括・WavLM λ=0.03 | 0.16522 | 0.14988 | 15.15 | 2.8460 | 0.9737 |
| 一括・WavLM λ=0.1 | 0.16214 | 0.14082 | 14.14 | 2.9178 | 0.9732 |
| 逐次・無編集（両目標共通） | 0.20181 | 0.15930 | 15.15 | 2.3168 | 0.9840 |
| 逐次・PPG・補正なし | 0.33674 | 0.09517 | 76.77 | 5.7610 | 0.8308 |
| 逐次・PPG・FBあり | 0.34260 | 0.10192 | 68.69 | 5.7887 | 0.8199 |
| 逐次・WavLM・補正なし | 0.14991 | 0.14246 | 15.15 | 4.9636 | 0.8790 |
| 逐次・WavLM・FBあり | 0.15751 | 0.15142 | 21.21 | 5.0586 | 0.8698 |

特徴量距離は元音声を目標とした5発話の単純平均。CERはWhisper large-v3とJSUT正解文99文字のmicro平均。
上表のMCD・STOIは元軌道合成参照で、音響指標の集計方法は既存比較と同じ。
JSとcosine距離は尺度が異なるため、同じλを同じ実効的制約強度とは解釈しない。

WavLM目標は今回のPPG目標より、編集後のMCD/STOIと逐次CERの悪化を抑えた。
一方、同じWavLM目標ではFBありが補正なしを上回らず、距離・CER・MCD・STOIはいずれも悪化した。
逐次WavLM補正なしはWavLM距離が最小でも、一括WavLM λ=0よりCER・MCD・STOIは悪い。
特徴量距離の改善を、そのまま音声品質全般の改善とは扱わない。上表の元軌道合成参照では、編集前より良いMCD/STOIは得ていない。
接頭辞と全発話での特徴量の違いや、一定の特徴量バイアスだけによる補正の限界は原因候補だが、本実験では分離していない。
同一話者・同一発話目標の5発話・400 ms探索の予備比較で、A→B変換や他の探索窓への優劣は判断しない。

元音声が特徴量の目標なので、**元音声参照**の補助指標も別CSVに保存した。
元軌道合成自体のWavLM距離は0.19869、PPG JSは0.15488で、トークン化前にも原音との差がある。

| 条件 | 元音声参照MCD dB ↓ | 元音声参照STOI ↑ | CER % ↓ |
|---|---:|---:|---:|
| 元軌道合成 | 6.7602 | 0.8391 | 14.14 |
| 編集前RVQ8 | 6.7919 | 0.8371 | 15.15 |
| 一括・PPG λ=0 | 6.8801 | 0.8349 | 13.13 |
| 一括・WavLM λ=0 | 6.6818 | 0.8439 | 10.10 |
| 逐次・PPG・補正なし | 7.5319 | 0.7730 | 76.77 |
| 逐次・PPG・FBあり | 7.5177 | 0.7547 | 68.69 |
| 逐次・WavLM・補正なし | 6.6804 | 0.8549 | 15.15 |
| 逐次・WavLM・FBあり | 6.7665 | 0.8462 | 21.21 |

元音声参照ではWavLM目標のMCD/STOIが編集前より改善しており、元軌道合成から離れたことだけで悪化とは判断できない。
ただし、FB補正を加えた効果は補正なしを上回っていない。主評価と補助評価の参照を混在させない。

オフラインの条件あたり実測平均はPPG約24秒、WavLM約51〜52秒。
逐次最適化はPPG約132〜133秒、WavLM約247秒。キャッシュ利用込み・候補の音声生成と特徴量再抽出を含む時間で、ASRは別。
元のPPGオフライン15条件は、以前のRVQ遷移制約実験の保存音声とバイト単位で一致した。
CSVの共通列処理を修正して再開した履歴を保存し、検索・特徴抽出のコードはその前後で変更していない。
修正前31条件の実行時ソースを各summaryに記録し、波形・トークン・特徴量・探索履歴を保持した。
全94件のテストが通った。独立検証では60条件・5,142制御更新・66,208座標の履歴を照合し、
全観測接頭辞から特徴量とFBバイアスを再生した。保存WAVの特徴量の最大絶対差は0、CPUトークン再復号の軌道差は最大1.20×10⁻⁶未満。
両目標の逐次無編集音声は5発話すべてバイト単位で一致した。候補評価は1,098,737回、条件ごとの計算時間の合計は約82.7分
（RTX 5090、CUDA 12.8、初期化・ASR・独立検証は別）。詳細は比較レポート・manifest・verification・比較CSVを参照。

### UTMOS22 strongの追加評価

作業ブランチ `research/utmos-evaluation` はmainから作成し、検証済みのWavLM比較ブランチをfast-forwardで取り込んだ。
保存済み75音声（15条件×5発話、波形ハッシュで70種類）に、参照不要のUTMOS22 strong予測を追加した。
SpeechMOS v1.2.0の固定commitと固定SHA256の重みを使用。16 kHz・モノラル・1発話ずつで、
追加の音量正規化・無音除去・外部パディングなし、スコアの再スケーリングやクリップなし。
数値はモデルによる予測であり、聴取者によるMOSやUTMOSの全アンサンブル、UTMOSv2とは区別する。

| 条件 | UTMOS平均 ↑ | 標本標準偏差 | CER % ↓ |
|---|---:|---:|---:|
| 元音声 | 4.012 | 0.260 | 9.09 |
| 元軌道合成 | 3.376 | 0.335 | 14.14 |
| 編集前RVQ8 | 3.542 | 0.316 | 15.15 |
| 一括・PPG λ=0 | 3.164 | 0.388 | 13.13 |
| 一括・PPG λ=0.03 | 3.169 | 0.388 | 14.14 |
| 一括・PPG λ=0.1 | 3.276 | 0.327 | 17.17 |
| 一括・WavLM λ=0 | 3.682 | 0.294 | 10.10 |
| 一括・WavLM λ=0.03 | 3.635 | 0.351 | 15.15 |
| 一括・WavLM λ=0.1 | 3.725 | 0.329 | 14.14 |
| 逐次・無編集（両目標共通） | 3.542 | 0.317 | 15.15 |
| 逐次・PPG・補正なし | 2.070 | 0.334 | 76.77 |
| 逐次・PPG・FBあり | 2.099 | 0.182 | 68.69 |
| 逐次・WavLM・補正なし | 3.685 | 0.321 | 15.15 |
| 逐次・WavLM・FBあり | 3.414 | 0.314 | 21.21 |

条件平均は5発話の等重み、標準偏差はddof=1。CERとUTMOSは別の属性を測り、順位も一致しない。
今回の平均ではWavLM目標の方がPPG目標よりUTMOSが高い。逐次WavLMへのFB追加は、補正なしよりUTMOSも低下した。
元軌道合成よりトークン化後の予測値が高くても、元軌道の物理的正確さが高まったとは解釈しない。
日本語5発話の補助評価で、一般的な自然さの優劣や人のMOS評点に直接置き換えない。

発話別CSV・条件別CSV・既存音声指標との結合CSV・モデル版と前処理は `outputs/jsut_wavlm_feedback/utmos/` に保存。
基の音声・CSV・manifest・独立検証記録のハッシュは維持した。
公開torch.hubインターフェースを独立にロードして70波形すべてを再推論し、75行のスコア差は最大0だった。
キャッシュ再実行は新規推論0回で平均・標準偏差が一致し、レポート再適用で列とセクションが重複しないことも確認した。
全99テストが成功し、表への列追加後にもUTMOSの5テストを再確認した。

### 初期位置固定からのWavLM逐次制御

作業ブランチ `research/held-start-wavlm-feedback` はmainから新規作成し、検証済みUTMOS/WavLM実装をfast-forwardで引き継いだ。
探索実行版は `53b15a1`。その後の表示変更は参照音声の見出しとCER解釈の説明のみで、実行版のソースハッシュを保持した。
従来と同じテスト5発話・正解99文字で、全12調音座標を抽出軌道の第0フレームに固定した。
この一定位置の窓だけをAE/RVQ8で符号化し、同じ8段のコード組を全時刻へ繰り返したものを編集初期値にした。
元の動的軌道のトークンは使わず、元音声のWavLM-large hidden_states[9]を目標とする。
F0・音量・話者・合成ゲインは元の値を保持し、発声タイミング・抑揚は与える。

40 ms窓・50 Hz・RVQ128×8、20 ms更新・60 ms観測遅延・400 ms探索窓、各段2位置を1パス。
全8段を編集し、1段目遷移NLL重み0.03、12近傍＋4広域候補、バッチ4は従来通り。
FBなしも候補合成に基づく逐次最適化を行い、FBありだけが出力と発行時予測の特徴量差を予測補正へ戻す。
目標音声の特徴量は事前に全体が既知であり、接頭辞WavLMは非因果モデルの近似適用である。

| 条件 | WavLM距離 ↓ | CER % ↓ | MCD dB ↓ | STOI ↑ | UTMOS ↑ |
|---|---:|---:|---:|---:|---:|
| 元音声 | 0.00000 | 9.09 | 6.7636 | 0.8401 | 4.012 |
| 元軌道合成 | 0.19869 | 14.14 | 0.0000 | 1.0000 | 3.376 |
| 通常RVQ8再構成 | 0.20234 | 15.15 | 2.3156 | 0.9840 | 3.542 |
| 最初の位置を厳密に固定 | 0.50805 | 570.71 | 9.2282 | 0.6238 | 1.736 |
| 固定軌道をRVQ8で再構成 | 0.50542 | 570.71 | 9.2665 | 0.6361 | 1.725 |
| 逐次・固定トークンのまま | 0.50593 | 570.71 | 9.2658 | 0.6361 | 1.724 |
| 固定初期値・WavLM・FBなし | 0.24624 | 64.65 | 6.8516 | 0.7737 | 2.837 |
| 固定初期値・WavLM・FBあり | 0.26790 | 59.60 | 7.2273 | 0.7574 | 3.023 |

WavLM距離とUTMOSは発話の等重み平均、CERは99文字のmicro集計。MCD/STOI参照は元軌道合成。
固定条件ではBASIC5000_0053と0066の認識が長い反復出力となり、挿入誤りでCERが100%を超える。
570.71%はクリップしていない実際のCERで、認識全文と挿入・削除・置換数も発話別CSVに保持した。
音声の明瞭性を単独のCER値だけで判断しない。

固定状態から編集すると語の回復が見られ、FB追加で集計CERは64/99から59/99誤りへ、UTMOS平均は2.837から3.023へ改善した。
ただしFBのCER改善は2発話、悪化は2発話、同値は1発話。WavLM距離・元軌道合成参照のMCD/STOIはFBなしより悪い。
通常RVQ8再構成のCER15.15%・UTMOS3.542には達せず、初期位置固定からの回復は部分的である。
この結果は既存方式の5発話での比較であり、FB一般の優位性や元の調音動作の一意な回復を示すものではない。

保存先は `outputs/jsut_held_start_wavlm_feedback/`。8条件×5発話の40音声、初期化・編集後軌道・トークン・発声条件・観測・補正・履歴と軌道図を保存した。
全101テストが成功。保存波形から2,511接頭辞を独立再抽出し、2,571制御更新と15,234採用編集を再生した。
保存WavLM特徴量・観測特徴量・FB補正・復号軌道・距離の最大差はすべて0だった。
UTMOSの再表示はキャッシュからの新規推論0回で集計が一致した。
探索条件の実測時間合計は約41.6分（初期発話の診断評価の併走を含み、最終ASR・UTMOS・独立検証は別）。

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
