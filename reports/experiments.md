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
