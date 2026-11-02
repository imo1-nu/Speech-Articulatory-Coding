# ブランチ整理の対応表

[実験一覧](experiments.md) / [JSON対応表](branch-map.json)

整理前のブランチ名と先端コミットを保存した。JSUT研究枝の大半は同じmainを指しており、実験コード自体は未コミットだった。
したがって、枝名を切り替えて過去の実験コード版を取り出せる状態ではなかった。
今回、依存関係に沿って機能別に保存した。過去の実行時点のコミットを新たに捏造するものではない。

| 旧ローカルブランチ | 整理前コミット | 整理後 |
|---|---|---|
| `main` | `3df75e4266466ff4bd1c1502d86e65570ed4a178` | 保持 |
| `research/audio-resynthesis` | `71d36425e7d90d85966b5f6b32cbb8b7eafc7879` | main統合・push確認後にローカル枝を削除 |
| `research/branch-workflow` | `59c9298b4f539c71f707c050f271499dc42e24d4` | main統合・push確認後にローカル枝を削除 |
| `research/create-ema-from-jsut` | `3df75e4266466ff4bd1c1502d86e65570ed4a178` | main統合・push確認後にローカル枝を削除 |
| `research/feedback-experiment1` | `3df75e4266466ff4bd1c1502d86e65570ed4a178` | main統合・push確認後にローカル枝を削除 |
| `research/feedback-jacobian` | `3df75e4266466ff4bd1c1502d86e65570ed4a178` | main統合・push確認後にローカル枝を削除 |
| `research/feedback-test` | `3df75e4266466ff4bd1c1502d86e65570ed4a178` | main統合・push確認後にローカル枝を削除 |
| `research/feedback-vowels-raw` | `3df75e4266466ff4bd1c1502d86e65570ed4a178` | main統合・push確認後にローカル枝を削除 |
| `research/japanese-demo-acoustics` | `03781a032085f86d1912b3c87579ce3241e097a1` | 保持 |
| `research/jsut-40ms-25hz` | `3df75e4266466ff4bd1c1502d86e65570ed4a178` | main統合・push確認後にローカル枝を削除 |
| `research/jsut-articulatory-trajectories` | `3df75e4266466ff4bd1c1502d86e65570ed4a178` | main統合・push確認後にローカル枝を削除 |
| `research/jsut-codebook-sweep` | `3df75e4266466ff4bd1c1502d86e65570ed4a178` | main統合・push確認後にローカル枝を削除 |
| `research/jsut-cross-utterance-ppg` | `3df75e4266466ff4bd1c1502d86e65570ed4a178` | main統合・push確認後にローカル枝を削除 |
| `research/jsut-feedback-delay60` | `3df75e4266466ff4bd1c1502d86e65570ed4a178` | main統合・push確認後にローカル枝を削除 |
| `research/jsut-japanese-ppg` | `3df75e4266466ff4bd1c1502d86e65570ed4a178` | main統合・push確認後にローカル枝を削除 |
| `research/jsut-ppg-local-search` | `3df75e4266466ff4bd1c1502d86e65570ed4a178` | main統合・push確認後にローカル枝を削除 |
| `research/jsut-ppg-soft-transition` | `3df75e4266466ff4bd1c1502d86e65570ed4a178` | main統合・push確認後にローカル枝を削除 |
| `research/jsut-sequential-ppg-feedback` | `3df75e4266466ff4bd1c1502d86e65570ed4a178` | main統合・push確認後にローカル枝を削除 |
| `research/jsut-token-audio-comparison` | `3df75e4266466ff4bd1c1502d86e65570ed4a178` | main統合・push確認後にローカル枝を削除 |
| `research/jsut-vq-token-transitions` | `3df75e4266466ff4bd1c1502d86e65570ed4a178` | main統合・push確認後にローカル枝を削除 |
| `research/jsut-weighted-overlap` | `3df75e4266466ff4bd1c1502d86e65570ed4a178` | main統合・push確認後にローカル枝を削除 |
| `research/jsut-window-sweep` | `3df75e4266466ff4bd1c1502d86e65570ed4a178` | main統合・push確認後にローカル枝を削除 |
| `research/resynthesis-trial` | `e22b6e50f09918d6e03529d20f1fbc6674ca0c6a` | 保持 |
| `upstream-original` | `d59139be64cdbd5a6cb76c73fda00af0d0e97e59` | 保持 |
| `origin` | `3df75e4266466ff4bd1c1502d86e65570ed4a178` | main統合・push確認後にローカル枝を削除 |

## 機能別の保存コミット

- `b39d7d4eaea7926280b398fb96f5acb87243e44d` — feat: extract JSUT articulatory trajectories and pitch
- `03717f3877350d01a1e1a50d0afba29ce697288b` — feat: tokenize trajectories with VQ-VAE and analyze Markov transitions
- `8dd5de789024997dd1488847022ed4564722f79d` — feat: compare synthesis and sweep trajectory tokenization settings
- `79a69288fb974d714f6dbf995548c8c6d5e25310` — feat: extract pinned Japanese PPGs and compare test audio
- `2b8528788b8c3c9c64a03885e995ede206550347` — feat: edit discrete tokens with PPG loss and transition regularization
- `247214e5acdee160bf80d3a13824b077deb0beaa` — feat: simulate sequential PPG feedback with strict acoustic delay
- `66d0fd7d096cad08d5c36c63adce27099dc3fa3c` — feat: evaluate generated audio with Whisper large-v3 and acoustic metrics
- `d700a5859aa87683254b4fd8daa3a82b65486a71` — feat: evaluate cross-utterance A-to-B PPG control

整理用の`chore/organize-research-experiments`も、mainへの統合・push後に削除する。
独自履歴を持つ旧2枝、`upstream-original`、リモート枝は削除しない。今後の研究は更新後のmainから新しい枝を作成する。
作業前のパッチ・未追跡ファイル・全ref一覧はローカル`outputs/maintenance/organize-<UTC時刻>/`へ退避した。
