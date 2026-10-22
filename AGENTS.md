# ブランチ運用ルール

このリポジトリでは、以下の役割を維持する。

- `upstream-original`: 著者版を保持するブランチ。著者リポジトリの `upstream/main` を追跡し、独自の変更は加えない。
- `main`: 研究で使用する安定版。動作を確認した変更を取り込み、自分の fork の `origin/main` と同期する。
- 作業ブランチ: 新しい研究・実験・修正は、必ず `main` から新しく作成したブランチで行う。

## 作業手順

1. 作業開始時に現在のブランチと未コミットの変更を確認する。
2. `main` から、内容が分かる名前のブランチを作成する。研究には `research/<topic>`、修正には `fix/<topic>`、整備には `chore/<topic>` を使用できる。
3. 作業ブランチ上で変更・実験・必要な検証を行う。`main` や `upstream-original` で直接作業しない。
4. 安定版に取り込む変更は、検証後に作業ブランチから `main` にマージする。
5. 著者版の更新は `upstream-original` で受け取り、研究への導入・検証は `main` から作成した別の作業ブランチで行う。

```sh
git switch main
git switch -c research/<topic>
```

## 初期状態とリモート

- 運用開始時の著者版コミット: `d59139b` (`Update setup.py`)。
- `origin` は自分の fork `https://github.com/imo1-nu/Speech-Articulatory-Coding` を指す。
- `upstream` は著者リポジトリ `https://github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding` を指す。
- `upstream-original` の追跡先は `upstream/main` とする。安定版の `main` は `origin/main` を追跡する。
- 著者版の更新は、作業ツリーがクリーンな状態で以下のように取得する。早送りできない場合は履歴を確認し、強制的に上書きしない。

```sh
git fetch upstream
git switch upstream-original
git merge --ff-only upstream/main
git switch main
git switch -c research/update-upstream
git merge upstream-original
```

著者版の更新を取り込んだ作業ブランチでも必要な検証を実施し、安定して使用できることを確認してから `main` にマージする。
