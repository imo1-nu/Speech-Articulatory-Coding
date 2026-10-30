"""Export CSV tables, convenient audio files, validation and a Japanese report."""

import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from .adapter import EMA_NAMES
from .control_run import save_json
from .perturbation import apply_perturbation
from .run import sha256


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--report", type=Path, default=Path("reports/feedback_control_i.md"))
    args = parser.parse_args(argv)
    root = args.root
    jacobian = read(root / "jacobian/jacobian_summary.json")
    recovery = read(root / "recovery/recovery_summary.json")
    source = read(root / "source/source_summary.json")
    timing = {phase: read(root / phase / "timing.json") for phase in ("jacobian", "recovery", "source")}
    records = {phase: [json.loads(line) for line in (root / phase / "evaluations.jsonl").read_text().splitlines()]
               for phase in timing}
    by_id = {phase: {r["id"]: r for r in rows} for phase, rows in records.items()}
    accepted = [r for r in source["trajectory"] if r["status"] == "accepted"]
    final_id = accepted[-1]["evaluation_id"] if accepted else "00000"
    source_config = yaml_config(root / "source/config.yaml")
    initial_results = Path(source_config["baseline_results"])
    label = source_config["label"]
    for destination, sample in [("baseline.wav", "00000"), ("corrected.wav", final_id)]:
        shutil.copyfile(root / f"source/samples/{sample}.wav", root / "source" / destination)
    shutil.copyfile(initial_results / label / "original.wav", root / "source/original.wav")
    with np.load(root / "source/baseline_code.npz", allow_pickle=False) as data:
        code = dict(data)
    with np.load(root / "source/coordinates.npz", allow_pickle=False) as data:
        corrected = apply_perturbation(code, data["window"], data["reference_sd"], source["final_delta_sd"])
    np.savez_compressed(root / "source/corrected_code.npz", **corrected)
    np.savez_compressed(root / "source/final_delta_sd.npz", delta_sd=source["final_delta_sd"])
    for key in code:
        if key != "ema":
            np.testing.assert_array_equal(code[key], corrected[key])
    with np.load(root / f"source/samples/{final_id}.npz") as stored:
        np.testing.assert_array_equal(corrected["ema"], stored["ema"])
    checks = {"source_conditions_unchanged": True, "corrected_ema_matches_generated_audio": True}
    for phase, rows in records.items():
        assert len(rows) == timing[phase]["decode_count"]
        assert all(r["conditions_unchanged"] for r in rows)
        env = read(root / phase / "environment.json")
        assert all(sha256(p) == h for p, h in env["source_sha256"].items())
        for r in rows:
            with np.load(root / phase / f"samples/{r['id']}.npz", allow_pickle=False) as d:
                assert d["ema"].shape == code["ema"].shape
                assert np.isnan(d["filtered_hz"][~d["valid"]]).all()
            info = sf.info(root / phase / f"samples/{r['id']}.wav")
            assert info.frames == len(code["ema"]) * 320 and info.samplerate == 16000
        checks[phase] = {"evaluations": len(rows), "primary_invalid": sum(any(v is None for v in r["primary_hz"]) for r in rows),
                         "tracking_failures": sum(not r["tracking_ok"] for r in rows),
                         "all_conditions_unchanged": True, "source_hashes_match": True,
                         "max_evaluated_excursion_sd": max(float(np.linalg.norm(r["delta_sd"])) for r in rows)}
    assert all(np.linalg.norm(r["point_sd"]) <= source_config["source_correction"]["max_excursion_norm_sd"] + 1e-12 for r in accepted)
    for r in recovery["trials"]:
        if r["method"] == "random":
            match = next(x for x in recovery["trials"] if x["method"] == "jacobian" and all(x[k] == r[k] for k in ("amplitude_sd", "seed", "regularization", "gain")))
            assert abs(r["correction_norm_sd"] - match["correction_norm_sd"]) < 1e-12
    checks["random_correction_norms_match"] = True
    save_json(root / "validation.json", checks)
    write_csv(root / "jacobian/validation.csv", jacobian["validation"])
    write_csv(root / "recovery/trials.csv", recovery["trials"])
    write_csv(root / "recovery/aggregates.csv", recovery["aggregates"])
    all_metrics = []
    baseline = np.array(source["baseline_hz"])
    for row in recovery["trials"]:
        if row["evaluation_id"] is None:
            continue
        perturbed = np.array(by_id["recovery"][row["perturbed_id"]]["primary_hz"])
        actual = np.array(by_id["recovery"][row["evaluation_id"]]["primary_hz"])
        all_metrics.append({k: row[k] for k in ("amplitude_sd", "seed", "regularization", "gain", "method")} | {
            "before_hz_norm": float(np.linalg.norm(perturbed - baseline)),
            "after_hz_norm": float(np.linalg.norm(actual - baseline)),
            "before_normalized": row["before"], "after_normalized": row["after"], "improvement": row["improvement"]})
    write_csv(root / "recovery/errors.csv", all_metrics)
    trajectory_rows = [{"iteration": 0, "evaluation_id": "00000", "error": source["before"], "excursion_sd": 0,
                        **dict(zip(["F0", "F1", "F2", "F3"], by_id["source"]["00000"]["observed_hz"]))}]
    for k, row in enumerate(accepted, 1):
        trajectory_rows.append({"iteration": k, "evaluation_id": row["evaluation_id"], "error": row["after"],
            "excursion_sd": float(np.linalg.norm(row["point_sd"])),
            **dict(zip(["F0", "F1", "F2", "F3"], by_id["source"][row["evaluation_id"]]["observed_hz"]))})
    write_csv(root / "source/trajectory.csv", trajectory_rows)
    # An example chosen by fixed seed/amplitude, with the full aggregate table retained.
    example = next(r for r in recovery["trials"] if r["seed"] == 201 and r["amplitude_sd"] == .05
                   and r["regularization"] == .01 and r["gain"] == 1 and r["method"] == "jacobian")
    for name, sample in [("baseline", "00000"), ("perturbed", example["perturbed_id"]), ("corrected", example["evaluation_id"])]:
        shutil.copyfile(root / f"recovery/samples/{sample}.wav", root / f"recovery/example_{name}.wav")
    fig, axes = plt.subplots(6, 2, figsize=(13, 12), constrained_layout=True)
    for name, sample in [("baseline", "00000"), ("perturbed", example["perturbed_id"]), ("corrected", example["evaluation_id"])]:
        with np.load(root / f"recovery/samples/{sample}.npz") as data:
            for j, ax in enumerate(axes.flat):
                ax.plot(np.arange(len(data["ema"]))/50, data["ema"][:, j], label=name)
                ax.set(title=EMA_NAMES[j], xlabel="Nominal time (s)")
    axes.flat[0].legend()
    fig.savefig(root / "figures/recovery_ema_example.png", dpi=160)
    plt.close(fig)
    total_decodes = sum(t["decode_count"] for t in timing.values())
    total_decode_s = sum(t["decode_total_s"] for t in timing.values())
    total_wall = sum(t["total_wall_s"] for t in timing.values())
    original = np.array(source["original_target_hz"])
    final = np.array(source["final_hz"])
    raw_before, raw_after = np.linalg.norm(original - baseline), np.linalg.norm(original - final)
    final_observed = by_id["source"][final_id]["observed_hz"]
    original_observed = read(initial_results / label / "summary.json")["formants"]["original"]["f0"]["target_hz"]
    table_j = "\n".join(f"| {r['epsilon_sd']:.2f} | {r['diagnostics']['condition_number']:.3f} | {100*r['median_relative_error']:.1f}% | {r['median_cosine']:.3f} | {'通過' if r['passed'] else '不通過'} |" for r in jacobian["conditions"])
    table_r = []
    for regularization in source_config["recovery"]["regularizations"]:
        for gain in source_config["recovery"]["gains"]:
            a = next(r for r in recovery["aggregates"] if r["regularization"] == regularization and r["gain"] == gain and r["method"] == "jacobian")
            b = next(r for r in recovery["aggregates"] if r["regularization"] == regularization and r["gain"] == gain and r["method"] == "random")
            table_r.append(f"| {regularization:g} | {gain:g} | {100*a['median_improvement']:.1f}% | {100*b['median_improvement']:.2f}% | {round(8*a['improved_fraction_of_all'])}/8 | {round(8*b['improved_fraction_of_all'])}/8 |")
    table_t = "\n".join(f"| {phase} | {t['decode_count']} | {t['decode_total_s']:.3f} | {t['decode_median_s']:.5f} | {t['analysis_total_s']:.3f} | {t['total_wall_s']:.2f} |" for phase, t in timing.items())
    report = f"""# /i/ のヤコビアン推定・摂動復元・元音声への補正

作業ブランチ：`research/feedback-jacobian`。結果：`{root}`。

**小さなEMA摂動からの復元は8条件すべてで改善した。元音声を目標とした補正では、
正規化F1/F2誤差が{100*source['improvement']:.1f}%減少したが、変更量の上限に達し残差がある。**
これは /i/ の単一話者・単一区間でのオフライン実験であり、リアルタイム制御・吃音シミュレーションではない。

## 条件と再現手順

実験1と同じSPARC multi・CREPE・RTX 5090を使用した。モデルの学習は行わない。
保存済みcodeを使い、EMAのみを変更した。pitch/loudness/periodicity/pitch_stats/spk_emb/ft_lenを全生成で照合し、不変だった。
音響特徴は常に `[F0,F1,F2,F3]` を測定・保存し、主な推定・修正対象は安定して取得できるF1/F2の2次元とした。
F0/F3は補助評価とし、pitchを操作する制御は実施していない。

解析区間はWAV時刻0.48–0.64秒。Praat上限5000 Hz、他の解析・採用基準は実験1から固定した。
各フレームのF1/F2が基準から150/250 Hzを超えて変化した場合はピーク同定の変化の疑いとして無効にする補助判定を事前に設定した。
今回の全生成で主特徴の欠測・補助判定の不通過はなかった。これはピーク同定の正しさの独立検証ではない。

EMA摂動は中心0.56秒、sigma=0.06秒、半幅0.18秒のガウス型窓。
窓端の値を引いて外側を0とし、最大値1に正規化する。同じ窓をヤコビアン推定と修正に使用する。
座標zは12次元の**参照SD単位**で、実際のEMA変更は `window * reference_sd * z`。
参照SDは /a,i,u/ の指定区間にあるEMA計27フレームをプールした変数別SDである。
訓練データのSDでもmmでもなく、小規模な参照データに依存する。

主ヤコビアンは厳密な単一フレーム微分ではなく、この時間窓を通した特徴中央値の感度（2×12）である。
NPZにHz/参照SD、Hz/EMA単位、正規化した行列を保存した。
音響正規化には固定の基準再合成値 s=[{baseline[0]:.3f},{baseline[1]:.3f}] Hz を使用し、
誤差は `||(target - actual)/s||` とする。元音声への補正でもsを固定する。

```sh
bash scripts/feedback_control.sh --phase jacobian
bash scripts/feedback_control.sh --phase recovery
bash scripts/feedback_control.sh --phase source
PYTHONPATH=\"$PWD/.cache/feedback-deps\" MPLCONFIGDIR=\"$PWD/.cache/matplotlib\" \\
  ~/.local/share/sparc/venv/bin/python -m experiments.feedback.control_plots {root}
PYTHONPATH=\"$PWD/.cache/feedback-deps\" MPLCONFIGDIR=\"$PWD/.cache/matplotlib\" \\
  ~/.local/share/sparc/venv/bin/python -m experiments.feedback.control_report {root}
```

再実行する場合は全3段階で同じ新しい `--output outputs/feedback_control_i_new` を指定する。
既存結果は上書きしない。設定は `configs/feedback_control_i.yaml`。
モデル・環境・入力code・参照code・ソースのハッシュは各段階のenvironment/provenance JSONに保存した。

## 実験2：局所感度と線形予測

各摂動幅につき12変数×正負の24音声を生成して中央差分を計算した。
検証は別の4乱数seed（101–104）、2振幅（zのL2ノルム0.02/0.05）、正負の計16摂動で行った。
検証用方向は推定に用いた座標軸方向と異なる。各摂動幅の比較では同じ16音声を再利用した。
相対予測誤差は正規化座標の `||pred-actual||/||actual||`、方向一致はコサイン類似度。
事前の探索的移行基準はrank=2、有効率≥90%、中央値相対誤差≤50%、中央値cosine≥0.8とした。

| 摂動幅 (参照SD) | 正規化Jの条件数 | 中央値相対予測誤差 | 中央値cosine | 移行基準 |
|---|---:|---:|---:|---|
{table_j}

通過条件内で検証誤差が最小だった {jacobian['selected_epsilon_sd']} 参照SDを以降に使用した。
この選択に使ったデータでの指標なので、独立した最終性能推定ではない。復元実験には別seedを使用する。
0.01と0.02 SDの行列は符号が24要素中23要素で一致したが、正規化行列の相対差は22.4%あった。
0.05/0.10 SDでは0.02 SDに対する相対差が55.3%/57.3%となり、有限幅への依存性が残る。
中央対称性の残差もNPZ/JSONに保存した。方向の一致は良いが、大きさの予測には誤差がある。

## 摂動から基準再合成への復元

別seed201–204×2振幅0.05/0.10 SDの計8条件で比較した。
各摂動状態の近傍でヤコビアンを再推定し、正規化座標のdamped least squaresを用いた。
正則化3条件、ゲイン4条件について、修正なし・ヤコビアン・同ノルムのランダム方向を比較した。
1回の修正ノルムは0.10 SD以下、基準からの総ノルムは0.5 SD以下とする。
ランダム方向は固定seedで作り、実際に適用した修正ノルムの一致を確認した。

| 正則化lambda | ゲイン | J修正の改善率中央値 | ランダムの改善率中央値 | Jで改善 | ランダムで改善 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table_r)}

修正なしは全条件で改善率0%。改善率は正規化誤差の減少割合で、負値は悪化を示す。
上表は各行8条件の集計である。同じ8条件で複数ゲイン・正則化を比較しているため、独立な試行数を96とは数えない。
最も高い中央値は90.5%だが、これは今回比較した設定内の探索結果である。
全条件の値は `recovery/trials.csv`、Hz単位と正規化誤差は `recovery/errors.csv` に保存した。

## 摂動なしの基準から元音声への補正

各反復で近傍ヤコビアンを再推定し、正則化3条件×ゲイン4条件の候補を実際に生成した。
品質判定を通り実測誤差を減らした候補のうち最小誤差のものを採用した。
これは生成結果を比較するオフライン探索であり、単一ゲインのオンライン制御ではない。

| 特徴 | 元音声の目標 (Hz) | 基準再合成 (Hz) | 10反復後 (Hz) | 基準の符号付き誤差 (Hz) | 補正後の符号付き誤差 (Hz) |
|---|---:|---:|---:|---:|---:|
| F1 | {original[0]:.2f} | {baseline[0]:.2f} | {final[0]:.2f} | {baseline[0]-original[0]:+.2f} | {final[0]-original[0]:+.2f} |
| F2 | {original[1]:.2f} | {baseline[1]:.2f} | {final[1]:.2f} | {baseline[1]-original[1]:+.2f} | {final[1]-original[1]:+.2f} |

正規化誤差は {source['before']:.6f} → {source['after']:.6f}、改善率 **{100*source['improvement']:.1f}%**。
Hz単位の2次元誤差ノルムは {raw_before:.2f} → {raw_after:.2f} Hz（改善率{100*(raw_before-raw_after)/raw_before:.1f}%）。
最終EMA変更ノルムは {source['final_excursion_norm_sd']:.3f} SDで設定上限0.5に達した。
有限差分の探索点は修正点から±0.02 SDだけ外側も測るため、探索点のノルムは0.5を少し超え得る。
10回で終了し、残差があり、制約下の最適解や到達不能性を証明したわけではない。

F0は元音声{original_observed:.2f} Hz、基準再合成{by_id['source']['00000']['observed_hz'][0]:.2f} Hz、補正後{final_observed[0]:.2f} Hzだった。
F0は修正目的に含めず、固定pitch条件下での付随変化として保存した。F3は従来どおり主評価から外している。
元音声への補正は1例の探索で、ランダム制御との比較は復元段階で行った。母音・話者への一般化は未検証。

## 実行時間

GPU同期を行い、Decoder処理と音響解析・保存の時間を分離した。
保存済みcodeを使うため、各反復にEncoder処理は含まない。

| 段階 | 実生成回数 | Decoder累計 (秒) | 1回の中央値 (秒) | 音響解析累計 (秒) | 実行時間 (秒) |
|---|---:|---:|---:|---:|---:|
{table_t}
| 合計 | {total_decodes} | {total_decode_s:.3f} | — | {sum(t['analysis_total_s'] for t in timing.values()):.3f} | {total_wall:.2f} |

実行時間はrunner開始からの計測で、モデル読み込み・環境と重みのハッシュ記録・解析・保存を含む。
Pythonの起動・モジュールimport、実装作業、後処理の図作成・レポート生成は含まない。
各段階のモデル読み込みは約8.4秒で、重みのハッシュ記録にも時間がかかる。
初回生成は0.17–0.18秒程度、通常の生成は約0.005秒であった。
同一EMAの再要求はキャッシュを使い、上表は実際にDecoderを呼んだ回数のみを数えている。
事前ベンチマーク10回（Decoder累計0.258秒、モデルロード8.31秒）は上表とは別である。
これを含めた総生成は807回、Decoder累計は約4.88秒となる。

## 保存物・検証・限界

- 各段階の `evaluations.jsonl`：生成ID、EMA変更量、全4音響特徴、品質、生成・解析・保存時間。
- `samples/ID.wav` と `samples/ID.npz`：全生成音声、EMA、摂動ベクトル、音響軌跡・品質マスク。
- `jacobian/*.npz`：2×12の主ヤコビアンとF0/F3を含む補助4×12の感度。未取得値はNaN。
- `recovery/example_*.wav`：seed201、振幅0.05 SD、lambda0.01、gain1の比較用音声。全条件集計も保持。
- `source/original.wav`、`source/baseline.wav`、`source/corrected.wav`、`source/corrected_code.npz`：元音声への補正前後。
- `figures/`：ヤコビアン、摂動幅への感度、予測対実測、復元成績、元音声への補正、EMA軌跡。
- `validation.json`：全生成の条件不変性、保存データ、欠測処理、ソースハッシュ、ランダム方向のノルム一致。

重み不要の自動テストは18件成功。既存13件に加え、線形写像のヤコビアン・スケール変換、中央差分の2次精度、
正則化逆行列と制約、EMA以外の条件不変性、欠測勾配の扱いを検証した。
実SPARCの結果とは区別し、ログは結果ディレクトリの `tests.log` に保存した。

今回は /i/・1話者・短い1区間の予備検証である。/a/・/u/、複数話者、別の時間窓での検証は未実施。
採用基準は暫定値であり、フォルマントの真値やSPARCの訓練分布内であることを保証しない。
次は独立した母音データで同条件を検証し、F2/F3の測定安定性と、局所線形性・正則化への依存を確認する。
今回の元音声への補正率を、DIVAの発話中フィードバックや吃音症状への効果と解釈しない。
"""
    args.report.parent.mkdir(exist_ok=True)
    args.report.write_text(report)
    print(f"Saved report: {args.report}")
    print(json.dumps({"decode_count": total_decodes, "decode_total_s": total_decode_s,
                      "runner_wall_sum_s": total_wall, "validation": checks}, indent=2))


def yaml_config(path):
    import yaml
    return yaml.safe_load(path.read_text())


if __name__ == "__main__":
    main()
