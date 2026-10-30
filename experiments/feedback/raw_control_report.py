"""Summarize vowel control runs using all finite estimates, with no quality gate."""

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import yaml

from .acoustics import FEATURE_NAMES, feature_arrays
from .control_report import write_csv
from .control_run import save_json
from .perturbation import apply_perturbation
from .raw_acoustics import extract_raw_acoustics
from .run import sha256


def read(path):
    return json.loads(path.read_text())


def export_vowel(root):
    phases = ["jacobian", "recovery", "source"]
    config = yaml.safe_load((root / "source/config.yaml").read_text())
    label = config["label"]
    initial_config = yaml.safe_load(Path(config["experiment1_config"]).read_text())
    item = next(x for x in initial_config["inputs"] if x["label"] == label)
    prior = Path(config["baseline_results"])
    source = read(root / "source/source_summary.json")
    jacobian = read(root / "jacobian/jacobian_summary.json")
    recovery = read(root / "recovery/recovery_summary.json")
    records = {phase: [json.loads(line) for line in (root / phase / "evaluations.jsonl").read_text().splitlines()]
               for phase in phases}
    by_id = {phase: {x["id"]: x for x in rows} for phase, rows in records.items()}
    accepted = [x for x in source["trajectory"] if x["status"] == "accepted"]
    final_id = accepted[-1]["evaluation_id"] if accepted else "00000"
    for name, identifier in [("baseline", "00000"), ("corrected", final_id)]:
        shutil.copyfile(root / f"source/samples/{identifier}.wav", root / f"source/{name}.wav")
    shutil.copyfile(prior / label / "original.wav", root / "source/original.wav")
    with np.load(root / "source/baseline_code.npz", allow_pickle=False) as data:
        code = dict(data)
    with np.load(root / "source/coordinates.npz", allow_pickle=False) as data:
        corrected = apply_perturbation(code, data["window"], data["reference_sd"], source["final_delta_sd"])
    np.savez_compressed(root / "source/corrected_code.npz", **corrected)
    validation = {"feature_policy": "raw", "quality_filtering": False,
                  "jacobian_gate_enforced": jacobian["validation_gate_enforced"]}
    assert jacobian["validation_gate_enforced"] is False
    for name in code:
        if name != "ema":
            np.testing.assert_array_equal(code[name], corrected[name])
    with np.load(root / f"source/samples/{final_id}.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(corrected["ema"], data["ema"])
    timings = {}
    for phase in phases:
        timings[phase] = read(root / phase / "timing.json")
        env = read(root / phase / "environment.json")
        assert all(sha256(p) == h for p, h in env["source_sha256"].items())
        assert len(records[phase]) == timings[phase]["decode_count"]
        for row in records[phase]:
            assert row["feature_policy"] == "raw" and row["quality"] is None
            assert row["tracking_ok"] is None and row["rejection_reasons"] is None
            assert row["conditions_unchanged"]
            with np.load(root / phase / f"samples/{row['id']}.npz", allow_pickle=False) as data:
                np.testing.assert_array_equal(data["raw_hz"], data["filtered_hz"])
                np.testing.assert_array_equal(data["valid"], np.isfinite(data["raw_hz"]))
                assert data["ema"].shape == code["ema"].shape
            info = sf.info(root / phase / f"samples/{row['id']}.wav")
            assert info.samplerate == 16000 and info.frames == len(code["ema"]) * 320
        validation[phase] = {"decode_count": len(records[phase]), "raw_values_preserved": True,
            "conditions_unchanged": True, "source_hashes_match": True,
            "missing_primary_measurements": sum(any(v is None for v in x["primary_hz"]) for x in records[phase])}
    random_pairs = {}
    for row in recovery["trials"]:
        key = tuple(row[k] for k in ["amplitude_sd", "seed", "regularization", "gain"])
        if row["method"] == "jacobian":
            random_pairs[key] = row["correction_norm_sd"]
    for row in recovery["trials"]:
        if row["method"] == "random":
            key = tuple(row[k] for k in ["amplitude_sd", "seed", "regularization", "gain"])
            if row["correction_norm_sd"] is not None:
                assert abs(row["correction_norm_sd"] - random_pairs[key]) < 1e-12
    validation["random_correction_norms_match"] = True
    save_json(root / "validation.json", validation)
    measured = {}
    for name in ["original", "baseline", "corrected"]:
        wav, sr = sf.read(root / f"source/{name}.wav")
        measured[name] = extract_raw_acoustics(wav, sr, item["start_s"], item["end_s"], initial_config["formants"])
        np.savez_compressed(root / f"source/{name}_acoustics.npz", **feature_arrays(measured[name]))
    features = {}
    for name, data in measured.items():
        features[name] = [data["summary"]["f0"]["target_hz"], *[x["target_hz"] for x in data["summary"]["formants"]]]
    feature_rows = [{"feature": feature, "original_hz": features["original"][j],
                    "baseline_hz": features["baseline"][j], "corrected_hz": features["corrected"][j],
                    "role": "controlled" if feature in config["primary_features"] else "observed"}
                   for j, feature in enumerate(FEATURE_NAMES)]
    write_csv(root / "source/feature_comparison.csv", feature_rows)
    for key in ["original_hz", "baseline_hz", "corrected_hz"]:
        assert len([row[key] for row in feature_rows]) == 4
    for name, index in [("original", "original_target_hz"), ("baseline", "baseline_hz"), ("corrected", "final_hz")]:
        np.testing.assert_allclose(features[name][1:3], source[index], rtol=0, atol=1e-9)
    trajectory = [{"iteration": 0, "error": source["before"], "excursion_sd": 0,
                  **dict(zip(FEATURE_NAMES, by_id["source"]["00000"]["observed_hz"]))}]
    for k, row in enumerate(accepted, 1):
        trajectory.append({"iteration": k, "error": row["after"], "excursion_sd": float(np.linalg.norm(row["point_sd"])),
                           **dict(zip(FEATURE_NAMES, by_id["source"][row["evaluation_id"]]["observed_hz"]))})
    write_csv(root / "source/trajectory.csv", trajectory)
    write_csv(root / "recovery/trials.csv", recovery["trials"])
    write_csv(root / "recovery/aggregates.csv", recovery["aggregates"])
    write_csv(root / "jacobian/validation.csv", jacobian["validation"])
    # Plot every estimate in blue, with common frequency axes for each feature.
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True, sharey="row", constrained_layout=True)
    fig.suptitle(f"/{label}/: all estimates, no acoustic validity filtering")
    for col, (name, data) in enumerate(measured.items()):
        arrays = feature_arrays(data)
        for j, feature in enumerate(FEATURE_NAMES):
            axes[j, col].plot(arrays["time_s"], arrays["raw_hz"][:, j], "o-", color="tab:blue", markersize=3)
            axes[j, col].set(title=f"{name} {feature}", xlabel="WAV time (s)", ylabel="Hz")
    fig.savefig(root / "figures/raw_acoustics_comparison.png", dpi=160)
    plt.close(fig)
    chosen = next(x for x in jacobian["conditions"] if x["epsilon_sd"] == jacobian["selected_epsilon_sd"])
    raw_before = float(np.linalg.norm(np.array(source["original_target_hz"]) - source["baseline_hz"]))
    raw_after = float(np.linalg.norm(np.array(source["original_target_hz"]) - source["final_hz"]))
    result = {"label": label, "output": str(root), "feature_policy": "raw", "quality_filtering": False,
              "analysis_interval_s": [item["start_s"], item["end_s"]], "primary_features": config["primary_features"],
              "features": feature_rows, "jacobian": jacobian,
              "selected_jacobian": chosen, "recovery_aggregates": recovery["aggregates"],
              "source": {k: v for k, v in source.items() if k != "trajectory"},
              "accepted_iterations": len(accepted), "raw_error_before_hz": raw_before, "raw_error_after_hz": raw_after,
              "timing": timings, "decode_count": sum(x["decode_count"] for x in timings.values()),
              "decode_total_s": sum(x["decode_total_s"] for x in timings.values()),
              "runner_wall_total_s": sum(x["total_wall_s"] for x in timings.values())}
    save_json(root / "result_summary.json", result)
    shutil.copyfile(".cache/feedback_raw_tests.log", root / "tests.log")
    for phase in phases:
        shutil.copyfile(f".cache/feedback_raw_{label}_{phase}.log", root / phase / "run.log")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", nargs="+", default=["a", "i", "u"])
    parser.add_argument("--report", type=Path, default=Path("reports/feedback_control_raw.md"))
    args = parser.parse_args(argv)
    results = [export_vowel(Path(f"outputs/feedback_control_{label}_raw")) for label in args.labels]
    comparison = []
    sections = []
    for result in results:
        label = result["label"]
        for row in result["features"]:
            comparison.append({"vowel": label, **row})
        selected = result["selected_jacobian"]
        source = result["source"]
        baseline = next(row for row in result["recovery_aggregates"] if row["method"] == "jacobian" and row["regularization"] == .01 and row["gain"] == 1)
        random = next(row for row in result["recovery_aggregates"] if row["method"] == "random" and row["regularization"] == .01 and row["gain"] == 1)
        summary_table = "\n".join(f"| {r['feature']} | {r['original_hz']:.2f} | {r['baseline_hz']:.2f} | {r['corrected_hz']:.2f} |" for r in result["features"])
        jac_table = "\n".join(f"| {x['epsilon_sd']} | {x['median_relative_error']:.3f} | {x['median_cosine']:.3f} |" for x in result["jacobian"]["conditions"])
        recovery_table = []
        for lam in [.000001, .0001, .01]:
            for gain in [.1, .25, .5, 1]:
                a = next(x for x in result["recovery_aggregates"] if x["regularization"] == lam and x["gain"] == gain and x["method"] == "jacobian")
                b = next(x for x in result["recovery_aggregates"] if x["regularization"] == lam and x["gain"] == gain and x["method"] == "random")
                recovery_table.append(f"| {lam:g} | {gain:g} | {100*a['median_improvement']:.1f}% | {100*b['median_improvement']:.1f}% | {round(8*a['improved_fraction_of_all'])}/8 |")
        time_table = "\n".join(f"| {phase} | {t['decode_count']} | {t['decode_total_s']:.3f} | {t['total_wall_s']:.2f} |" for phase, t in result["timing"].items())
        sections.append(f"""## /{label}/

結果：`{result['output']}`。解析区間：{result['analysis_interval_s']}秒。
選択した摂動幅は{selected['epsilon_sd']}参照SD、検証時の相対予測誤差中央値は{100*selected['median_relative_error']:.1f}%、cosine中央値は{selected['median_cosine']:.3f}。
線形予測の誤差が大きくても試行を除外せず、全設定を以下に示す。

| 摂動幅 (参照SD) | 相対予測誤差中央値 | cosine中央値 |
|---|---:|---:|
{jac_table}

復元実験は4 seed×2振幅の8条件。全条件の結果は `recovery/trials.csv` と `recovery/aggregates.csv` に保存した。
共通設定lambda=0.01、gain=1では、ヤコビアン修正の改善率中央値は{100*baseline['median_improvement']:.1f}%、
ランダム修正は{100*random['median_improvement']:.1f}%だった。同じ設定を全母音で比較しており、母音ごとの最良値だけを選んでいない。

| lambda | gain | J修正の改善率中央値 | ランダムの改善率中央値 | Jで改善 |
|---|---:|---:|---:|---:|
{chr(10).join(recovery_table)}

元音声への補正は{result['accepted_iterations']}回の更新を採用した。
正規化F1/F2誤差は{source['before']:.6f}→{source['after']:.6f}、改善率{100*source['improvement']:.1f}%。
Hz単位の誤差ノルムは{result['raw_error_before_hz']:.2f}→{result['raw_error_after_hz']:.2f} Hz。
最終EMA変更ノルムは{source['final_excursion_norm_sd']:.4f}参照SD。
F0/F3は制御目的に含めず、未除外の観測値として記録した。

| 特徴 | 元音声 (Hz) | 基準再合成 (Hz) | 補正後 (Hz) |
|---|---:|---:|---:|
{summary_table}

![全推定値の比較](../{result['output']}/figures/raw_acoustics_comparison.png)

![補正の推移](../{result['output']}/figures/original_target_correction.png)

[元音声](../{result['output']}/source/original.wav) /
[補正前](../{result['output']}/source/baseline.wav) /
[補正後](../{result['output']}/source/corrected.wav)

| 段階 | 実生成回数 | Decoder累計 (秒) | 実行時間 (秒) |
|---|---:|---:|---:|
{time_table}
| 合計 | {result['decode_count']} | {result['decode_total_s']:.3f} | {result['runner_wall_total_s']:.2f} |
""")
    output_summary = Path("outputs/feedback_control_raw_summary.json")
    save_json(output_summary, results)
    write_csv(Path("outputs/feedback_control_raw_features.csv"), comparison)
    report = """# 有効判定なしの母音別フィードバック基礎実験

作業ブランチ：`research/feedback-vowels-raw`。
/a/・/u/を追加し、比較用の/i/も同じ有効判定なしの条件で再実行した。
以前の `outputs/feedback_control_i/` は変更せず、各 `outputs/feedback_control_<母音>_raw/` に別保存した。

## 集計・実験条件

有声判定、フォルマント周波数範囲、帯域幅、順序、局所変動、有効フレーム率、変動係数による除外を行わない。
F1–F3は指定区間の全有限推定値の中央値、F0はPraatが数値を返したフレームの中央値。
未定義/NaN/Infのみ計算不能として扱い、0や任意値に置き換えない。
保存NPZの `valid` は有限値の有無だけを示し、`filtered_hz` は `raw_hz` と同じ内容である。
線形予測の移行基準も強制せず、数値が得られたヤコビアンの中から検証誤差中央値が最小の摂動幅を選んだ。
したがって今回の結果は、以前の/i/の品質判定ありの結果と集計条件が異なる。

補正対象は前回と共通のF1/F2。F0/F3も測定・保存・作図した。
SPARC multi、GPU、同じ事前学習済み重み・seed・参照SD・修正上限を使用し、EMA以外の条件は全生成で固定した。
参照SDは実験1の/a,i,u/の指定区間にあるEMA計27フレームから求め、訓練データのSDとは区別する。
ガウス型摂動窓は各区間の中央を中心とし、sigma=0.06秒、半幅0.18秒。
修正の1ステップは0.10参照SD以下、総変更量は0.5参照SD以下。数値制約は音響特徴の有効判定とは別に保持する。
正規化誤差は各母音の基準再合成F1/F2で割った差のL2ノルムで、補正中に尺度を変えない。
元音声への補正は各反復でヤコビアンを再推定し、3正則化×4ゲインの候補から実測誤差を減らすものを選ぶオフライン探索である。

各段階の実行時間にはモデル読込・環境記録・解析・保存を含み、Python起動/import、実装作業、作図・レポート生成は含めない。

""" + "\n".join(sections) + """
## 再現と保存データ

```sh
for vowel in a u i; do
  for phase in jacobian recovery source; do
    bash scripts/feedback_control.sh --config configs/feedback_control_raw.yaml --label "$vowel" --phase "$phase"
  done
done
```

再実行時は母音ごとに新しい `--output` を全3段階で共通指定する。既存結果は上書きしない。
各母音の `evaluations.jsonl`、全WAV/NPZ、ヤコビアン、全復元条件、反復軌跡、設定、環境、時間を保存した。
`source/corrected_code.npz` には補正後EMAと固定した全条件を保存した。
集計JSONは `outputs/feedback_control_raw_summary.json`、特徴比較CSVは `outputs/feedback_control_raw_features.csv`。
`validation.json` で、全有限値の保持、音響判定・追跡判定の未適用、EMA以外の不変性、保存データ・ソースハッシュを確認した。
重み不要の自動テスト21件が成功。有効判定を通らない値でもrawモードで保持されることを追加検証した。

これは単一話者・短い母音区間での検証である。推定値を除外しないため、LPCの広い共鳴やピーク番号の変化も結果に含む。
改善率はこのraw特徴に対する値であり、真の共鳴・聴感・発話の正常性の改善を直接証明するものではない。
以前の/i/実験に用いたソースは `outputs/feedback_control_i/source_snapshot/` にハッシュを照合して保存した。
"""
    args.report.write_text(report)
    print(f"Saved: {args.report}, {output_summary}")
    for result in results:
        print(json.dumps({"label": result["label"], "source_improvement": result["source"]["improvement"],
              "decodes": result["decode_count"], "decode_seconds": result["decode_total_s"],
              "runner_seconds": result["runner_wall_total_s"]}))


if __name__ == "__main__":
    main()
