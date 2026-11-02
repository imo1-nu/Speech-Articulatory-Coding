"""Compare transition-regularized token searches with identical models and search budgets."""

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
import torch

from . import local_ppg_search as search
from .compare_jsut_audio import audio_player
from .japanese_ppg import JapanesePPGExtractor, sha256_file
from .model_jsut_tokens import configure_plots, write_json


def aggregate(records):
    frames = np.array([r['common_ppg_frames'] for r in records])
    edges = np.array([r['token_count']-1 for r in records])
    result = {'transition_weight': records[0]['transition_weight'], 'utterances': len(records)}
    for stage in ['baseline', 'final']:
        result[f'{stage}_js'] = float(np.average([r[f'{stage}_js'] for r in records], weights=frames))
        for key in ['transition_nll', 'unseen_transition_fraction', 'self_transition_fraction']:
            result[f'{stage}_{key}'] = float(np.average([r[f'{stage}_{key}'] for r in records], weights=edges))
    result['edited_tokens'] = sum(r['edited_tokens'] for r in records)
    result['token_count'] = sum(r['token_count'] for r in records)
    return result


def write_comparison(output, runs):
    import matplotlib.pyplot as plt
    aggregates = [aggregate(run['records']) for run in runs]
    for filename, rows in [('summary.csv', [r for run in runs for r in run['records']]),
                           ('aggregates.csv', aggregates)]:
        with (output/filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = [f'{r["transition_weight"]:g}' for r in aggregates]
    axes[0].bar(labels, [r['final_js'] for r in aggregates])
    axes[0].axhline(aggregates[0]['baseline_js'], color='gray', linestyle='--', label='Before editing')
    axes[0].set(xlabel='Transition weight', ylabel='Mean JS (nats)'); axes[0].legend()
    axes[1].bar(labels, [r['final_transition_nll'] for r in aggregates])
    axes[1].axhline(aggregates[0]['baseline_transition_nll'], color='gray', linestyle='--', label='Before editing')
    axes[1].set(xlabel='Transition weight', ylabel='Mean transition NLL (nats/edge)'); axes[1].legend()
    fig.tight_layout(); fig.savefig(output/'tradeoff.png', dpi=150); plt.close(fig)
    table = ''.join(f'<tr><td>{r["transition_weight"]:g}</td><td>{r["final_js"]:.5f}</td>'
                    f'<td>{r["final_transition_nll"]:.4f}</td><td>{100*r["final_unseen_transition_fraction"]:.2f}%</td>'
                    f'<td>{100*r["final_self_transition_fraction"]:.2f}%</td><td>{r["edited_tokens"]}/{r["token_count"]}</td></tr>'
                    for r in aggregates)
    sections = []
    for index, first in enumerate(runs[0]['records']):
        folder = runs[0]['directory']+'/'+first['directory']
        players = ''.join(f'<div><h3>{label}</h3>{audio_player(folder+"/"+name+".wav")}</div>'
                          for name, label in [('original','元音声（目標）'), ('original_ema','元軌道から合成'), ('before','トークン編集前')])
        for run in runs:
            row = run['records'][index]
            path = run['directory']+'/'+row['directory']
            players += (f'<div><h3>λ={row["transition_weight"]:g}</h3>{audio_player(path+"/after.wav")}'
                        f'<p>JS {row["final_js"]:.5f} / 遷移NLL {row["final_transition_nll"]:.4f}<br>'
                        f'編集 {row["edited_tokens"]}/{row["token_count"]}</p>'
                        f'<a href="{path}/comparison.png">PPG図</a> / <a href="{run["directory"]}/index.html">詳細</a></div>')
        sections.append(f'<section><h2>{html.escape(first["utterance_id"])}</h2><p>{html.escape(first["text"])}</p>'
                        f'<div class="players">{players}</div></section>')
    content = '''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>遷移確率ソフト制約の比較</title><style>body{font-family:system-ui;max-width:1250px;margin:30px auto;padding:0 18px}.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:18px}section{border:1px solid #ccc;padding:18px;margin:24px 0}audio,img{width:100%}td,th{padding:10px;border-bottom:1px solid #ddd}h3{font-size:1rem}table{border-collapse:collapse}</style>
<h1>40 ms・128コード：遷移確率ソフト制約</h1><p>平均JS＋λ×平均遷移負対数確率を最小化しました。
遷移確率は学習4500発話のみから推定（加算平滑化α=1）、発話境界をまたがず、自己遷移を含みます。
各条件は同じ編集前トークン列から独立に開始し、F0・音量・話者情報・候補生成・探索予算を揃えています。
重みが正の場合、合計損失が低下すればPPG差が増える編集も採用します。遷移尤度の向上は音質改善を保証しません。</p>
<p>以下のJSはフレーム数、遷移指標は遷移数による重み付き平均です。<a href="summary.csv">発話別CSV</a> / <a href="aggregates.csv">集計CSV</a></p>
<table><tr><th>λ</th><th>平均JS ↓</th><th>遷移NLL ↓</th><th>未観測遷移率</th><th>自己遷移率</th><th>編集数</th></tr>'''
    content += table+'</table><img src="tradeoff.png" alt="PPG差と遷移NLLの比較">'+''.join(sections)
    content += "<script>document.querySelectorAll('audio').forEach(p=>p.onplay=()=>document.querySelectorAll('audio').forEach(q=>{if(q!==p)q.pause()}));</script></html>"
    (output/'index.html').write_text(content, encoding='utf-8')
    return aggregates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('outputs/jsut_window_sweep/window_040ms'))
    parser.add_argument('--ppg-dir', type=Path, default=Path('outputs/jsut_ppg_40ms128'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/jsut_ppg_soft_transition'))
    parser.add_argument('--weights', type=float, nargs='+', default=[0, .01, .03, .1])
    parser.add_argument('--utterances', nargs='+', default=search.DEFAULT_IDS)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    if (len(set(args.weights)) != len(args.weights) or not np.isfinite(args.weights).all()
            or min(args.weights) < 0 or args.threads < 1 or len(set(args.utterances)) != len(args.utterances)):
        parser.error('Invalid weights, utterances, or thread count')
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error('Output directory is not empty; choose a new --output-dir')
    torch.set_num_threads(args.threads)
    args.transition_alpha = 1.0
    prepared = search.prepare(args)
    generator, digest = search.load_generator(args.device)
    extractor = JapanesePPGExtractor(args.device)
    models = generator, digest, extractor
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_plots()
    runs = []
    for index, weight in enumerate(args.weights):
        directory = f'condition_{index:02d}_lambda_{weight:g}'
        destination = args.output_dir/directory
        child_args = ['--run-dir', str(args.run_dir), '--ppg-dir', str(args.ppg_dir),
                      '--output-dir', str(destination), '--transition-weight', str(weight),
                      '--transition-alpha', '1', '--device', args.device, '--threads', str(args.threads),
                      '--utterances', *args.utterances]
        search.main(child_args, prepared=prepared, models=models)
        run = json.loads((destination/'manifest.json').read_text())
        runs.append({'weight': weight, 'directory': directory, 'records': run['records'],
                     'manifest_sha256': sha256_file(destination/'manifest.json')})
        aggregates = write_comparison(args.output_dir, runs)
        write_json(args.output_dir/'manifest.json', {'weights': args.weights, 'alpha': 1,
                   'input_run_dir': str(args.run_dir.resolve()), 'input_ppg_dir': str(args.ppg_dir.resolve()),
                   'search_budget': {'passes': 2, 'positions_per_pass': 24, 'neighbors': 12, 'global_candidates': 4},
                   'runs': runs, 'aggregates': aggregates})
    print(f'Report: {args.output_dir}/index.html')
    return 0


if __name__ == '__main__':
    main()
