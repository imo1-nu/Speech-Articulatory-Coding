"""Compare frozen-AE RVQ prefixes with the verified VQ/continuous-AE ablation."""

import argparse
import csv
import html
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from . import evaluate_generated_audio as metrics
from .compare_jsut_audio import audio_player
from .compare_trajectory_reconstruction import trajectory_statistics, summarize_trajectories, synthesize
from .japanese_ppg import sha256_file
from .local_ppg_search import load_generator
from .model_jsut_tokens import checked_feature, load_run, write_json
from .sequential_ppg_search import save_csv
from .trajectory_rvq import load_rvq
from .trajectory_vq import WindowDataset, overlap_average


STAGES = (1, 2, 4, 8)


def report(output, manifest, rows):
    aggregates = metrics.aggregate(rows)
    table = ''.join(f'<tr><td>{r["condition"]}</td><td>{100*r["cer"]:.2f}</td><td>{r["mcd_db"]:.4f}</td>'
                    f'<td>{r["stoi"]:.4f}</td><td>{r["f0_rmse_hz"]:.2f}</td><td>{100*r["vuv_error_rate"]:.2f}</td></tr>' for r in aggregates)
    trajectory = ''.join(f'<tr><td>{name}</td><td>{r["normalized_mse"]:.7f}</td><td>{r["delta_mse"]:.7f}</td>'
                         f'<td>{r["velocity_energy_ratio"]:.4f}</td><td>{r["trajectory_6_to_10_hz_energy_ratio"]:.4f}</td></tr>'
                         for name,r in manifest['test_trajectory_metrics'].items())
    cards = []
    for record in manifest['audio_records']:
        folder = record['directory']
        players = ''.join(f'<div><h3>{v}</h3>{audio_player(folder+"/"+v+".wav")}</div>' for v in manifest['variants'])
        transcripts = ''.join(f'<li>{r["condition"]}: {html.escape(r["asr_text"])} (CER {100*r["cer"]:.1f}%)</li>'
                              for r in rows if r['utterance_id'] == record['utterance_id'])
        events = []
        for event in record['k_events']:
            clips = ''.join(f'<div><h3>{v}</h3>{audio_player(event["directory"]+"/"+v+".wav")}</div>' for v in manifest['variants'])
            events.append(f'<details><summary>参照PPGのk/ky候補 {event["start_seconds"]:.2f}–{event["end_seconds"]:.2f}s</summary><div class="players">{clips}</div></details>')
        cards.append(f'<section><h2>{record["utterance_id"]}</h2><p>{html.escape(record["text"])}</p><div class="players">{players}</div><ul>{transcripts}</ul>'+''.join(events)+'</section>')
    usage = ''.join(f'<li>段{i+1}: テスト使用 {r["used"]}/128、perplexity {r["perplexity"]:.2f}</li>' for i,r in enumerate(manifest['test_code_usage']))
    content = '''<!doctype html><html lang="ja"><meta charset="utf-8"><title>RVQ再構成比較</title>
<style>body{font-family:system-ui;max-width:1400px;margin:30px auto;padding:16px}.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}audio{width:100%}td,th{padding:8px;border-bottom:1px solid #ddd}section{border:1px solid #ccc;margin:20px 0;padding:16px}</style>
<h1>40ms・50Hz：RVQ / VQ / 連続AE</h1>
<p>RVQは連続AEを固定し、訓練4500発話の潜在残差だけで各段128中心をMiniBatchKMeans学習。段数1/2/4/8は同じ8段モデルの接頭部です。
従来VQはAEと量子化器を同時学習しており、学習方法も異なります。RVQは350/700/1400/2800 bit/s、VQ128は350、VQ512は450 bit/s（トークンのみ、エントロピー符号化なし）。連続AEは離散符号ではありません。</p>
<p>軌道はテスト250発話。音声は既存5発話でF0・音量・話者・ゲイン固定。音響指標の参照は元軌道合成、CERはWhisper large-v3とJSUT正解文。
DTWなし。k/ky候補は既存の参照PPGから選択した区間で、正解音素境界ではありません。音声5発話の結果は予備評価です。</p>
<p><a href="manifest.json">設定・軌道・使用コード</a> / <a href="audio_metrics.csv">発話別音声指標</a> / <a href="audio_aggregates.csv">音声集計</a></p>
<h2>テスト軌道</h2><table><tr><th>条件</th><th>MSE ↓</th><th>差分MSE ↓</th><th>速度エネルギー比</th><th>6–10Hz比</th></tr>'''
    content += trajectory+'</table><h2>音声</h2><table><tr><th>条件</th><th>CER % ↓</th><th>MCD dB ↓</th><th>STOI ↑</th><th>F0 RMSE Hz ↓</th><th>VUV % ↓</th></tr>'+table+'</table><h2>各RVQ段のコード使用</h2><ul>'+usage+'</ul>'+''.join(cards)
    content += "<script>document.querySelectorAll('audio').forEach(p=>p.onplay=()=>document.querySelectorAll('audio').forEach(q=>{if(q!==p)q.pause()}));</script></html>"
    (output/'index.html').write_text(content, encoding='utf-8')
    save_csv(output/'audio_aggregates.csv', aggregates)
    return aggregates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path('outputs/jsut_rvq/model'))
    parser.add_argument('--baseline-dir', type=Path, default=Path('outputs/jsut_reconstruction_ablation/comparison'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/jsut_rvq/comparison'))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Choose a new output directory')
    torch.set_num_threads(4)
    baseline = json.loads((args.baseline_dir/'manifest.json').read_text())
    rvq_config = json.loads((args.model_dir/'config.json').read_text())
    config, splits, _, mean, std = load_run(Path(rvq_config['reference_dir']), 'cpu')
    expected_asr = {'language':'ja','task':'transcribe','num_beams':1,'do_sample':False,'max_new_tokens':256,'prompt':None}
    if (rvq_config['ae_sha256'] != baseline['ae_sha256']
            or rvq_config['reference_checkpoint_sha256'] != baseline['reference_vq128_sha256']
            or json.loads((args.model_dir/'splits.json').read_text()) != splits
            or baseline['asr_model'] != metrics.ASR_MODEL or baseline['asr_revision'] != metrics.ASR_REVISION
            or baseline['asr_settings'] != expected_asr):
        raise ValueError('Baseline/model provenance mismatch')
    models = {f'rvq{n}':load_rvq(args.model_dir,n,args.device)[0] for n in STAGES}
    totals = {name:[] for name in models}; counts = np.zeros((8,128),dtype=np.int64)
    records = baseline['audio_records']; wanted = {r['directory'] for r in records}; cache = {}
    args.output_dir.mkdir(parents=True)
    for relative in tqdm(splits['test'], desc='RVQ test250'):
        ema = checked_feature(config, relative)['ema']
        windows = WindowDataset([ema], mean, std, 2, 1)
        batch = torch.stack([windows[i] for i in range(len(windows))]).to(args.device)
        with torch.inference_mode():
            full = models['rvq8']; _, tokens = full.quantize(full.ae.encoder(batch))
        tokens = tokens.cpu().numpy()
        for i in range(8): counts[i] += np.bincount(tokens[:,i],minlength=128)
        reconstructions = {}
        for name,model in models.items():
            # Synthesis receives only decoded token IDs; there is no continuous latent bypass.
            with torch.inference_mode():
                decoded = model.decode_tokens(torch.from_numpy(tokens[:,:model.stages]).to(args.device))
            reconstruction = overlap_average(decoded.cpu().numpy().reshape(-1,2,12),output_length=len(ema))*std+mean
            totals[name].append(trajectory_statistics(ema,reconstruction,std))
            reconstructions[name] = reconstruction
        folder = Path(relative).with_suffix('').as_posix()
        token_path = args.output_dir/'tokens'/Path(relative)
        token_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(token_path,tokens=tokens)
        if folder in wanted: cache[folder] = reconstructions
    usage = []
    for c in counts:
        p = c[c>0]/c.sum()
        usage.append({'used':int((c>0).sum()),'counts':c.tolist(),'perplexity':float(np.exp(-(p*np.log(p)).sum()))})
    manifest = {'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'source_sha256':sha256_file(Path(__file__)), 'rvq_source_sha256':sha256_file(Path(__file__).with_name('trajectory_rvq.py')),
                'metric_source_sha256':sha256_file(Path(metrics.__file__)),
                'packages':{name:importlib.metadata.version(name) for name in ['torch','numpy','scikit-learn','pyworld','pysptk','pystoi','transformers']},
                'codebooks_sha256':sha256_file(args.model_dir/'codebooks.npz'), 'rvq_config':rvq_config,
                'baseline_manifest_sha256':sha256_file(args.baseline_dir/'manifest.json'),
                'asr_model':metrics.ASR_MODEL,'asr_revision':metrics.ASR_REVISION,'asr_settings':baseline['asr_settings'],
                'audio_metric_reference':'original trajectory synthesis','cer_reference':'JSUT gold text',
                'test_trajectory_metrics':{**baseline['test_trajectory_metrics'],**summarize_trajectories(totals)},
                'test_code_usage':usage,'variants':['original','original_ema','vq128','vq512','continuous_ae',*models],
                'audio_records':[]}
    with (args.baseline_dir/'audio_metrics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    numeric = [k for k in rows[0] if k not in ('condition','utterance_id','audio_sha256','reference_text','asr_text','normalized_reference','normalized_hypothesis')]
    for row in rows:
        for k in numeric: row[k] = json.loads(row[k]) if row[k] else None
    # Reuse only metrics for byte-identical baseline audio and pinned evaluation settings.
    for row in rows:
        record = next(r for r in records if r['utterance_id']==row['utterance_id'])
        if sha256_file(args.baseline_dir/record['directory']/f'{row["condition"]}.wav') != row['audio_sha256']:
            raise ValueError('Baseline waveform changed')
    generator,generator_hash = load_generator(args.device)
    if generator_hash != baseline['generator_sha256']: raise ValueError('Generator changed')
    recognizer = metrics.WhisperRecognizer(args.device)
    source = json.loads((Path(rvq_config['reference_dir'])/'audio_comparison/manifest.json').read_text())
    for record in records:
        folder = record['directory']; directory = args.output_dir/folder; directory.mkdir(parents=True)
        base = args.baseline_dir/folder
        for variant in manifest['variants'][:5]: shutil.copyfile(base/f'{variant}.wav',directory/f'{variant}.wav')
        reference = metrics.read_audio(base/'original_ema.wav'); ref_features = metrics.world_features(reference)
        old = next(r for r in source['records'] if r['utterance_id']==record['utterance_id'])
        fixed = dict(np.load(Path(rvq_config['reference_dir'])/'audio_comparison'/folder/'conditions.npz'))
        for name,ema in cache[folder].items():
            wave = synthesize(generator,ema,fixed,old['synthesis_common_gain'],args.device)
            sf.write(directory/f'{name}.wav',wave,16000,subtype='PCM_16')
            text = recognizer.transcribe(wave)
            row = {'condition':name,'utterance_id':record['utterance_id'],'audio_sha256':sha256_file(directory/f'{name}.wav'),
                   'reference_text':record['text'],'asr_text':text,**metrics.character_errors(record['text'],text),
                   **metrics.acoustic_scores(reference,wave,ref_features)}
            rows.append(row)
            print(record['utterance_id'],name,f'CER={row["cer"]:.4f} MCD={row["mcd_db"]:.4f} STOI={row["stoi"]:.4f}',flush=True)
        np.savez_compressed(directory/'reconstructions.npz',**cache[folder])
        for event in record['k_events']:
            event_dir = args.output_dir/event['directory']; event_dir.mkdir(parents=True)
            lo = max(0,int((event['start_seconds']-.12)*16000)); hi = min(len(reference),int((event['end_seconds']+.15)*16000))
            for variant in manifest['variants']:
                wave = metrics.read_audio(directory/f'{variant}.wav')
                sf.write(event_dir/f'{variant}.wav',wave[lo:hi],16000,subtype='PCM_16')
        manifest['audio_records'].append(record)
        save_csv(args.output_dir/'audio_metrics.csv',rows)
        manifest['audio_aggregates'] = report(args.output_dir,manifest,rows)
        write_json(args.output_dir/'manifest.json',manifest)
    print(f'Report: {args.output_dir}/index.html',flush=True)


if __name__ == '__main__': main()
