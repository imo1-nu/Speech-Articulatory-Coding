"""Evaluate movement from source A toward target B using Whisper large-v3 and target-referenced audio metrics."""

import argparse
import html
import json
import os
from pathlib import Path

import numpy as np
import torch

from . import cross_utterance_ppg as cross
from . import evaluate_generated_audio as metrics
from .compare_jsut_audio import audio_player
from .japanese_ppg import sha256_file
from .model_jsut_tokens import write_json
from .sequential_ppg_search import save_csv


def aggregate(rows):
    aggregated=metrics.aggregate(rows)
    for group in aggregated:
        selected=[r for r in rows if r['condition']==group['condition']]
        group['cer_B']=group.pop('cer')
        group['cer_A']=sum(r['source_character_errors'] for r in selected)/sum(r['source_reference_characters'] for r in selected)
        group['ppg_js_to_B']=float(np.average([r['ppg_js_to_B'] for r in selected],weights=[r['ppg_frames'] for r in selected]))
    return aggregated


def write_report(output,rows,aggregates):
    def f(value):return '—' if value is None else f'{value:.3f}'
    table=''.join(f'<tr><td>{html.escape(cross.condition_label(r["condition"]))}</td>'
                  f'<td>{100*r["cer_B"]:.1f}%</td><td>{100*r["cer_A"]:.1f}%</td>'
                  f'<td>{r["ppg_js_to_B"]:.5f}</td><td>{f(r["mcd_db"])}</td><td>{f(r["stoi"])}</td>'
                  f'<td>{f(r["f0_rmse_hz"])}</td></tr>' for r in aggregates)
    cards=[]
    for r in rows:
        cards.append(f'<section><h3>{html.escape(r["pair_id"])} / {html.escape(cross.condition_label(r["condition"]))}</h3>'
                     f'{audio_player(r["audio_relative"])}<p>A：{html.escape(r["source_text"])}<br>B：{html.escape(r["reference_text"])}</p>'
                     f'<p>Whisper large-v3：{html.escape(r["asr_text"])}</p>'
                     f'<p>CER(B) {100*r["cer_B"]:.1f}% / CER(A) {100*r["cer_A"]:.1f}%</p></section>')
    content='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A→B変換のCERと音響評価</title>
<style>body{font-family:system-ui;max-width:1350px;margin:30px auto;padding:0 18px}td,th{padding:9px;border-bottom:1px solid #ddd}table{border-collapse:collapse}section{border:1px solid #ccc;padding:15px;margin:18px 0}audio{width:100%;max-width:500px}</style>
<h1>A→B変換：Whisper large-v3と音響指標</h1><p><a href="../index.html">制御実験</a> / <a href="summary.csv">ペア別CSV</a> / <a href="aggregates.csv">集計CSV</a></p>
<p>CER(B)はBの正解文への近さで、低いほどBの発話内容を認識できています。CER(A)はAの正解文との比較で、Aが残っているかの補助指標です。
CER(A)が増えただけではBへ変換できたとはいえません。両方が高ければ、AともBとも異なる認識結果です。
全音声を同じWhisper large-v3、日本語・文字起こし・greedy、最大新規256トークンで独立に認識します。正解文はASRへ渡しません。
CERはNFKC・小文字化・句読点等と空白を除去した文字編集距離で、漢字・かな・数字の表記差を同一化しません。長い挿入認識では100%を超えることがあります。</p>
<p>MCD・STOI・F0 RMSE・V/UV誤り率の参照はB原音です。AのF0・音量・話者を固定しているため、
音響指標には内容だけでなくA/Bの抑揚等の差も含まれます。Bの時間軸を保ち、DTW・時間ずらし・個別音量正規化を使いません。
MCDは24次mel-cepstrum・c0除外・参照RMS−40 dB以上、F0はWORLD DIO＋StoneMask・両有声フレーム、STOIは非extended版。
PPG差は最終的に出力した音声からの未補正PPGで評価します。これらの値は主観的な自然さの評価を代替しません。</p>
<table><tr><th>条件</th><th>CER(B) ↓</th><th>CER(A)</th><th>PPG JS(B) ↓</th><th>MCD(B,dB) ↓</th><th>STOI(B) ↑</th><th>F0 RMSE(B,Hz) ↓</th></tr>'''
    content+=table+'</table>'+''.join(cards)+'</html>'
    (output/'index.html').write_text(content,encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir',type=Path,default=Path('outputs/jsut_ppg_A_to_B'))
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--device',default='cuda');parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args(argv);output=args.output_dir or args.experiment_dir/'audio_metrics'
    if output.exists() and (not output.is_dir() or any(output.iterdir())):parser.error('Choose a new output directory')
    torch.set_num_threads(args.threads)
    experiment=json.loads((args.experiment_dir/'manifest.json').read_text())
    expected=(4+3*len(experiment['arguments']['horizons']))*len(experiment['pairs'])
    if len(experiment['records'])!=expected:parser.error('Control experiment is not complete')
    pairs={r['pair_id']:r for r in experiment['pairs']}
    references={pair_id:metrics.read_audio(args.experiment_dir/'pairs'/pair_id/'target_original/output.wav') for pair_id in pairs}
    reference_features={pair_id:metrics.world_features(audio) for pair_id,audio in references.items()}
    recognizer=metrics.WhisperRecognizer(args.device)
    cache={}
    prior=Path('outputs/jsut_ppg_delay060_step020/audio_metrics/manifest.json')
    if prior.exists():
        cached=json.loads(prior.read_text())
        if (cached['asr_model_id']==metrics.ASR_MODEL and cached['asr_revision']==metrics.ASR_REVISION
                and cached['language']=='ja' and cached['num_beams']==1 and cached['do_sample'] is False):
            cache={r['audio_sha256']:r['asr_text'] for r in cached['records']}
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    manifest={'asr_model_id':metrics.ASR_MODEL,'asr_revision':metrics.ASR_REVISION,
              'language':'ja','task':'transcribe','num_beams':1,'do_sample':False,'max_new_tokens':256,'prompt':None,
              'cer_reference_B':'target gold text','cer_reference_A':'source gold text',
              'audio_metric_reference':'original B audio, native timing; no DTW',
              'experiment_manifest_sha256':sha256_file(args.experiment_dir/'manifest.json'),
              'cache_manifest_sha256':sha256_file(prior) if prior.exists() else None,'records':rows}
    for index,item in enumerate(experiment['records']):
        pair=pairs[item['pair_id']];path=args.experiment_dir/item['audio'];digest=sha256_file(path)
        audio=metrics.read_audio(path); reused=digest in cache
        text=cache[digest] if reused else recognizer.transcribe(audio);cache[digest]=text
        source_errors=metrics.character_errors(pair['source_text'],text)
        target_errors=metrics.character_errors(pair['target_text'],text)
        row={'condition':item['condition'],'pair_id':item['pair_id'],'utterance_id':item['pair_id'],
             'source_id':pair['source_id'],'target_id':pair['target_id'],
             'audio_sha256':digest,'audio_relative':os.path.relpath(path.resolve(),output.resolve()),
             'source_text':pair['source_text'],'reference_text':pair['target_text'],'asr_text':text,'asr_cache_reused':reused,
             **target_errors,'cer_B':target_errors['cer'],'cer_A':source_errors['cer'],
             'source_character_errors':source_errors['character_errors'],
             'source_reference_characters':source_errors['reference_characters'],
             'ppg_js_to_B':item['ppg_js_to_B'],'ppg_frames':item['ppg_frames'],
             **metrics.acoustic_scores(references[item['pair_id']],audio,reference_features[item['pair_id']])}
        rows.append(row);aggregates=aggregate(rows)
        save_csv(output/'summary.csv',rows);save_csv(output/'aggregates.csv',aggregates)
        manifest['aggregates']=aggregates;write_json(output/'manifest.json',manifest);write_report(output,rows,aggregates)
        print(f'{index+1}/{expected} {item["pair_id"]} {item["condition"]}: CER(B)={row["cer_B"]:.3f} CER(A)={row["cer_A"]:.3f} ASR={text}',flush=True)
    write_json(output/'asr_cache.json',cache)
    print(f'Report: {output}/index.html',flush=True)
    return 0


if __name__=='__main__':main()
