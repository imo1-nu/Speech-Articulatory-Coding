"""Compare saved PPG and WavLM held-start control without rerunning either search."""

import argparse
import csv
import html
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch

from . import evaluate_generated_audio as metrics
from .compare_jsut_audio import audio_player
from .feedback_features import FeatureObjective
from .japanese_ppg import sha256_file
from .model_jsut_tokens import write_json
from .sequential_ppg_search import save_csv


COMMON = ('original','original_ema','normal_rvq','held_exact','held_rvq','unedited')
LABELS = dict(zip(COMMON,('元音声（共通目標）','元軌道合成','通常RVQ8再構成',
                        '位置を厳密に固定','固定軌道をRVQ8で再構成','逐次・固定トークンのまま')))
LABELS.update({'wavlm_no_feedback':'WavLM・FBなし','wavlm_feedback':'WavLM・FBあり',
               'ppg_no_feedback':'PPG・FBなし','ppg_feedback':'PPG・FBあり'})
TEXT_FIELDS = {'condition','utterance_id','audio_sha256','reference_text','asr_text',
               'normalized_reference','normalized_hypothesis'}


def read_csv(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def numeric_rows(rows):
    return [{k:(v if k in TEXT_FIELDS else None if v in ('',None) else float(v))
             for k,v in row.items()} for row in rows]


def indexed_records(manifest):
    result = {(r['utterance_id'],r['condition']):r for r in manifest['records']}
    if len(result)!=len(manifest['records']):raise ValueError('Duplicate source records')
    return result


def validate_pair(directories,manifests):
    w,p = manifests['wavlm'],manifests['ppg']
    if not w['complete'] or not p['complete']:raise ValueError('Incomplete source experiment')
    if w['settings'].get('objective','wavlm')!='wavlm' or p['settings'].get('objective')!='ppg':
        raise ValueError('Incorrect source objectives')
    ids = [{r['utterance_id']:r for r in m['inputs']} for m in (w,p)]
    if set(ids[0])!=set(ids[1]) or not ids[0]:raise ValueError('Different utterance sets')
    for key in ('initial_position_frame','horizon_ms','step_ms','delay_ms','transition_weight'):
        if w['settings'][key]!=p['settings'][key]:raise ValueError(f'Different control setting: {key}')
    if w['generator_sha256']!=p['generator_sha256']:raise ValueError('Different synthesizers')
    for suffix in ('codebooks.npz','transitions.npz'):
        hashes = [[h for name,h in m['dependencies'].items() if name.endswith(suffix)] for m in (w,p)]
        if len(hashes[0])!=1 or hashes[0]!=hashes[1]:raise ValueError('Different RVQ/prior')
    for source in ('feature_feedback_search.py','feedback_features.py','trajectory_rvq.py','local_ppg_search.py'):
        if w['sources'][source]!=p['sources'][source]:raise ValueError('Different shared search implementation')
    indices = {k:indexed_records(m) for k,m in manifests.items()}
    checked = 0
    for uid in ids[0]:
        if ids[0][uid]!=ids[1][uid]:raise ValueError('Different source utterance metadata')
        folder = ids[0][uid]['directory']
        with np.load(directories['wavlm']/folder/'initialization.npz') as a, np.load(directories['ppg']/folder/'initialization.npz') as b:
            for key in ('first_position','held_ema','held_rvq_ema','initial_tokens','original_ema','pitch','loudness','spk_emb'):
                np.testing.assert_array_equal(a[key],b[key])
        for condition in COMMON:
            pair = [indices[k][uid,condition] for k in ('wavlm','ppg')]
            if pair[0]['audio_sha256']!=pair[1]['audio_sha256']:raise ValueError('Shared control waveform differs')
            for kind,row in zip(('wavlm','ppg'),pair):
                if sha256_file(directories[kind]/row['audio'])!=row['audio_sha256']:raise ValueError('Source waveform changed')
            checked+=1
    return {'utterances':len(ids[0]),'shared_waveform_pairs_byte_identical':checked,
            'initial_positions_tokens_and_phonation_identical':True,'control_settings_and_search_code_match':True}


def write_report(output,manifest,audio_rows):
    groups = metrics.aggregate(numeric_rows(audio_rows));save_csv(output/'audio_aggregates.csv',groups)
    table = ''.join(f'<tr><td>{r["condition"]}</td><td>{LABELS[r["condition"]]}</td><td>{r["utterances"]}</td>'
                    f'<td>{100*r["cer"]:.2f}</td><td>{r["mcd_db"]:.3f}</td><td>{r["stoi"]:.4f}</td></tr>' for r in groups)
    features = ''
    for condition in LABELS:
        rows = [r for r in manifest['records'] if r['condition']==condition]
        if rows:
            features += f'<tr><td>{LABELS[condition]}</td><td>{np.mean([r["wavlm_distance"] for r in rows]):.5f}</td><td>{np.mean([r["ppg_js"] for r in rows]):.5f}</td></tr>'
    cards = []
    for source in manifest['inputs']:
        uid = source['utterance_id'];folder = source['directory']
        players = ''.join(f'<div><h3>{html.escape(LABELS[c])}</h3>{audio_player(r["audio"])}</div>'
                          for c in LABELS for r in manifest['records'] if r['utterance_id']==uid and r['condition']==c)
        texts = ''.join(f'<li>{LABELS[r["condition"]]}: {html.escape(r["asr_text"])}</li>' for r in audio_rows if r['utterance_id']==uid)
        plots = ''.join(f'<details><summary>{kind.upper()}の調音軌道</summary><img src="{folder}/{kind}_trajectories.png" style="width:100%"></details>' for kind in ('wavlm','ppg'))
        cards.append(f'<section><h2>{uid}</h2><p>{html.escape(source["text"])}</p><div class="players">{players}</div>{plots}<ul>{texts}</ul></section>')
    page = '''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>固定初期位置からのPPG・WavLM編集比較</title><style>body{font-family:system-ui;max-width:1500px;margin:25px auto;padding:15px}.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:15px}audio{width:100%}td,th{padding:8px;border-bottom:1px solid #ccc}section{border:1px solid #ccc;margin:20px 0;padding:16px}</style>
<h1>最初の調音位置を固定した状態から：PPGとWavLMの比較</h1>
<p>同じテスト5発話・同じ初期位置・RVQ8トークン・F0・音量・話者・合成ゲインを使用。
元音声のPPGまたはWavLMを目標として、固定状態から逐次編集します。WavLM音声は前回の保存音声を再利用しています。</p>
<p>40 ms窓・50 Hz・20 ms更新・60 ms観測遅延・400 ms探索窓、全8段編集、1段目遷移NLL重み0.03。
PPG JSとWavLM cosine距離は尺度が異なり、同じ重みを同じ実効的制約強度とは解釈しません。
FBなしも候補音声に基づく探索を行い、FBありでは出力と発行時予測の差で予測を補正します。
両方式で同じ音声入力遅延を課しますが、PPGの最初の観測は100 ms、WavLMは80 msからです。目標特徴量の全体は事前に既知です。</p>
<p>MCD/STOIの参照は元軌道合成、CERはWhisper large-v3とJSUT正解文。固定条件の反復認識による挿入誤りでCERが100%を超える場合があります。
音質・認識・特徴量距離を別々に確認してください。<a href="manifest.json">設定・入力検証</a> / <a href="audio_metrics.csv">発話別音声指標</a></p>
<h2>特徴量距離（発話平均）</h2><table><tr><th>条件</th><th>WavLM距離 ↓</th><th>PPG JS ↓</th></tr>'''
    page += features+'</table><h2>音声指標</h2><table><tr><th>条件ID</th><th>説明</th><th>発話数</th><th>CER % ↓</th><th>MCD dB ↓</th><th>STOI ↑</th></tr>'+table+'</table>'+''.join(cards)
    page += "<script>document.querySelectorAll('audio').forEach(p=>p.onplay=()=>document.querySelectorAll('audio').forEach(q=>{if(q!==p)q.pause()}));</script></html>"
    (output/'index.html').write_text(page,encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wavlm-dir',type=Path,default=Path('outputs/jsut_held_start_wavlm_feedback'))
    parser.add_argument('--ppg-dir',type=Path,default=Path('outputs/jsut_held_start_ppg_feedback'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/jsut_held_start_feature_comparison'))
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args();torch.set_num_threads(4)
    directories = {'wavlm':args.wavlm_dir,'ppg':args.ppg_dir}
    manifests = {k:json.loads((p/'manifest.json').read_text()) for k,p in directories.items()}
    for m in manifests.values():
        for path,digest in m['dependencies'].items():
            if sha256_file(Path(path))!=digest:raise ValueError('Changed source dependency')
    verification = validate_pair(directories,manifests)
    signature = {str(path/'manifest.json'):sha256_file(path/'manifest.json') for path in directories.values()}
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        saved = json.loads((args.output_dir/'manifest.json').read_text())
        if saved['inputs_signature']!=signature:raise ValueError('Existing comparison has different sources')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    features = {k:FeatureObjective(k,args.device) for k in directories}
    for kind in directories:
        if features[kind].metadata!=manifests[kind]['feature']:raise ValueError('Feature checkpoint changed')
    audio_indices = {k:{(r['utterance_id'],r['condition']):r for r in read_csv(p/'audio_metrics.csv')} for k,p in directories.items()}
    record_indices = {k:indexed_records(m) for k,m in manifests.items()}
    manifest = {'complete':False,'inputs_signature':signature,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'source_sha256':sha256_file(Path(__file__)),'verification':verification,
                'sources':{name:sha256_file(Path(__file__).with_name(name)) for name in
                           ('compare_held_start_features.py','feedback_features.py','japanese_ppg.py','evaluate_generated_audio.py')},
                'features':{k:f.metadata for k,f in features.items()},'inputs':manifests['wavlm']['inputs'],'records':[],
                'settings':{'shared':manifests['ppg']['settings'],'source_runs':{k:str(v) for k,v in directories.items()}},
                'audio_metric_reference':'original_ema','asr_model':manifests['ppg']['asr_model'],
                'asr_revision':manifests['ppg']['asr_revision'],'asr_settings':manifests['ppg']['asr_settings']}
    if any(manifests['wavlm'][k]!=manifest[k] for k in ('asr_model','asr_revision','asr_settings')):raise ValueError('Different ASR settings')
    rows = []
    for source in manifest['inputs']:
        uid = source['utterance_id'];folder = source['directory'];destination = args.output_dir/folder;destination.mkdir(parents=True,exist_ok=True)
        target_audio = metrics.read_audio(args.wavlm_dir/folder/'original.wav')
        targets = {k:f.extract(target_audio[None])[0] for k,f in features.items()}
        for condition in LABELS:
            kind = 'wavlm' if condition in COMMON else condition.split('_',1)[0]
            source_condition = condition if condition in COMMON else condition[len(kind)+1:]
            record = record_indices[kind][uid,source_condition];path = directories[kind]/record['audio']
            if sha256_file(path)!=record['audio_sha256']:raise ValueError('Source audio changed')
            relative = f'{folder}/{condition}.wav';shutil.copyfile(path,args.output_dir/relative)
            row = dict(audio_indices[kind][uid,source_condition]);row['condition']=condition
            if row['audio_sha256']!=record['audio_sha256']:raise ValueError('ASR waveform provenance mismatch')
            rows.append(row)
            wave = metrics.read_audio(path);distances = {}
            for feature_name,feature in features.items():
                values = feature.extract(wave[None],fresh=True)[0];target = targets[feature_name];n = min(len(values),len(target))
                distances['ppg_js' if feature_name=='ppg' else 'wavlm_distance'] = float(feature.distances(target[:n],values[:n]).mean())
            native_key = 'ppg_js' if kind=='ppg' else 'wavlm_distance'
            if abs(distances[native_key]-record[native_key])>1e-5:raise ValueError('Stored search distance differs from fresh comparison')
            manifest['records'].append({'condition':condition,'utterance_id':uid,'audio':relative,'audio_sha256':record['audio_sha256'],
                                        'source_run':str(directories[kind]),'source_condition':source_condition,**distances})
        for kind,directory in directories.items():shutil.copyfile(directory/folder/'trajectories.png',destination/f'{kind}_trajectories.png')
        print(f'Compared {uid}',flush=True)
    save_csv(args.output_dir/'audio_metrics.csv',rows);save_csv(args.output_dir/'feature_metrics.csv',manifest['records'])
    manifest['complete']=True;write_report(args.output_dir,manifest,rows);write_json(args.output_dir/'manifest.json',manifest)
    print(f'Report: {args.output_dir}/index.html',flush=True)


if __name__=='__main__':main()
