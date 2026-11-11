"""Add pinned UTMOS22 strong scores to saved audio comparisons without changing base metrics."""

import argparse
import csv
import html
import importlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np
import soundfile as sf
import torch

from .evaluate_generated_audio import read_audio
from .japanese_ppg import sha256_file
from .model_jsut_tokens import write_json
from .sequential_ppg_search import save_csv

MODEL = 'UTMOS22 strong / SpeechMOS'
REPOSITORY = 'https://github.com/tarepan/SpeechMOS.git'
REVISION = 'ed25eacbfa42b99156c36ebec67a733b5dbb9b79'
WEIGHTS_NAME = 'utmos22_strong_step7459_v1.pt'
WEIGHTS_URL = f'https://github.com/tarepan/SpeechMOS/releases/download/v1.0.0/{WEIGHTS_NAME}'
WEIGHTS_SHA256 = '38aa51ab79e2a4e09a1449758a4b37e9cbb2e8235a49662a732d33a9ba1e9bff'
SCORING_POLICY = 'full_clip_mono_16khz_single_no_extra_normalization_no_trim_no_clamp_v1'
MARKER_BEGIN = '<!-- UTMOS:BEGIN -->'
MARKER_END = '<!-- UTMOS:END -->'


def prepare_assets(cache_dir):
    cache_dir.mkdir(parents=True,exist_ok=True)
    source = cache_dir/'SpeechMOS'
    if not source.exists():
        subprocess.run(['git','clone','--depth','1','--branch','v1.2.0',REPOSITORY,str(source)],check=True)
    actual = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if actual != REVISION: raise ValueError('SpeechMOS revision differs from pinned version')
    subprocess.run(['git','-C',str(source),'diff','--exit-code','HEAD','--'],check=True,stdout=subprocess.DEVNULL)
    weights = cache_dir/WEIGHTS_NAME
    if not weights.exists():
        temporary = weights.with_suffix('.part')
        try:
            with urllib.request.urlopen(WEIGHTS_URL,timeout=60) as response, temporary.open('wb') as stream:
                shutil.copyfileobj(response,stream)
            if sha256_file(temporary)!=WEIGHTS_SHA256: raise ValueError('Downloaded UTMOS weight checksum mismatch')
            temporary.replace(weights)
        finally:
            if temporary.exists(): temporary.unlink()
    if sha256_file(weights)!=WEIGHTS_SHA256: raise ValueError('UTMOS weight checksum mismatch')
    tracked = subprocess.check_output(['git','-C',str(source),'ls-files','*.py'],text=True).splitlines()
    hashes = {p:sha256_file(source/p) for p in tracked}
    return source,weights,hashes


class UTMOSPredictor:
    def __init__(self,source,weights,device):
        # Import the inspected pinned package; do not execute a moving remote torch.hub entrypoint.
        sys.path.insert(0,str(source.resolve()))
        module = importlib.import_module('speechmos.utmos22.strong.model')
        if not Path(module.__file__).resolve().is_relative_to(source.resolve()):
            raise ValueError('A different speechmos package shadows the pinned source')
        self.model = module.UTMOS22Strong()
        self.model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
        self.device = device
        self.model = self.model.float().eval().requires_grad_(False).to(device)

    def score(self,wave):
        wave = np.asarray(wave,dtype=np.float32)
        if wave.ndim!=1 or len(wave)<400 or not np.isfinite(wave).all():
            raise ValueError('UTMOS requires a finite mono waveform of at least 400 samples at 16 kHz')
        with torch.inference_mode():
            result = self.model(torch.from_numpy(wave.copy())[None].to(self.device),16000)
        if result.shape!=(1,) or not torch.isfinite(result).all(): raise ValueError('Invalid UTMOS prediction')
        # The upstream forward already averages frames and applies score * 2 + 3.
        return float(result.item())


def read_csv(path):
    with path.open() as stream: return list(csv.DictReader(stream))


def collect_items(run_dir):
    manifest = json.loads((run_dir/'manifest.json').read_text())
    if manifest.get('complete') is False: raise ValueError('The base experiment is incomplete')
    sources = manifest.get('inputs',manifest.get('audio_records',[]))
    directories = {r['utterance_id']:r['directory'] for r in sources}
    paths = {(r['utterance_id'],r['condition']):r['audio'] for r in manifest.get('records',[]) if 'audio' in r and 'condition' in r}
    rows = read_csv(run_dir/'audio_metrics.csv');items=[];seen=set()
    for row in rows:
        key = row['utterance_id'],row['condition']
        if key in seen: raise ValueError('Duplicate utterance/condition in base metrics')
        seen.add(key)
        relative = paths.get(key,f'{directories[key[0]]}/{key[1]}.wav')
        path = run_dir/relative
        if not path.resolve().is_relative_to(run_dir.resolve()): raise ValueError('Audio path escapes run directory')
        if sha256_file(path)!=row['audio_sha256']: raise ValueError(f'Base waveform changed: {relative}')
        items.append({'condition':key[1],'utterance_id':key[0],'audio':relative,'audio_sha256':row['audio_sha256']})
    if not items: raise ValueError('No base audio metrics')
    return items,rows


def score_items(run_dir,items,predict,cache):
    rows=[]
    for item in items:
        digest=item['audio_sha256'];hit=digest in cache
        path=run_dir/item['audio'];info=sf.info(path)
        if not hit:
            wave=read_audio(path);value=float(predict(wave))
            if not np.isfinite(value): raise ValueError('Nonfinite UTMOS score')
            cache[digest]={'utmos':value,'samples_16k':len(wave)}
        value=cache[digest]
        rows.append({**item,'utmos':value['utmos'],'source_sample_rate':info.samplerate,
                     'samples_16k':value['samples_16k'],'cached':hit})
    return rows


def aggregate_utmos(rows):
    output=[]
    for condition in dict.fromkeys(r['condition'] for r in rows):
        values=np.asarray([r['utmos'] for r in rows if r['condition']==condition],dtype=np.float64)
        output.append({'condition':condition,'utmos_utterances':len(values),'utmos_mean':float(values.mean()),
                       'utmos_std':float(values.std(ddof=1)) if len(values)>1 else None,
                       'utmos_min':float(values.min()),'utmos_max':float(values.max())})
    return output


def join_aggregates(rows,aggregates):
    lookup={r['condition']:r for r in aggregates}
    if any(r['condition'] not in lookup for r in rows): raise ValueError('Condition missing from UTMOS aggregates')
    return [{**row,**{k:v for k,v in lookup[row['condition']].items() if k!='condition'}} for row in rows]


def enrich_report(path,aggregates,audio_aggregates):
    page=path.read_text(encoding='utf-8')
    page=re.sub(re.escape(MARKER_BEGIN)+r'.*?'+re.escape(MARKER_END),'',page,flags=re.S)
    audio={r['condition']:r for r in audio_aggregates}
    table=''
    for row in aggregates:
        deviation='—' if row['utmos_std'] is None else f'{row["utmos_std"]:.3f}'
        cer=100*float(audio[row['condition']]['cer'])
        table+=(f'<tr><td>{html.escape(row["condition"])}</td><td>{row["utmos_utterances"]}</td>'
                f'<td>{row["utmos_mean"]:.3f}</td><td>{deviation}</td><td>{cer:.2f}</td></tr>')
    section=(MARKER_BEGIN+'<section id="utmos-evaluation"><h2>UTMOS22 strong：参照なしの音声品質予測</h2>'
             '<p>高いほど良い予測スコアです。人によるMOS評点ではありません。モデルはSpeechMOSのUTMOS22 strongで、'
             'UTMOSの全アンサンブルやUTMOSv2とは異なります。16 kHz・1発話ずつ、追加の音量正規化・無音除去・長さパディングなしで評価。'
             '平均と標準偏差は発話を等重みで集計しています。日本語・今回の合成方式での聴感との対応は別途確認が必要です。</p>'
             '<p><a href="utmos/metrics.csv">発話別UTMOS</a> / <a href="utmos/aggregates.csv">UTMOS集計</a> / '
             '<a href="utmos/audio_aggregates_with_utmos.csv">CER等との比較CSV</a> / '
             '<a href="utmos/manifest.json">モデル版・重みハッシュ・評価設定</a></p>'
             '<table><tr><th>条件</th><th>発話数</th><th>UTMOS ↑</th><th>標準偏差</th><th>CER % ↓</th></tr>'+table+'</table></section>'+MARKER_END)
    anchor='<h2>特徴量・計算時間'
    if anchor in page: page=page.replace(anchor,section+anchor,1)
    elif '</body>' in page: page=page.replace('</body>',section+'</body>',1)
    elif '</html>' in page: page=page.replace('</html>',section+'</html>',1)
    else: page+=section
    path.write_text(page,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=Path('outputs/jsut_wavlm_feedback'))
    parser.add_argument('--cache-dir',type=Path,default=Path('.cache/utmos'))
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4)
    source,weights,hashes=prepare_assets(args.cache_dir)
    if args.prepare_only:print(f'UTMOS assets ready: {source} / {weights}');return
    items,base_rows=collect_items(args.run_dir)
    signature={'model':MODEL,'repository':REPOSITORY,'revision':REVISION,'source_hashes':hashes,
               'weights_url':WEIGHTS_URL,'weights_sha256':WEIGHTS_SHA256,'scoring_policy':SCORING_POLICY,
               'device':args.device,'packages':{p:importlib.metadata.version(p) for p in ('torch','torchaudio','numpy','soundfile')},
               'audio_loader_sha256':sha256_file(Path(__file__).with_name('evaluate_generated_audio.py'))}
    destination=args.run_dir/'utmos';destination.mkdir(exist_ok=True)
    cache={};previous_parameters=None
    if (destination/'manifest.json').exists():
        old=json.loads((destination/'manifest.json').read_text())
        if old['signature']==signature:
            previous_parameters=old.get('model_parameters')
            if sha256_file(destination/'metrics.csv')!=old['metrics_sha256']:raise ValueError('Cached UTMOS CSV changed')
            for row in read_csv(destination/'metrics.csv'):
                cache[row['audio_sha256']]={'utmos':float(row['utmos']),'samples_16k':int(row['samples_16k'])}
    predictor=None
    if any(item['audio_sha256'] not in cache for item in items):predictor=UTMOSPredictor(source,weights,args.device)
    started=time.monotonic()
    rows=score_items(args.run_dir,items,None if predictor is None else predictor.score,cache)
    aggregates=aggregate_utmos(rows)
    save_csv(destination/'metrics.csv',rows);save_csv(destination/'aggregates.csv',aggregates)
    scores={(r['utterance_id'],r['condition']):r['utmos'] for r in rows}
    save_csv(destination/'audio_metrics_with_utmos.csv',[{**r,'utmos':scores[(r['utterance_id'],r['condition'])]} for r in base_rows])
    base_aggregates=read_csv(args.run_dir/'audio_aggregates.csv')
    save_csv(destination/'audio_aggregates_with_utmos.csv',join_aggregates(base_aggregates,aggregates))
    for name in ('comparison_summary','comparison_summary_natural_reference','audio_aggregates_natural_reference'):
        path=args.run_dir/f'{name}.csv'
        if path.exists():save_csv(destination/f'{name}_with_utmos.csv',join_aggregates(read_csv(path),aggregates))
    for item in items:
        if sha256_file(args.run_dir/item['audio'])!=item['audio_sha256']:raise ValueError('Input waveform changed during evaluation')
    manifest={'signature':signature,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'evaluation_source_sha256':sha256_file(Path(__file__)),
              'base_manifest_sha256':sha256_file(args.run_dir/'manifest.json'),
              'base_metrics_sha256':sha256_file(args.run_dir/'audio_metrics.csv'),
              'metrics_sha256':sha256_file(destination/'metrics.csv'),'input_rows':len(items),
              'unique_waveforms':len({r['audio_sha256'] for r in rows}),'new_inferences':sum(not r['cached'] for r in rows),
              'seconds':time.monotonic()-started,'model_parameters':sum(p.numel() for p in predictor.model.parameters()) if predictor else previous_parameters,
              'aggregation':'equal utterance mean; sample standard deviation ddof=1; no score clipping',
              'aggregates':aggregates,'records':rows}
    write_json(destination/'manifest.json',manifest)
    enrich_report(args.run_dir/'index.html',aggregates,base_aggregates)
    for row in aggregates:print(f'{row["condition"]}: UTMOS={row["utmos_mean"]:.4f} n={row["utmos_utterances"]}',flush=True)
    print(f'Report: {args.run_dir}/index.html',flush=True)


if __name__=='__main__':main()
