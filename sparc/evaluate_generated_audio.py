"""Whisper large-v3 CER, WORLD/SPTK MCD and F0, and STOI for generated JSUT audio."""

import argparse
import csv
import html
import importlib.metadata
import json
import os
from pathlib import Path
import unicodedata

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio

from .compare_jsut_audio import audio_player
from .japanese_ppg import sha256_file
from .model_jsut_tokens import write_json


ASR_MODEL='openai/whisper-large-v3'
ASR_REVISION='06f233fe06e710322aca913c1bc4249a0d71fce1'
ASR_LOCAL=Path('.cache/whisper-large-v3')/ASR_REVISION
LABELS={'original':'元音声','original_ema':'元軌道合成','before':'編集前・一括合成','offline':'一括局所探索・参考'}


def normalize_cer(text):
    text=unicodedata.normalize('NFKC',text).lower()
    return ''.join(c for c in text if not c.isspace() and unicodedata.category(c)[0] not in ('P','Z','C'))


def character_errors(reference,hypothesis):
    reference,hypothesis=normalize_cer(reference),normalize_cer(hypothesis)
    if not reference: raise ValueError('Empty normalized reference')
    # Each cell contains (distance, substitutions, deletions, insertions).
    previous=[(j,0,0,j) for j in range(len(hypothesis)+1)]
    for i,ref in enumerate(reference,1):
        current=[(i,0,i,0)]
        for j,hyp in enumerate(hypothesis,1):
            if ref==hyp: current.append(previous[j-1]); continue
            d,s,de,ins=previous[j-1]; candidates=[(d+1,s+1,de,ins)]
            d,s,de,ins=previous[j]; candidates.append((d+1,s,de+1,ins))
            d,s,de,ins=current[j-1]; candidates.append((d+1,s,de,ins+1))
            current.append(min(candidates,key=lambda x:x[0]))
        previous=current
    errors,s,d,i=previous[-1]
    return {'cer':errors/len(reference),'character_errors':errors,'reference_characters':len(reference),
            'substitutions':s,'deletions':d,'insertions':i,
            'normalized_reference':reference,'normalized_hypothesis':hypothesis}


def read_audio(path):
    audio,sr=sf.read(path,dtype='float32',always_2d=True); audio=audio.mean(1)
    if sr!=16000:
        audio=torchaudio.functional.resample(torch.from_numpy(audio),sr,16000).numpy()
    if not len(audio) or not np.isfinite(audio).all(): raise ValueError('Invalid audio')
    return audio


def world_features(audio):
    import pyworld
    import pysptk
    audio=np.ascontiguousarray(audio,dtype=np.float64)
    f0,times=pyworld.dio(audio,16000,f0_floor=50,f0_ceil=550,frame_period=10)
    f0=pyworld.stonemask(audio,f0,times,16000)
    spectrum=pyworld.cheaptrick(audio,f0,times,16000,fft_size=1024)
    cepstrum=pysptk.sp2mc(np.maximum(spectrum,1e-12),order=24,alpha=.42)
    rms=librosa.feature.rms(y=audio,frame_length=400,hop_length=160,center=True,pad_mode='constant')[0]
    return {'f0':f0,'mcep':cepstrum,'rms':rms}


def acoustic_scores(reference,hypothesis,reference_features=None):
    from pystoi import stoi
    ref=world_features(reference) if reference_features is None else reference_features
    hyp=world_features(hypothesis)
    frames=min(len(ref['f0']),len(hyp['f0']),len(ref['rms']))
    mask=ref['rms'][:frames]>max(float(ref['rms'].max())*.01,1e-8)  # -40 dB relative RMS
    difference=ref['mcep'][:frames,1:25]-hyp['mcep'][:frames,1:25]
    mcd=(10/np.log(10))*np.sqrt(2*np.sum(difference**2,axis=1))
    if not np.isfinite(mcd).all(): raise ValueError('Nonfinite mel-cepstral distortion')
    ref_f0,hyp_f0=ref['f0'][:frames],hyp['f0'][:frames]
    both=(ref_f0>0)&(hyp_f0>0)
    hz=ref_f0[both]-hyp_f0[both]
    cents=1200*np.log2(hyp_f0[both]/ref_f0[both])
    length=min(len(reference),len(hypothesis))
    return {'mcd_db':float(mcd[mask].mean()) if mask.any() else None,
            'mcd_all_frames_db':float(mcd.mean()),'mcd_speech_frames':int(mask.sum()),
            'stoi':float(stoi(reference[:length],hypothesis[:length],16000,extended=False)),
            'f0_rmse_hz':float(np.sqrt(np.mean(hz**2))) if both.any() else None,
            'f0_rmse_cents':float(np.sqrt(np.mean(cents**2))) if both.any() else None,
            'both_voiced_frames':int(both.sum()),
            'vuv_error_rate':float(np.mean((ref_f0>0)!=(hyp_f0>0))),
            'acoustic_frames':frames,'common_audio_samples':length}


class WhisperRecognizer:
    def __init__(self,device='cuda'):
        from transformers import WhisperForConditionalGeneration,WhisperProcessor
        self.device=torch.device(device)
        self.dtype=torch.float16 if self.device.type=='cuda' else torch.float32
        self.processor=WhisperProcessor.from_pretrained(ASR_LOCAL,local_files_only=True)
        self.model=WhisperForConditionalGeneration.from_pretrained(ASR_LOCAL,local_files_only=True,
                         torch_dtype=self.dtype,use_safetensors=True,attn_implementation='sdpa').to(self.device).eval()
        self.model.generation_config.forced_decoder_ids=None

    def transcribe(self,audio):
        if len(audio)>16000*30: raise ValueError('This evaluation supports utterances up to 30 seconds')
        inputs=self.processor(audio,sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
        with torch.inference_mode():
            tokens=self.model.generate(inputs.input_features.to(self.device,dtype=self.dtype),
                        attention_mask=inputs.attention_mask.to(self.device),language='ja',task='transcribe',
                        num_beams=1,do_sample=False,max_new_tokens=256)
        return self.processor.batch_decode(tokens,skip_special_tokens=True)[0].strip()


def aggregate(rows):
    output=[]
    for condition in dict.fromkeys(r['condition'] for r in rows):
        group=[r for r in rows if r['condition']==condition]
        def average(key,weight=None):
            valid=[r for r in group if r[key] is not None]
            return float(np.average([r[key] for r in valid],weights=[r[weight] for r in valid] if weight else None)) if valid else None
        def rmse(key):
            valid=[r for r in group if r[key] is not None and r['both_voiced_frames']]
            return float(np.sqrt(np.average([r[key]**2 for r in valid],weights=[r['both_voiced_frames'] for r in valid]))) if valid else None
        errors=sum(r['character_errors'] for r in group); chars=sum(r['reference_characters'] for r in group)
        output.append({'condition':condition,'utterances':len(group),'cer':errors/chars,
                       'character_errors':errors,'reference_characters':chars,
                       'mcd_db':average('mcd_db','mcd_speech_frames'),'stoi':average('stoi'),
                       'f0_rmse_hz':rmse('f0_rmse_hz'),'f0_rmse_cents':rmse('f0_rmse_cents'),
                       'vuv_error_rate':average('vuv_error_rate','acoustic_frames')})
    return output


def save_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def write_report(output,rows,aggregates):
    def number(x,digits=3): return '—' if x is None else f'{x:.{digits}f}'
    table=''.join(f'<tr><td>{html.escape(LABELS.get(r["condition"],r["condition"]))}</td>'
                  f'<td>{100*r["cer"]:.1f}%</td><td>{number(r["mcd_db"])}</td><td>{number(r["stoi"])}</td>'
                  f'<td>{number(r["f0_rmse_hz"])}</td><td>{100*r["vuv_error_rate"]:.1f}%</td></tr>' for r in aggregates)
    cards=[]
    for row in rows:
        cards.append(f'<section><h3>{html.escape(row["utterance_id"])} / {html.escape(LABELS.get(row["condition"],row["condition"]))}</h3>'
                     f'{audio_player(row["audio_relative"])}<p>正解：{html.escape(row["reference_text"])}</p>'
                     f'<p>Whisper：{html.escape(row["asr_text"])}</p><p>CER {100*row["cer"]:.1f}% '
                     f'（置換{row["substitutions"]}・削除{row["deletions"]}・挿入{row["insertions"]}）</p></section>')
    content='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>音声生成の客観評価</title>
<style>body{font-family:system-ui;max-width:1250px;margin:30px auto;padding:0 18px}td,th{padding:9px;border-bottom:1px solid #ddd}table{border-collapse:collapse}section{border:1px solid #ccc;margin:18px 0;padding:15px}audio{width:100%;max-width:500px}</style>
<h1>Whisper large-v3 CER・MCD・STOI・F0評価</h1><p><a href="../index.html">制御実験のレポート</a> / <a href="summary.csv">発話別CSV</a> / <a href="aggregates.csv">条件別CSV</a></p>
<p>CERはopenai/whisper-large-v3を日本語・文字起こし・greedy decodingで実行し、JSUT正解文と比較します。
正解・認識文ともNFKC・小文字化・句読点と空白等の除去を適用し、漢字・かなや数字表記の同一化はしません。元音声のCERはASR自体の誤りの目安です。
CERは文字誤り数を正解文字数で割るmicro平均。MCDはWORLDスペクトルからSPTKで求めた24次mel-cepstrum（c0除外・α=0.42）の標準距離で、
参照RMSが最大値の−40 dBを上回るフレームで平均します。時間伸縮・DTWは使わず、共通時刻だけを比較します。
F0はWORLD DIO＋StoneMask、50〜550 Hz・10 ms刻み。RMSEは両音声が有声のフレーム、V/UV誤り率は全共通フレームで計算します。
STOIは16 kHz・共通長・非extended版です。音量の個別正規化はしません。これらは主観的な自然さやMOSの代わりではありません。</p>
<table><tr><th>条件</th><th>CER ↓</th><th>MCD(dB) ↓</th><th>STOI ↑</th><th>F0 RMSE(Hz) ↓</th><th>V/UV誤り ↓</th></tr>'''
    content+=table+'</table>'+''.join(cards)+'</html>'
    (output/'index.html').write_text(content,encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir',type=Path,default=Path('outputs/jsut_ppg_delay060_step020'))
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--device',default='cuda');parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args(argv)
    output=args.output_dir or args.experiment_dir/'audio_metrics'
    if output.exists() and (not output.is_dir() or any(output.iterdir())): parser.error('Choose a new output directory')
    torch.set_num_threads(args.threads)
    experiment=json.loads((args.experiment_dir/'manifest.json').read_text())
    if len(experiment['records'])!=len(experiment['arguments']['horizons'])*3*len(experiment['arguments']['utterances']):
        parser.error('Control experiment is not complete')
    offline=json.loads((Path(experiment['arguments']['offline_dir'])/'manifest.json').read_text())
    ppg_dir=Path(experiment['arguments']['ppg_dir']); ppg_manifest=json.loads((ppg_dir/'manifest.json').read_text())
    items=[]
    for ref in offline['records']:
        if ref['utterance_id'] not in experiment['arguments']['utterances']: continue
        u=ref['utterance_id']; source=next(r for r in ppg_manifest['records'] if r['utterance_id']==u)
        for name in ['original','original_ema','before','offline']:
            path=(ppg_dir/source['variants']['original_ema']['audio'] if name=='original_ema'
                  else args.experiment_dir/'references'/u/f'{name}.wav')
            items.append({'condition':name,'utterance_id':u,'path':path,'text':ref['text']})
    for record in experiment['records']:
        ref=next(r for r in offline['records'] if r['utterance_id']==record['utterance_id'])
        items.append({'condition':record['condition'],'utterance_id':record['utterance_id'],
                      'path':args.experiment_dir/record['directory']/'output.wav','text':ref['text']})
    recognizer=WhisperRecognizer(args.device)
    output.mkdir(parents=True,exist_ok=True)
    sources={u:read_audio(args.experiment_dir/'references'/u/'original.wav') for u in experiment['arguments']['utterances']}
    features={u:world_features(audio) for u,audio in sources.items()}
    rows,asr_cache=[],{}
    provenance={'asr_model_id':ASR_MODEL,'asr_revision':ASR_REVISION,'language':'ja','task':'transcribe',
                'num_beams':1,'do_sample':False,'max_new_tokens':256,'prompt':None,
                'normalization':'NFKC, lower, remove Unicode P/Z/C and whitespace; no kanji/kana or digit folding',
                'mcd':'WORLD CheapTrick power spectrum -> SPTK sp2mc order24 alpha.42, c0 excluded; same times; reference RMS > max*.01',
                'f0':'WORLD dio + stonemask, 16kHz,10ms,50-550Hz, both voiced for RMSE',
                'alignment':'common original timestamps, no DTW or time shift','individual_gain_normalization':False,
                'experiment_manifest_sha256':sha256_file(args.experiment_dir/'manifest.json'),
                'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','pyworld','pysptk','pystoi','numpy','setuptools']},
                'records':[]}
    for index,item in enumerate(items):
        audio=read_audio(item['path']); digest=sha256_file(item['path'])
        text=asr_cache.get(digest)
        if text is None:
            text=recognizer.transcribe(audio);asr_cache[digest]=text
        row={'condition':item['condition'],'utterance_id':item['utterance_id'],'audio_sha256':digest,
             'audio_relative':os.path.relpath(item['path'].resolve(),output.resolve()),
             'reference_text':item['text'],'asr_text':text,**character_errors(item['text'],text),
             **acoustic_scores(sources[item['utterance_id']],audio,features[item['utterance_id']])}
        rows.append(row);aggregates=aggregate(rows)
        save_csv(output/'summary.csv',rows);save_csv(output/'aggregates.csv',aggregates)
        provenance['records']=rows;write_json(output/'manifest.json',provenance)
        write_report(output,rows,aggregates)
        print(f'{index+1}/{len(items)} {item["condition"]} {item["utterance_id"]}: CER={row["cer"]:.3f} ASR={text}',flush=True)
    write_json(output/'asr_cache.json',asr_cache)
    index=args.experiment_dir/'index.html'
    page=index.read_text(encoding='utf-8')
    link=os.path.relpath((output/'index.html').resolve(),args.experiment_dir.resolve())
    page=page.replace('</h1>',f'</h1><p><a href="{html.escape(link,quote=True)}">Whisper large-v3 CER・MCD・STOI・F0評価</a></p>',1)
    index.write_text(page,encoding='utf-8')
    print(f'Report: {output}/index.html',flush=True)
    return 0


if __name__=='__main__': main()
