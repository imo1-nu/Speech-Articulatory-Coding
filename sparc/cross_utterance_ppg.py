"""Start from A articulation, steer toward native-time B PPG, and evaluate A/B content separately."""

import argparse
import csv
import html
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import soundfile as sf
import torch

from . import delayed_ppg_search as delayed
from . import evaluate_generated_audio as metrics
from . import local_ppg_search as local
from . import sequential_ppg_search as seq
from .compare_jsut_audio import audio_player
from .japanese_ppg import JapanesePPGExtractor, frame_times, load_audio, sha256_file
from .model_jsut_tokens import configure_plots, infer_utterance, save_npz, write_json


DEFAULT_PAIRS=['BASIC5000_0053:BASIC5000_0050','BASIC5000_0050:BASIC5000_0066','BASIC5000_0066:BASIC5000_0053']
REFERENCE_LABELS={'target_original':'B原音（目標）','source_initial':'A軌道→B長さ→トークン→合成',
                  'target_tokens_oracle':'Bトークン＋AのF0・音量・話者', 'target_ema_oracle':'B元軌道＋AのF0・音量・話者'}


def parse_pairs(values):
    pairs=[]
    for value in values:
        ids=value.split(':')
        if len(ids)!=2 or not all(ids) or ids[0]==ids[1] or tuple(ids) in pairs:
            raise ValueError('Pairs must be unique A:B IDs with A != B')
        pairs.append(tuple(ids))
    return pairs


def retime_source(values,frames):
    """Interpolate continuous source features on normalized time, never interpolate token IDs."""
    values=np.asarray(values,dtype=np.float32)
    if values.ndim!=2 or len(values)<2 or frames<2 or not np.isfinite(values).all():
        raise ValueError('Expected finite continuous features with at least two frames')
    if len(values)==frames:
        return values.copy()
    source=np.linspace(0,1,len(values)); destination=np.linspace(0,1,frames)
    return np.stack([np.interp(destination,source,values[:,i]) for i in range(values.shape[1])],axis=1).astype(np.float32)


def pair_inputs(source_fixed,target_fixed,target_ppg,model,mean,std):
    """Only B duration and B PPG affect search inputs; B articulation is reserved for oracle controls."""
    frames=len(target_fixed['ema'])
    ema=retime_source(source_fixed['ema'],frames)
    fixed={key:retime_source(source_fixed[key],frames) for key in ['pitch','loudness']}
    fixed['spk_emb']=source_fixed['spk_emb'].copy()
    tokens,reconstruction,_=infer_utterance(model,ema,mean,std,512,torch.device('cpu'))
    if len(target_ppg)<frames*2: raise ValueError('B PPG is shorter than B articulatory duration')
    return {'source_ema_retimed':ema,'fixed':fixed,'initial_tokens':tokens,
            'initial_reconstruction':reconstruction,'target_ppg':target_ppg[:frames*2].copy()}


def synthesize_ema(generator,ema,fixed,gain,device):
    art=np.concatenate([ema,fixed['pitch'],fixed['loudness']],axis=1)
    with torch.inference_mode():
        output=generator(torch.from_numpy(art)[None].to(device).transpose(1,2),
                         torch.from_numpy(fixed['spk_emb'])[None].to(device))[0,0]
        output=torch.floor(output*gain*32768).clamp(-32768,32767)/32768
    return output.cpu().numpy()


def aggregate_control(rows):
    output=[]
    for condition in dict.fromkeys(r['condition'] for r in rows):
        group=[r for r in rows if r['condition']==condition]
        row={'condition':condition,'pairs':len(group),'evaluations':sum(r['evaluations'] for r in group),
             'edited_tokens':sum(r['edited_tokens'] for r in group),'token_count':sum(r['token_count'] for r in group)}
        for key,weight in [('ppg_js_to_B','ppg_frames'),('ppg_js_to_source_initial','ppg_frames'),('transition_nll','edges')]:
            selected=[r for r in group if r[key] is not None]
            row[key]=float(np.average([r[key] for r in selected],weights=[r[weight] for r in selected])) if selected else None
        output.append(row)
    return output


def condition_label(condition):
    if condition in REFERENCE_LABELS: return REFERENCE_LABELS[condition]
    horizon=int(condition.split('_')[1].removesuffix('ms'))
    mode=condition.split('ms_',1)[1]
    return f'{horizon} ms・{seq.LABELS[mode]}'


def write_report(output,pairs,rows):
    aggregates=aggregate_control(rows)
    seq.save_csv(output/'summary.csv',rows); seq.save_csv(output/'aggregates.csv',aggregates)
    table=''.join(f'<tr><td>{html.escape(condition_label(r["condition"]))}</td><td>{r["ppg_js_to_B"]:.5f}</td>'
                  f'<td>{r["ppg_js_to_source_initial"]:.5f}</td><td>{r["edited_tokens"]}/{r["token_count"]}</td>'
                  f'<td>{r["evaluations"]}</td></tr>' for r in aggregates)
    sections=[]
    for pair in pairs:
        folder=f'pairs/{pair["pair_id"]}'
        players=f'<div><h3>A原音</h3>{audio_player(folder+"/source_original.wav")}</div>'
        for row in rows:
            if row['pair_id']!=pair['pair_id']: continue
            players+=(f'<div><h3>{html.escape(condition_label(row["condition"]))}</h3>{audio_player(row["audio"])}'
                      f'<p>対B JS {row["ppg_js_to_B"]:.5f} / 対A初期合成JS {row["ppg_js_to_source_initial"]:.5f}</p>'
                      f'<a href="{row["directory"]}/ppg.png">PPG比較図</a></div>')
        sections.append(f'<section><h2>{html.escape(pair["source_id"])} → {html.escape(pair["target_id"])}</h2>'
                        f'<p>A：{html.escape(pair["source_text"])}<br>B：{html.escape(pair["target_text"])}</p>'
                        f'<p>Aの{pair["source_frames"]}フレームをBの{pair["target_frames"]}フレームへ線形補間。BのPPGは補間しません。</p>'
                        f'<div class="players">{players}</div></section>')
    content='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A軌道からBのPPGへの制御</title>
<style>body{font-family:system-ui;max-width:1350px;margin:30px auto;padding:0 18px}.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px}section{border:1px solid #ccc;padding:18px;margin:24px 0}audio{width:100%}td,th{padding:9px;border-bottom:1px solid #ddd}h3{font-size:1rem}table{border-collapse:collapse}</style>
<h1>Aの軌道・合成条件から、Bの発話内容へ近づける実験</h1><p><a href="audio_metrics/index.html">Whisper large-v3 CER・音響指標</a> / <a href="summary.csv">発話ペア別CSV</a> / <a href="aggregates.csv">集計CSV</a></p>
<p>Aの連続調音軌道・F0・音量をBのフレーム数に線形補間してから、40 ms・128コードのVQでトークン化します。
話者埋め込みもA由来で固定し、Bから使うのは発話長と元音声PPGです。BのPPGや原音は時間伸縮しません。末尾はBの調音フレーム長に合わせた共通区間で評価します。
したがって、目標はBの発話内容への変換であり、Bの声質や抑揚まで同じ波形を得る実験ではありません。</p>
<p>更新・確定20 ms、観測音声は現在の60 ms前まで、λ=0.03。現在位置と差の大きい未来1位置を各16候補・1パスで探索します。
200・400・800 msの各探索範囲で、無編集・予測補正なし・予測補正ありを比較します。
補正なしもBのPPGへ向けて探索するため、FB固有の効果は同じ探索範囲の補正有無で比較します。
補正ありのFBは予測と遅延観測の差による音素別log確率バイアスです。過去に出力した音声・トークンは変更しません。</p>
<p>B元軌道・BトークンをAの合成条件で鳴らす比較音声も用意します。これらは到達可能性を調べる参考で、探索の初期値・候補提案には使いません。
PPG差の改善が認識性能の改善を意味するとは限らないため、B正解文とA正解文の両方に対するCERも評価します。
目標PPGと元特徴は事前抽出済みです。</p>
<table><tr><th>条件</th><th>対B PPG JS ↓</th><th>対A初期合成 PPG JS</th><th>編集数</th><th>評価回数</th></tr>'''
    content+=table+'</table>'+''.join(sections)+"<script>document.querySelectorAll('audio').forEach(p=>p.onplay=()=>document.querySelectorAll('audio').forEach(q=>{if(q!==p)q.pause()}));</script></html>"
    (output/'index.html').write_text(content,encoding='utf-8')
    return aggregates


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=Path('outputs/jsut_window_sweep/window_040ms'))
    parser.add_argument('--ppg-dir',type=Path,default=Path('outputs/jsut_ppg_40ms128'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/jsut_ppg_A_to_B'))
    parser.add_argument('--pairs',nargs='+',default=DEFAULT_PAIRS)
    parser.add_argument('--horizons',nargs='+',type=int,default=[200,400,800])
    parser.add_argument('--device',default='cuda');parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args(argv)
    try: pairs=parse_pairs(args.pairs)
    except ValueError as exc: parser.error(str(exc))
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())): parser.error('Choose a new output directory')
    if len(set(args.horizons))!=len(args.horizons) or any(h<20 or h%20 for h in args.horizons) or args.threads<1: parser.error('Invalid horizons or threads')
    args.utterances=list(dict.fromkeys(u for pair in pairs for u in pair));args.transition_alpha=1
    torch.set_num_threads(args.threads)
    source,digest,normalized,_,counts,selected,model,mean,std,markov=local.prepare(args)
    items={record['utterance_id']:(record,fixed,tokens,ppg,gain) for record,fixed,tokens,ppg,gain in selected}
    generator,generator_digest=local.load_generator(args.device);extractor=JapanesePPGExtractor(args.device)
    if extractor.checkpoint_sha256!=source['ppg_model']['checkpoint_sha256']: raise ValueError('PPG model mismatch')
    configure_plots();args.output_dir.mkdir(parents=True,exist_ok=True)
    save_npz(args.output_dir/'transitions.npz',**markov,alpha=np.float64(1),vq_checkpoint_sha256=np.str_(digest))
    metadata=[]; prepared={}; rows=[]
    manifest={'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'vq_checkpoint_sha256':digest,'generator_checkpoint_sha256':generator_digest,
              'ppg_checkpoint_sha256':extractor.checkpoint_sha256,
              'settings':{'commit_ms':20,'observation_delay_ms':60,'transition_weight':.03,
                          'retiming':'source continuous EMA, pitch and loudness -> target frame count; no target PPG warp',
                          'fixed_from_source':['pitch','loudness','speaker_embedding'],
                          'oracle_target_tokens_used_for_search':False,'oracle_target_ema_used_for_search':False,
                          'feedback_definition':'predicted vs observed phone-mass calibration; target B guides both search modes'},
              'pairs':metadata,'records':rows}
    def publish():
        manifest['aggregates']=write_report(args.output_dir,metadata,rows)
        write_json(args.output_dir/'manifest.json',manifest)
    for source_id,target_id in pairs:
        a,source_fixed,_,_,gain=items[source_id];b,target_fixed,target_tokens,target_ppg,_=items[target_id]
        pair_id=f'{source_id}__to__{target_id}';folder=args.output_dir/'pairs'/pair_id;folder.mkdir(parents=True)
        data=pair_inputs(source_fixed,target_fixed,target_ppg,model,mean,std)
        pair={'pair_id':pair_id,'source_id':source_id,'target_id':target_id,
              'source_text':a['text'],'target_text':b['text'],'source_frames':len(source_fixed['ema']),
              'target_frames':len(target_fixed['ema']),'gain':gain,
              'source_audio_sha256':a['variants']['original']['audio_sha256'],
              'target_audio_sha256':b['variants']['original']['audio_sha256']}
        metadata.append(pair);prepared[pair_id]=(data,gain)
        shutil.copyfile(args.ppg_dir/a['variants']['original']['audio'],folder/'source_original.wav')
        target_dir=folder/'target_original';target_dir.mkdir()
        shutil.copyfile(args.ppg_dir/b['variants']['original']['audio'],target_dir/'output.wav')
        evaluator=seq.PrefixEvaluator(generator,extractor,normalized,data['fixed'],gain,args.device,model,mean,std)
        baseline,baseline_ppg=evaluator.at(len(data['initial_tokens']),np.empty(0,dtype=np.float32),np.empty(0,dtype=np.float32),160)(data['initial_tokens'][None])
        data['baseline_ppg']=baseline_ppg[0]
        oracle_tokens_audio,oracle_tokens_ppg=evaluator.at(len(target_tokens),np.empty(0,dtype=np.float32),np.empty(0,dtype=np.float32),160)(target_tokens[None])
        oracle_ema_audio=synthesize_ema(generator,target_fixed['ema'],data['fixed'],gain,args.device)
        refs=[('target_original',None,data['target_ppg'],None),
              ('source_initial',baseline[0],baseline_ppg[0],data['initial_tokens']),
              ('target_tokens_oracle',oracle_tokens_audio[0],oracle_tokens_ppg[0],target_tokens),
              ('target_ema_oracle',oracle_ema_audio,extractor.extract(torch.from_numpy(oracle_ema_audio[None,None])),None)]
        save_npz(folder/'inputs.npz',source_ema_retimed=data['source_ema_retimed'],initial_tokens=data['initial_tokens'],
                 target_ppg=data['target_ppg'],baseline_ppg=data['baseline_ppg'],pitch=data['fixed']['pitch'],
                 loudness=data['fixed']['loudness'],spk_emb=data['fixed']['spk_emb'],
                 oracle_target_tokens=target_tokens,oracle_target_ema=target_fixed['ema'])
        for condition,audio,ppg,tokens in refs:
            directory=folder/condition;directory.mkdir(exist_ok=True)
            if audio is not None: sf.write(directory/'output.wav',audio,16000,subtype='PCM_16')
            distance=local.posterior_score(data['target_ppg'],ppg)
            row={'condition':condition,'pair_id':pair_id,'source_id':source_id,'target_id':target_id,
                 'directory':directory.relative_to(args.output_dir).as_posix(),
                 'audio':(directory/'output.wav').relative_to(args.output_dir).as_posix(),
                 'ppg_js_to_B':float(distance.mean()),
                 'ppg_js_to_source_initial':float(local.posterior_score(data['baseline_ppg'],ppg).mean()),
                 'ppg_frames':len(distance),'token_count':len(data['initial_tokens']),'edges':len(data['initial_tokens'])-1,
                 'transition_nll':local.transition_nll(tokens,markov['transition_probs']) if tokens is not None else None,
                 'edited_tokens':int(np.count_nonzero(tokens!=data['initial_tokens'])) if tokens is not None else 0,
                 'evaluations':0,'seconds':0.}
            save_npz(directory/'result.npz',final_ppg=ppg,target_ppg=data['target_ppg'],js=distance)
            seq.plot_utterance(directory/'ppg.png',data['target_ppg'],data['baseline_ppg'],ppg,ppg,[])
            rows.append(row);publish()
    for horizon in args.horizons:
        for mode in seq.MODES:
            condition=f'horizon_{horizon:03d}ms_{mode}'
            for pair in metadata:
                pair_id=pair['pair_id'];data,gain=prepared[pair_id]
                directory=args.output_dir/'pairs'/pair_id/condition;directory.mkdir()
                evaluator=seq.PrefixEvaluator(generator,extractor,normalized,data['fixed'],gain,args.device,model,mean,std)
                tick=time.monotonic()
                with (directory/'trace.jsonl').open('w') as log:
                    def on_step(item):
                        log.write(json.dumps(item)+'\n');log.flush()
                        if item['committed_until_token']%20==0:
                            print(f'{pair_id} {condition} t={item["decision_time_ms"]}ms JS(plan)={item["final_planning_js"]:.5f}',flush=True)
                    result=delayed.delayed_search(data['initial_tokens'],data['target_ppg'],normalized,np.flatnonzero(counts),
                                                  markov['transition_probs'],evaluator,horizon_tokens=horizon//20,mode=mode,on_step=on_step)
                sf.write(directory/'output.wav',result['audio'],16000,subtype='PCM_16')
                saved,sr=sf.read(directory/'output.wav',dtype='float32')
                if sr!=16000 or not np.array_equal(saved,result['audio']):raise ValueError('Saved waveform changed')
                ppg=extractor.extract(load_audio(directory/'output.wav')[0])
                distance=local.posterior_score(data['target_ppg'],ppg)
                row={'condition':condition,'pair_id':pair_id,'source_id':pair['source_id'],'target_id':pair['target_id'],
                     'directory':directory.relative_to(args.output_dir).as_posix(),
                     'audio':(directory/'output.wav').relative_to(args.output_dir).as_posix(),
                     'ppg_js_to_B':float(distance.mean()),
                     'ppg_js_to_source_initial':float(local.posterior_score(data['baseline_ppg'],ppg).mean()),
                     'ppg_frames':len(distance),'token_count':len(result['tokens']),'edges':len(result['tokens'])-1,
                     'transition_nll':local.transition_nll(result['tokens'],markov['transition_probs']),
                     'edited_tokens':int(np.count_nonzero(data['initial_tokens']!=result['tokens'])),
                     'evaluations':sum(x['evaluations'] for x in result['trace']),'seconds':time.monotonic()-tick}
                save_npz(directory/'result.npz',initial_tokens=data['initial_tokens'],edited_tokens=result['tokens'],
                         final_ppg=ppg,target_ppg=data['target_ppg'],baseline_ppg=data['baseline_ppg'],
                         prediction_ppg=result['prediction_ppg'],observed_ppg=result['observed_ppg'],
                         feedback_biases=result['biases'],commanded_audio=result['commanded_audio'],
                         pitch=data['fixed']['pitch'],loudness=data['fixed']['loudness'],spk_emb=data['fixed']['spk_emb'],
                         time=frame_times(len(ppg)),js=distance)
                write_json(directory/'terminal_observation.json',result['terminal_observation'])
                seq.plot_utterance(directory/'ppg.png',data['target_ppg'],data['baseline_ppg'],ppg,result['observed_ppg'],result['trace'])
                rows.append(row);publish()
                print(f'{pair_id} {condition}: JS(B)={row["ppg_js_to_B"]:.6f}',flush=True)
    print(f'Report: {args.output_dir}/index.html',flush=True)
    return 0


if __name__=='__main__': main()
