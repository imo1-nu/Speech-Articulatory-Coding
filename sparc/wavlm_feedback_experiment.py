"""Reproducible PPG versus WavLM offline and delayed RVQ feedback experiment."""

import argparse
import csv
import html
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import soundfile as sf
import torch

from . import evaluate_generated_audio as metrics
from .compare_jsut_audio import audio_player
from .compare_trajectory_reconstruction import trajectory_statistics, summarize_trajectories
from .feedback_features import FeatureObjective
from .feature_feedback_search import AcousticEvaluator, optimize, delayed
from .japanese_ppg import sha256_file
from .local_ppg_search import load_generator, transition_statistics
from .model_jsut_tokens import checked_feature, load_run, save_npz, write_json
from .rvq_transition_search import dynamics
from .sequential_ppg_search import save_csv
from .trajectory_rvq import load_rvq
from .trajectory_vq import overlap_average


def save_feature_metrics(path,records):
    # Nested audit metadata stays in JSON; keep the CSV schema identical across modes.
    excluded={'terminal_observation','execution_sources'}
    save_csv(path,[{k:v for k,v in row.items() if k not in excluded} for row in records])


def write_report(output,manifest,rows):
    groups=metrics.aggregate(rows) if rows else []
    if groups:save_csv(output/'audio_aggregates.csv',groups)
    records=manifest['records']
    table=''.join(f'<tr><td>{r["condition"]}</td><td>{r["utterances"]}</td><td>{100*r["cer"]:.2f}</td><td>{r["mcd_db"]:.3f}</td><td>{r["stoi"]:.4f}</td></tr>' for r in groups)
    feature_table=''
    for condition in dict.fromkeys(r['condition'] for r in records):
        group=[r for r in records if r['condition']==condition]
        mean=lambda key:float(np.mean([r[key] for r in group]))
        feature_table+=f'<tr><td>{condition}</td><td>{len(group)}</td><td>{mean("wavlm_distance"):.5f}</td><td>{mean("ppg_js"):.5f}</td><td>{mean("first_stage_nll"):.4f}</td><td>{mean("seconds"):.1f}</td></tr>'
    cards=[]
    for source in manifest['inputs']:
        folder=source['directory']
        variants=[(f'{folder}/{v}.wav',v) for v in ('original','original_ema','before')]
        variants += [(r['audio'],r['condition']) for r in records if r['utterance_id']==source['utterance_id']]
        players=''.join(f'<div><h3>{html.escape(label)}</h3>{audio_player(path)}</div>' for path,label in variants)
        transcripts=''.join(f'<li>{r["condition"]}: {html.escape(r["asr_text"])}</li>' for r in rows if r['utterance_id']==source['utterance_id'])
        cards.append(f'<section><h2>{source["utterance_id"]}</h2><p>{html.escape(source["text"])}</p><div class="players">{players}</div><ul>{transcripts}</ul></section>')
    page='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WavLM特徴量FBとPPGの比較</title>
<style>body{font-family:system-ui;max-width:1500px;margin:25px auto;padding:15px}.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px}audio{width:100%}td,th{padding:8px;border-bottom:1px solid #ccc}section{border:1px solid #ccc;margin:20px 0;padding:16px}</style>
<h1>RVQ8：WavLM特徴量目標とPPG目標の比較</h1>
<p>WavLM-large hidden_states[9]・1024次元、波形標準化と左右160サンプルのゼロ埋め、平滑化なし。主距離はフレームごとのcosine距離。
PPGは35音素のJS。両者の数値尺度は異なります。目標は同じ発話の元音声、F0・音量・話者・ゲイン固定。</p>
<p>オフラインは最大2パス×8段×24時刻、遅延制御は20ms更新・観測波形t−60msまで・各段2時刻を1パス。各座標12近傍＋4広域候補、バッチ4。
全8段を編集し、遷移NLLは段1だけ。遅延FBは観測と発行時予測の特徴量差を補正する方式。目標と初期軌道は事前に利用可能です。
非因果モデルを利用可能な接頭辞にのみ適用するため、発話全体で抽出する特徴量と同一ではありません。</p>
<p>MCD/STOI/F0の参照は元軌道合成、CERはWhisper large-v3とJSUT正解文。発話別数値・特徴量時刻・トークン・補正バイアス・探索履歴を保存します。</p>
<p><a href="manifest.json">設定と記録</a> / <a href="audio_aggregates.csv">音声集計CSV</a> / <a href="feature_metrics.csv">特徴量CSV</a></p>
<h2>特徴量・計算時間（発話平均）</h2><table><tr><th>条件</th><th>発話数</th><th>WavLM距離 ↓</th><th>PPG JS ↓</th><th>段1 NLL ↓</th><th>秒</th></tr>'''
    page+=feature_table+'</table><h2>音声指標</h2><table><tr><th>条件</th><th>発話数</th><th>CER % ↓</th><th>MCD dB ↓</th><th>STOI ↑</th></tr>'+table+'</table>'+''.join(cards)
    page+="<script>document.querySelectorAll('audio').forEach(p=>p.onplay=()=>document.querySelectorAll('audio').forEach(q=>{if(q!==p)q.pause()}));</script></html>"
    (output/'index.html').write_text(page,encoding='utf-8')
    manifest['audio_aggregates']=groups


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--objective',choices=('ppg','wavlm','all'),default='all')
    parser.add_argument('--mode',choices=('offline','delayed','all'),default='all')
    parser.add_argument('--weights',type=float,nargs='+',default=[0,.03,.1])
    parser.add_argument('--horizon-ms',type=int,choices=(200,400,800),default=400)
    parser.add_argument('--utterances',nargs='+')
    parser.add_argument('--model-dir',type=Path,default=Path('outputs/jsut_rvq/model'))
    parser.add_argument('--baseline-dir',type=Path,default=Path('outputs/jsut_rvq/comparison_verified'))
    parser.add_argument('--prior-dir',type=Path,default=Path('outputs/jsut_rvq_first_stage_transition'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/jsut_wavlm_feedback'))
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not np.isfinite(args.weights).all() or min(args.weights)<0 or len(set(args.weights))!=len(args.weights):parser.error('Invalid weights')
    if args.utterances and len(set(args.utterances))!=len(args.utterances):parser.error('Duplicate utterances')
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:parser.error('Choose new output or --resume')
    torch.set_num_threads(4)
    sources={name:sha256_file(Path(__file__).with_name(name)) for name in ('wavlm_feedback_experiment.py','feature_feedback_search.py','feedback_features.py')}
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='resume'}
    baseline=json.loads((args.baseline_dir/'manifest.json').read_text())
    rvq=json.loads((args.model_dir/'config.json').read_text())
    config,splits,_,_,_=load_run(Path(rvq['reference_dir']),'cpu')
    model,mean,std=load_rvq(args.model_dir,8,args.device)
    prior=dict(np.load(args.prior_dir/'transitions.npz'))
    if (str(prior['codebooks_sha256'])!=baseline['codebooks_sha256']
            or sha256_file(args.model_dir/'codebooks.npz')!=baseline['codebooks_sha256']):raise ValueError('Prior/RVQ mismatch')
    dependencies={str(p):sha256_file(p) for p in [args.baseline_dir/'manifest.json',args.model_dir/'codebooks.npz',args.prior_dir/'transitions.npz']}
    if args.resume:
        previous=json.loads((args.output_dir/'manifest.json').read_text())
        if previous['sources']!=sources or previous['settings']!=settings or previous['dependencies']!=dependencies:
            raise ValueError('Cannot resume with changed source, settings, or dependencies')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    generator,generator_hash=load_generator(args.device)
    reference_manifest=json.loads(Path('outputs/jsut_reconstruction_ablation/comparison/manifest.json').read_text())
    if generator_hash!=reference_manifest['generator_sha256']:raise ValueError('Generator changed')
    if (baseline['asr_model']!=metrics.ASR_MODEL or baseline['asr_revision']!=metrics.ASR_REVISION
            or baseline['asr_settings']!=reference_manifest['asr_settings']):raise ValueError('ASR provenance mismatch')
    objectives={name:FeatureObjective(name,args.device) for name in ('ppg','wavlm')}
    if objectives['ppg'].metadata['checkpoint_sha256']!=reference_manifest['ppg_sha256']:raise ValueError('PPG changed')
    selected=[r for r in baseline['audio_records'] if args.utterances is None or r['utterance_id'] in args.utterances]
    if not selected or (args.utterances and len(selected)!=len(args.utterances)):raise ValueError('Unknown test utterance')
    manifest={'settings':settings,'sources':sources,'dependencies':dependencies,
              'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'generator_sha256':generator_hash,'features':{k:v.metadata for k,v in objectives.items()},
              'asr_model':metrics.ASR_MODEL,'asr_revision':metrics.ASR_REVISION,'asr_settings':reference_manifest['asr_settings'],
              'packages':{k:importlib.metadata.version(k) for k in ('torch','transformers','numpy')},
              'search_budget':{'offline_passes':2,'offline_positions_per_stage':24,'delayed_passes':1,
                               'delayed_positions_per_stage':2,'neighbors':12,'global_candidates':4,'batch_size':4,
                               'min_improvement':1e-5,'prior_stage':1,'delayed_weight':.03,
                               'delay_ms':60,'step_ms':20,'crossfade_ms':10,'feedback_gain':.5,
                               'feedback_rate_per_100ms':.25,'feedback_clip':1},'inputs':selected,'records':[],
              'baseline_features':[], 'source_history':previous.get('source_history',[]) if args.resume else []}
    write_json(args.output_dir/'manifest.json',manifest)
    source_audio=json.loads((Path(rvq['reference_dir'])/'audio_comparison/manifest.json').read_text())
    prepared=[]
    with (args.baseline_dir/'audio_metrics.csv').open() as stream:
        original_hashes={(r['utterance_id'],r['condition']):r['audio_sha256'] for r in csv.DictReader(stream)}
    for rec in selected:
        folder=rec['directory'];relative=folder+'.npz'
        if relative not in splits['test']:raise ValueError('Not a test utterance')
        destination=args.output_dir/folder;destination.mkdir(parents=True,exist_ok=True)
        for variant in ('original','original_ema','rvq8'):
            if sha256_file(args.baseline_dir/folder/f'{variant}.wav')!=original_hashes[(rec['utterance_id'],variant)]:
                raise ValueError('Baseline audio provenance mismatch')
            shutil.copyfile(args.baseline_dir/folder/f'{variant}.wav',destination/('before.wav' if variant=='rvq8' else variant+'.wav'))
        initial=np.load(args.baseline_dir/'tokens'/relative)['tokens']
        fixed=dict(np.load(Path(rvq['reference_dir'])/'audio_comparison'/folder/'conditions.npz'))
        gain=next(r['synthesis_common_gain'] for r in source_audio['records'] if r['utterance_id']==rec['utterance_id'])
        natural=metrics.read_audio(destination/'original.wav')
        targets={k:obj.extract(natural[None])[0] for k,obj in objectives.items()}
        save_npz(destination/'targets.npz',**{k:v for k,v in targets.items()},
                 **{k+'_times':obj.times(len(targets[k])) for k,obj in objectives.items()})
        before=metrics.read_audio(destination/'before.wav')
        before_values={k:obj.extract(before[None])[0] for k,obj in objectives.items()}
        baseline_distances={k:float(obj.distances(targets[k][:min(len(targets[k]),len(before_values[k]))],
                                                 before_values[k][:min(len(targets[k]),len(before_values[k]))]).mean())
                            for k,obj in objectives.items()}
        manifest['baseline_features'].append({'utterance_id':rec['utterance_id'],
                                              'wavlm_distance':baseline_distances['wavlm'],'ppg_js':baseline_distances['ppg']})
        prepared.append((rec,initial,fixed,gain,targets,before))
    kinds=('ppg','wavlm') if args.objective=='all' else (args.objective,)
    modes=('offline','delayed') if args.mode=='all' else (args.mode,)
    for mode in modes:
        for kind in kinds:
            obj=objectives[kind]
            variants=[(f'offline_{kind}_lambda_{w:g}',w,None) for w in args.weights] if mode=='offline' else [
                (f'delayed_{kind}_{control}_h{args.horizon_ms}',.03,control) for control in ('unedited','no_feedback','feedback')]
            for rec,initial,fixed,gain,targets,before in prepared:
                evaluator=AcousticEvaluator(model,mean,std,generator,obj,fixed,gain,args.device)
                reproduced=evaluator.at(len(initial))(initial[None])[0][0]
                if reproduced.shape!=before.shape or np.max(abs(reproduced-before))>3/32768:
                    raise ValueError('Baseline synthesis mismatch')
                for condition,weight,control in variants:
                    directory=args.output_dir/rec['directory']/condition
                    summary_path=directory/'summary.json'
                    if args.resume and summary_path.exists():
                        summary=json.loads(summary_path.read_text())
                        if summary['audio_sha256']!=sha256_file(args.output_dir/summary['audio']):raise ValueError('Saved audio changed')
                        if summary['result_sha256']!=sha256_file(directory/'result.npz'):raise ValueError('Saved tokens changed')
                        if summary['trace_sha256']!=sha256_file(directory/'trace.jsonl'):raise ValueError('Saved trace changed')
                    else:
                        directory.mkdir(parents=True,exist_ok=True)
                        tick=time.monotonic();synth_start=evaluator.synthesis_seconds;candidate_start=evaluator.synthesis_candidates
                        feature_start=obj.seconds;feature_calls=obj.calls
                        with (directory/'trace.jsonl').open('w') as stream:
                            def step(item):
                                stream.write(json.dumps(item)+'\n');stream.flush()
                                if 'start_token' in item and item['start_token']%25==0:
                                    print(f'{condition} {rec["utterance_id"]} t={item["decision_time_ms"]}ms evaluations={item["evaluations"]}',flush=True)
                            if mode=='offline':
                                result=optimize(initial,targets[kind],evaluator.at(len(initial)),obj,evaluator.candidates,
                                                prior['transition_probs'],weight,on_step=step)
                            else:
                                result=delayed(initial,targets[kind],evaluator,prior['transition_probs'],
                                               horizon_tokens=args.horizon_ms//20,mode=control,weight=weight,on_step=step)
                        path=directory/'after.wav';sf.write(path,result['audio'],16000,subtype='PCM_16')
                        saved=metrics.read_audio(path)
                        if not np.array_equal(saved,result['audio']):raise ValueError('PCM output changed')
                        values={k:feature.extract(saved[None],fresh=True)[0] for k,feature in objectives.items()}
                        distances={k:float(feature.distances(targets[k][:min(len(targets[k]),len(values[k]))],
                                                            values[k][:min(len(targets[k]),len(values[k]))]).mean())
                                   for k,feature in objectives.items()}
                        if mode=='offline' and abs(distances[kind]-result['final_distance'])>1e-5:
                            raise ValueError('Saved WAV feature distance differs from accepted search')
                        with torch.inference_mode():
                            decoded=model.decode_tokens(torch.as_tensor(result['tokens'],device=args.device)).cpu().numpy().reshape(-1,2,12)
                        ema=overlap_average(decoded,output_length=len(initial))*std+mean
                        original=checked_feature(config,rec['directory']+'.npz')['ema']
                        trajectory=summarize_trajectories({'x':[trajectory_statistics(original,ema,std)]})['x']
                        additional={k:result[k] for k in ('prediction_features','observed_features','control_observed_features','biases','commanded_audio') if k in result}
                        save_npz(directory/'result.npz',initial_tokens=initial,edited_tokens=result['tokens'],edited_ema=ema,
                                 **{k+'_features':v for k,v in values.items()},
                                 **{k+'_times':feature.times(len(values[k])) for k,feature in objectives.items()},**additional)
                        first=transition_statistics(result['tokens'][:,0],prior['transition_probs'],prior['transition_counts'])
                        n=min(len(targets['wavlm']),len(values['wavlm']))
                        summary={'condition':condition,'utterance_id':rec['utterance_id'],'objective':kind,'mode':mode,
                                 'control':control,'weight':weight,'audio':path.relative_to(args.output_dir).as_posix(),
                                 'audio_sha256':sha256_file(path),'result_sha256':sha256_file(directory/'result.npz'),
                                 'trace_sha256':sha256_file(directory/'trace.jsonl'),
                                 'execution_sources':sources,
                                 'wavlm_distance':distances['wavlm'],'ppg_js':distances['ppg'],
                                 'wavlm_raw_mse':float(np.square(targets['wavlm'][:n]-values['wavlm'][:n]).mean()),
                                 'first_stage_nll':first['transition_nll'],'unseen_transition_fraction':first['unseen_transition_fraction'],
                                 'edited_per_stage':np.count_nonzero(initial!=result['tokens'],axis=0).tolist(),
                                 'seconds':time.monotonic()-tick,'evaluations':result['evaluations'],
                                 'synthesis_seconds':evaluator.synthesis_seconds-synth_start,
                                 'synthesis_candidates':evaluator.synthesis_candidates-candidate_start,
                                 'objective_feature_seconds':obj.seconds-feature_start,'objective_feature_calls':obj.calls-feature_calls,
                                 'trajectory':trajectory,**dynamics(ema,std)}
                        if 'terminal_observation' in result:summary['terminal_observation']=result['terminal_observation']
                        write_json(summary_path,summary)
                    manifest['records'].append(summary)
                    save_feature_metrics(args.output_dir/'feature_metrics.csv',manifest['records'])
                    write_report(args.output_dir,manifest,[]);write_json(args.output_dir/'manifest.json',manifest)
                    print(f'DONE {condition} {rec["utterance_id"]}: WavLM={summary["wavlm_distance"]:.5f} PPG={summary["ppg_js"]:.5f} seconds={summary["seconds"]:.1f}',flush=True)
    # ASR is loaded after search to keep GPU memory available for candidate evaluation.
    recognizer=metrics.WhisperRecognizer(args.device)
    audio_rows=[]
    asr_cache={}
    with (args.baseline_dir/'audio_metrics.csv').open() as stream:
        for row in csv.DictReader(stream):asr_cache[row['audio_sha256']]=row['asr_text']
    if (args.output_dir/'audio_metrics.csv').exists():
        with (args.output_dir/'audio_metrics.csv').open() as stream:
            for row in csv.DictReader(stream):asr_cache[row['audio_sha256']]=row['asr_text']
    for rec,initial,fixed,gain,targets,before in prepared:
        reference=metrics.read_audio(args.output_dir/rec['directory']/'original_ema.wav')
        reference_features=metrics.world_features(reference)
        variants=[(v,f'{rec["directory"]}/{v}.wav') for v in ('original','original_ema','before')]
        variants += [(r['condition'],r['audio']) for r in manifest['records'] if r['utterance_id']==rec['utterance_id']]
        for condition,relative in variants:
            path=args.output_dir/relative;digest=sha256_file(path);wave=metrics.read_audio(path)
            text=asr_cache.get(digest)
            if text is None:text=recognizer.transcribe(wave);asr_cache[digest]=text
            audio_rows.append({'condition':condition,'utterance_id':rec['utterance_id'],'audio_sha256':digest,
                               'reference_text':rec['text'],'asr_text':text,**metrics.character_errors(rec['text'],text),
                               **metrics.acoustic_scores(reference,wave,reference_features)})
            save_csv(args.output_dir/'audio_metrics.csv',audio_rows)
        write_report(args.output_dir,manifest,audio_rows);write_json(args.output_dir/'manifest.json',manifest)
    manifest['complete']=True
    write_json(args.output_dir/'manifest.json',manifest)
    print(f'Report: {args.output_dir}/index.html',flush=True)


if __name__=='__main__':main()
