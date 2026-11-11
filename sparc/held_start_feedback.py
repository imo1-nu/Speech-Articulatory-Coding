"""Recover articulation from a held first position using delayed WavLM RVQ control."""

import argparse
import csv
import html
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
from .feature_feedback_search import AcousticEvaluator, delayed
from .feedback_features import FeatureObjective
from .japanese_ppg import sha256_file
from .local_ppg_search import TokenEvaluator, load_generator, transition_statistics
from .model_jsut_tokens import checked_feature, load_run, save_npz, write_json
from .rvq_transition_search import RVQEvaluator, dynamics
from .sequential_ppg_search import audio_hash, save_csv
from .trajectory_rvq import load_rvq


LABELS = {'original':'元音声（WavLM目標）', 'original_ema':'元軌道から合成',
          'normal_rvq':'通常RVQ8再構成（比較用）', 'held_exact':'最初の位置を厳密に固定',
          'held_rvq':'固定軌道をRVQ8で再構成', 'unedited':'逐次・固定トークンのまま',
          'no_feedback':'固定状態からWavLM逐次編集・FBなし',
          'feedback':'固定状態からWavLM逐次編集・FBあり'}


def held_initialization(first_position, frames, model, mean, std, device):
    """Only the first 12-D position and duration enter the RVQ initializer."""
    first_position = np.asarray(first_position, dtype=np.float32)
    if first_position.shape != (12,) or not np.isfinite(first_position).all() or frames < 2:
        raise ValueError('Expected a finite first position and at least two frames')
    held = np.repeat(first_position[None], frames, axis=0)
    window = np.tile((first_position-mean)/std, model.window_frames)[None].astype(np.float32)
    with torch.inference_mode():
        latent = model.ae.encoder(torch.from_numpy(window).to(device))
        _, code = model.quantize(latent)
    tokens = np.repeat(code.cpu().numpy(), frames, axis=0)
    return held, tokens


def audit_control(result, initial, mode):
    """Replay edits, availability limits, emitted chunks and frozen token/audio prefixes."""
    tokens = initial.copy()
    if len(result['trace']) != len(tokens):
        raise ValueError('Missing control decisions')
    accepted = 0
    for start, step in enumerate(result['trace']):
        cutoff = max(0,start*320-960)
        if step['start_token'] != start or step['information_cutoff_samples'] != cutoff:
            raise ValueError('Invalid information delay')
        available = result['audio'][:cutoff]
        if step['observer_input_samples']:
            if step['observer_input_samples'] != cutoff or step['observer_input_sha256'] != audio_hash(available):
                raise ValueError('Observer input mismatch')
        if step['prefix_sha256_before'] != audio_hash(result['audio'][:start*320]):
            raise ValueError('Output prefix changed')
        if step['retained_prefix_sha256_after'] != step['prefix_sha256_before']:
            raise ValueError('Committed audio changed')
        if step['chunk_sha256'] != audio_hash(result['audio'][start*320:(start+1)*320]):
            raise ValueError('Output chunk mismatch')
        for edit in step['edits']:
            position, stage = edit['position'], edit['stage']-1
            if not start <= position < step['planning_end_token']:
                raise ValueError('Edited a committed or out-of-horizon token')
            if tokens[position,stage] != edit['old_code']:
                raise ValueError('Token replay mismatch')
            if edit['objective'] > edit['previous_objective']+1e-12:
                raise ValueError('Accepted objective increased')
            tokens[position,stage] = edit['new_code']
            accepted += int(edit['accepted'])
    np.testing.assert_array_equal(tokens,result['tokens'])
    if mode != 'feedback' and np.any(result['biases']):
        raise ValueError('Feedback applied in control condition')
    if np.linalg.norm(result['biases'],axis=1).max() > 1+1e-10:
        raise ValueError('Feedback bias exceeds bound')
    if result['terminal_observation']['used_for_control']:
        raise ValueError('Terminal observation leaked into control')
    return {'decisions_checked':len(tokens),'accepted_edits_replayed':accepted,
            'information_delay_ms':60,'committed_tokens_and_waveform_immutable':True,
            'non_feedback_bias_zero':mode=='feedback' or not np.any(result['biases'])}


def plot_trajectories(path, original, held, reconstructed, outputs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,2,figsize=(13,8),sharex=True)
    t = np.arange(len(held))*.02
    for ax, channel, name in zip(axes.flat,(0,1,4,5,10,11),('TDX','TDY','TTX','TTY','LLX','LLY')):
        for value,label,style in [(original,'Extracted original','k-'),(held,'Held exact','k:'),
                                  (reconstructed,'Held RVQ','--'),
                                  (outputs['no_feedback'],'WavLM without FB','-'),
                                  (outputs['feedback'],'WavLM with FB','-')]:
            ax.plot(t,value[:,channel],style,label=label,alpha=.8,linewidth=1)
        ax.set_ylabel(name);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);axes[-1,0].set_xlabel('Time (s)');axes[-1,1].set_xlabel('Time (s)')
    fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)


def write_report(output, manifest, rows):
    groups = metrics.aggregate(rows) if rows else []
    if groups: save_csv(output/'audio_aggregates.csv',groups)
    table = ''.join(f'<tr><td>{LABELS[r["condition"]]}</td><td>{r["utterances"]}</td>'
                    f'<td>{100*r["cer"]:.2f}</td><td>{r["mcd_db"]:.3f}</td><td>{r["stoi"]:.4f}</td></tr>' for r in groups)
    features = ''
    for condition in LABELS:
        group = [r for r in manifest['records'] if r['condition']==condition]
        if group:
            features += f'<tr><td>{LABELS[condition]}</td><td>{len(group)}</td><td>{np.mean([r["wavlm_distance"] for r in group]):.5f}</td></tr>'
    cards = []
    for source in manifest['inputs']:
        folder = source['directory']
        players = ''.join(f'<div><h3>{html.escape(LABELS[c])}</h3>{audio_player(r["audio"])}</div>'
                          for c in LABELS for r in manifest['records']
                          if r['utterance_id']==source['utterance_id'] and r['condition']==c)
        transcripts = ''.join(f'<li>{LABELS[r["condition"]]}: {html.escape(r["asr_text"])}</li>'
                              for r in rows if r['utterance_id']==source['utterance_id'])
        plot = f'<img src="{folder}/trajectories.png" style="width:100%" alt="Trajectory comparison">' if (output/folder/'trajectories.png').exists() else ''
        cards.append(f'<section><h2>{source["utterance_id"]}</h2><p>{html.escape(source["text"])}</p><div class="players">{players}</div>{plot}<ul>{transcripts}</ul></section>')
    page = '''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>初期位置固定からのWavLM FB制御</title><style>body{font-family:system-ui;max-width:1400px;margin:25px auto;padding:15px}.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:15px}audio{width:100%}td,th{padding:8px;border-bottom:1px solid #ccc}section{border:1px solid #ccc;margin:20px 0;padding:16px}</style>
<h1>最初の調音位置を保持した状態からのWavLM制御</h1>
<p>全12調音座標を抽出軌道の第0フレームに固定。その一定軌道からRVQ8トークンを作り、全時刻を同一コード列で初期化します。
目標は同じ発話の元音声のWavLM-large hidden_states[9]。元の動的軌道のトークンは編集初期値に使いません。</p>
<p>F0・音量・話者・合成ゲインは元の値を維持します。発声タイミングや抑揚は与えられており、調音のみを回復する実験です。
厳密な固定軌道とRVQ復号後の近似固定軌道は区別して掲載します。RVQの窓復号により端点などに小さな動きが残る場合があります。</p>
<p>20 ms更新、観測は60 ms前まで、探索窓400 ms、各段2位置・1パス、全8段編集、1段目遷移NLL重み0.03。
FBなしも候補音声を生成して目標距離を評価します。FBありは出力と発行時予測の特徴量差で予測を補正します。
元音声の目標特徴量は全体が事前に利用可能です。WavLMは接頭辞への適用で、厳密な因果モデルではありません。</p>
<p>MCD/STOI参照は元軌道合成、CERはWhisper large-v3と正解文。<a href="manifest.json">設定・記録</a> / <a href="feature_metrics.csv">特徴量CSV</a></p>
<h2>WavLM距離（発話平均）</h2><table><tr><th>条件</th><th>発話数</th><th>距離 ↓</th></tr>'''
    page += features+'</table><h2>音声指標</h2><table><tr><th>条件</th><th>発話数</th><th>CER % ↓</th><th>MCD dB ↓</th><th>STOI ↑</th></tr>'+table+'</table>'+''.join(cards)
    page += "<script>document.querySelectorAll('audio').forEach(p=>p.onplay=()=>document.querySelectorAll('audio').forEach(q=>{if(q!==p)q.pause()}));</script></html>"
    (output/'index.html').write_text(page,encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/jsut_held_start_wavlm_feedback'))
    parser.add_argument('--utterances',nargs='+')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args();torch.set_num_threads(4)
    if args.utterances and len(args.utterances)!=len(set(args.utterances)):parser.error('Duplicate utterances')
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:parser.error('Use a new output or --resume')
    baseline_dir = Path('outputs/jsut_rvq/comparison_verified');model_dir = Path('outputs/jsut_rvq/model')
    prior_path = Path('outputs/jsut_rvq_first_stage_transition/transitions.npz')
    baseline = json.loads((baseline_dir/'manifest.json').read_text())
    reconstruction_path = Path('outputs/jsut_reconstruction_ablation/comparison/manifest.json')
    reconstruction = json.loads(reconstruction_path.read_text())
    if sha256_file(reconstruction_path)!=baseline['baseline_manifest_sha256']:
        raise ValueError('Original reconstruction manifest changed')
    rvq_config = json.loads((model_dir/'config.json').read_text())
    config,splits,_,_,_ = load_run(Path(rvq_config['reference_dir']),'cpu')
    sources = {name:sha256_file(Path(__file__).with_name(name)) for name in
               ('held_start_feedback.py','feature_feedback_search.py','feedback_features.py','trajectory_rvq.py',
                'rvq_transition_search.py','local_ppg_search.py','sequential_ppg_search.py','evaluate_generated_audio.py')}
    settings = {'output_dir':str(args.output_dir),'utterances':args.utterances,'device':args.device,
                'initial_position_frame':0,'horizon_ms':400,'step_ms':20,'delay_ms':60,'transition_weight':.03}
    dependencies = {str(p):sha256_file(p) for p in (baseline_dir/'manifest.json',model_dir/'codebooks.npz',prior_path,reconstruction_path)}
    if args.resume:
        manifest = json.loads((args.output_dir/'manifest.json').read_text())
        if manifest['sources']!=sources or manifest['settings']!=settings or manifest['dependencies']!=dependencies:
            raise ValueError('Source, dependencies or settings changed')
    else:
        selected = [r for r in baseline['audio_records'] if args.utterances is None or r['utterance_id'] in args.utterances]
        if not selected or (args.utterances and len(selected)!=len(args.utterances)):raise ValueError('Unknown utterance')
        manifest = {'complete':False,'settings':settings,'sources':sources,'dependencies':dependencies,
                    'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    'inputs':selected,'records':[], 'initialization': 'all 12 coordinates held at first extracted frame; repeated RVQ code tuple; original F0/loudness/speaker/gain retained'}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_json(args.output_dir/'manifest.json',manifest)
    model,mean,std = load_rvq(model_dir,8,args.device)
    prior = dict(np.load(prior_path))
    if str(prior['codebooks_sha256'])!=baseline['codebooks_sha256'] or dependencies[str(model_dir/'codebooks.npz')]!=baseline['codebooks_sha256']:
        raise ValueError('RVQ/prior provenance mismatch')
    generator,generator_hash = load_generator(args.device)
    if generator_hash!=reconstruction['generator_sha256']:raise ValueError('Generator changed')
    objective = FeatureObjective('wavlm',args.device)
    if baseline['asr_model']!=metrics.ASR_MODEL or baseline['asr_revision']!=metrics.ASR_REVISION:
        raise ValueError('ASR provenance mismatch')
    manifest.update({'generator_sha256':generator_hash,'feature':objective.metadata,'asr_model':metrics.ASR_MODEL,
                     'asr_revision':metrics.ASR_REVISION,'asr_settings':baseline['asr_settings']})
    source_audio = json.loads((Path(rvq_config['reference_dir'])/'audio_comparison/manifest.json').read_text())
    with (baseline_dir/'audio_metrics.csv').open() as stream:
        original_hashes = {(r['utterance_id'],r['condition']):r['audio_sha256'] for r in csv.DictReader(stream)}
    for rec in manifest['inputs']:
        folder = rec['directory'];destination = args.output_dir/folder;destination.mkdir(parents=True,exist_ok=True)
        if folder+'.npz' not in splits['test']:raise ValueError('Not a test utterance')
        original = checked_feature(config,folder+'.npz')['ema']
        fixed = dict(np.load(Path(rvq_config['reference_dir'])/'audio_comparison'/folder/'conditions.npz'))
        gain = next(r['synthesis_common_gain'] for r in source_audio['records'] if r['utterance_id']==rec['utterance_id'])
        for variant,name in [('original','original'),('original_ema','original_ema'),('rvq8','normal_rvq')]:
            source = baseline_dir/folder/f'{variant}.wav'
            if sha256_file(source)!=original_hashes[(rec['utterance_id'],variant)]:raise ValueError('Baseline changed')
            shutil.copyfile(source,destination/f'{name}.wav')
        held,initial = held_initialization(original[0],len(original),model,mean,std,args.device)
        evaluator = AcousticEvaluator(model,mean,std,generator,objective,fixed,gain,args.device)
        decoded = RVQEvaluator(model,mean,std,generator,None,fixed,gain,args.device).trajectories(initial[None])[0]
        # Direct synthesis from [first position, first position], bypassing RVQ for the exact-hold control.
        windows = np.repeat(held[0][None,None,:],2,axis=1)
        direct = TokenEvaluator(generator,None,windows,fixed,gain,args.device)
        exact_audio = direct.synthesize(np.zeros((1,len(held)),dtype=np.int64))[0,0].cpu().numpy()
        sf.write(destination/'held_exact.wav',exact_audio,16000,subtype='PCM_16')
        held_audio = evaluator.at(len(initial))(initial[None])[0][0]
        sf.write(destination/'held_rvq.wav',held_audio,16000,subtype='PCM_16')
        save_npz(destination/'initialization.npz',first_position=original[0],held_ema=held,held_rvq_ema=decoded,initial_tokens=initial,
                 original_ema=original,pitch=fixed['pitch'],loudness=fixed['loudness'],spk_emb=fixed['spk_emb'])
        natural = metrics.read_audio(destination/'original.wav');target = objective.extract(natural[None])[0]
        save_npz(destination/'target.npz',features=target,times=objective.times(len(target)))
        outputs = {}
        for mode in ('unedited','no_feedback','feedback'):
            directory = destination/mode;directory.mkdir(exist_ok=True)
            summary_path = directory/'summary.json'
            if args.resume and summary_path.exists():
                summary = json.loads(summary_path.read_text())
                for filename,key in [('after.wav','audio_sha256'),('result.npz','result_sha256'),('trace.jsonl','trace_sha256')]:
                    if sha256_file(directory/filename)!=summary[key]:raise ValueError('Saved control result changed')
                edited = np.load(directory/'result.npz')['edited_ema']
            else:
                tick = time.monotonic()
                with (directory/'trace.jsonl').open('w') as stream:
                    def step(item):
                        stream.write(json.dumps(item)+'\n');stream.flush()
                        if item['start_token']%25==0:print(f'{rec["utterance_id"]} {mode} t={item["decision_time_ms"]}ms evaluations={item["evaluations"]}',flush=True)
                    result = delayed(initial,target,evaluator,prior['transition_probs'],mode=mode,on_step=step)
                audit = audit_control(result,initial,mode)
                sf.write(directory/'after.wav',result['audio'],16000,subtype='PCM_16')
                saved = metrics.read_audio(directory/'after.wav')
                np.testing.assert_array_equal(saved,result['audio'])
                edited = RVQEvaluator(model,mean,std,generator,None,fixed,gain,args.device).trajectories(result['tokens'][None])[0]
                values = objective.extract(saved[None],fresh=True)[0]
                n = min(len(values),len(target))
                save_npz(directory/'result.npz',initial_tokens=initial,edited_tokens=result['tokens'],edited_ema=edited,
                         wavlm_features=values,**{k:result[k] for k in ('prediction_features','observed_features','control_observed_features','biases','commanded_audio')})
                first = transition_statistics(result['tokens'][:,0],prior['transition_probs'],prior['transition_counts'])
                summary = {'condition':mode,'utterance_id':rec['utterance_id'],'audio':f'{folder}/{mode}/after.wav',
                           'audio_sha256':sha256_file(directory/'after.wav'),'result_sha256':sha256_file(directory/'result.npz'),
                           'trace_sha256':sha256_file(directory/'trace.jsonl'),'wavlm_distance':float(objective.distances(target[:n],values[:n]).mean()),
                           'edited_per_stage':np.count_nonzero(initial!=result['tokens'],axis=0).tolist(),
                           'seconds':time.monotonic()-tick,'evaluations':result['evaluations'],'verification':audit,
                           'transition':first,'trajectory_mse_sd':float(np.square((edited-original)/std).mean()),**dynamics(edited,std)}
                write_json(summary_path,summary)
            outputs[mode] = edited
            manifest['records'] = [r for r in manifest['records'] if (r['utterance_id'],r['condition'])!=(rec['utterance_id'],mode)]+[summary]
            write_json(args.output_dir/'manifest.json',manifest);write_report(args.output_dir,manifest,[])
            print(f'DONE {rec["utterance_id"]} {mode}: WavLM={summary["wavlm_distance"]:.5f} seconds={summary["seconds"]:.1f}',flush=True)
        for condition in ('original','original_ema','normal_rvq','held_exact','held_rvq'):
            path = destination/f'{condition}.wav';values = objective.extract(metrics.read_audio(path)[None],fresh=True)[0]
            n = min(len(target),len(values))
            row = {'condition':condition,'utterance_id':rec['utterance_id'],'audio':f'{folder}/{condition}.wav',
                   'audio_sha256':sha256_file(path),'wavlm_distance':float(objective.distances(target[:n],values[:n]).mean())}
            manifest['records'] = [r for r in manifest['records'] if (r['utterance_id'],r['condition'])!=(rec['utterance_id'],condition)]+[row]
        plot_trajectories(destination/'trajectories.png',original,held,decoded,outputs)
        write_json(args.output_dir/'manifest.json',manifest);write_report(args.output_dir,manifest,[])
    save_csv(args.output_dir/'feature_metrics.csv',[{k:r[k] for k in ('condition','utterance_id','audio','audio_sha256','wavlm_distance')} for r in manifest['records']])
    # Release search models before loading Whisper-large-v3.
    del evaluator,model,generator,objective,direct
    if args.device.startswith('cuda'):torch.cuda.empty_cache()
    recognizer = metrics.WhisperRecognizer(args.device);rows = [];cache = {}
    for cache_path in (baseline_dir/'audio_metrics.csv',args.output_dir/'audio_metrics.csv'):
        if cache_path.exists():
            with cache_path.open() as stream:
                cache.update({r['audio_sha256']:r['asr_text'] for r in csv.DictReader(stream)})
    for source in manifest['inputs']:
        reference = metrics.read_audio(args.output_dir/source['directory']/'original_ema.wav')
        reference_features = metrics.world_features(reference)
        for condition in LABELS:
            r = next(r for r in manifest['records'] if r['utterance_id']==source['utterance_id'] and r['condition']==condition)
            wave = metrics.read_audio(args.output_dir/r['audio']);digest = sha256_file(args.output_dir/r['audio'])
            if digest not in cache:cache[digest] = recognizer.transcribe(wave)
            text = cache[digest]
            rows.append({'condition':condition,'utterance_id':source['utterance_id'],'audio_sha256':digest,
                         'reference_text':source['text'],'asr_text':text,**metrics.character_errors(source['text'],text),
                         **metrics.acoustic_scores(reference,wave,reference_features)})
            save_csv(args.output_dir/'audio_metrics.csv',rows)
    manifest['complete'] = True
    write_report(args.output_dir,manifest,rows);write_json(args.output_dir/'manifest.json',manifest)
    print(f'Report: {args.output_dir}/index.html',flush=True)


if __name__ == '__main__':main()
