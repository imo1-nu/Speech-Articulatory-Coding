"""20 ms control steps with a strict t-60 ms acoustic-information cutoff."""

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import soundfile as sf
import torch

from . import local_ppg_search as local
from . import sequential_ppg_search as seq
from .japanese_ppg import JapanesePPGExtractor, frame_times, load_audio, sha256_file
from .model_jsut_tokens import configure_plots, save_npz, write_json


def delayed_search(initial, target, normalized, allowed, probabilities, evaluator, *,
                   horizon_tokens, mode='feedback', delay_samples=960, on_step=None, plant=None):
    if mode not in seq.MODES or horizon_tokens < 1 or delay_samples < 0 or delay_samples % 160:
        raise ValueError('Invalid delayed-control protocol')
    tokens = initial.copy()
    emitted = np.empty(0, dtype=np.float32)
    commanded = np.empty(0, dtype=np.float32)
    continuation = np.empty(0, dtype=np.float32)
    forecasts = np.empty((0, 35), dtype=np.float32)
    observer = seq.DelayedObserver(lag_frames=0)
    trace, biases = [], []
    for start in range(len(tokens)):
        current_samples = start*320
        cutoff = max(0, current_samples-delay_samples)
        available = emitted[:cutoff].copy()
        previous_ready = len(observer.observed)
        # The STFT reflect pad needs >432 real input samples. Do not pad with unknown audio.
        if len(available) > 432:
            observed_prefix = evaluator.observe(available)
            fresh_frames = len(observed_prefix)-previous_ready
            observer.rate = 1-.75**(fresh_frames/10) if fresh_frames else 0
            observation = observer.update(observed_prefix, forecasts, feedback=mode == 'feedback')
        else:
            observation = {'observed_from_frame': previous_ready, 'observed_until_frame': previous_ready,
                           'innovation_norm': 0., 'bias_norm': float(np.linalg.norm(observer.bias)), 'eos_flush': False}
        if observation['observed_until_frame'] > cutoff//160:
            raise ValueError('Feedback looked past the information cutoff')
        # Blind last 60 ms come from issued model predictions, never from acoustic observations.
        prediction_context = commanded.copy()
        prediction_context[:cutoff] = available
        bias_before = observer.bias.copy()
        end = min(len(tokens), start+horizon_tokens)
        evaluate = evaluator.at(end, prediction_context, continuation, 160)
        frozen_tokens = tokens[:start].copy()
        planned, audio, ppg, stats = seq.plan_step(tokens[:end], start, target, evaluate,
                         normalized, allowed, probabilities, bias_before, optimize=mode != 'unedited',
                         weight=.03, gain=.5 if mode == 'feedback' else 0,
                         passes=1, positions=2, include_current=True)
        tokens[:end] = planned
        if not np.array_equal(tokens[:start], frozen_tokens):
            raise ValueError('A committed token changed')
        commit = start+1
        chunk = audio[current_samples:commit*320].copy()
        # These predictions are known at command issue time, independently of the plant output.
        commanded = np.concatenate((commanded, chunk))
        output = chunk.copy() if plant is None else seq.pcm16(plant(chunk.copy()))
        if output.shape != chunk.shape:
            raise ValueError('Plant changed chunk length')
        old_audio = emitted.copy()
        emitted = np.concatenate((emitted, output))
        if not np.array_equal(emitted[:len(old_audio)], old_audio):
            raise ValueError('Previously emitted samples changed')
        continuation = audio[commit*320:].copy()
        forecasts = np.concatenate((forecasts, ppg[start*2:commit*2]))
        item = {'start_token': start, 'planning_end_token': end, 'committed_until_token': commit,
                'decision_time_ms': start*20, 'information_cutoff_samples': cutoff,
                'information_cutoff_ms': cutoff/16, 'observer_input_samples': len(available) if len(available)>432 else 0,
                'observer_input_sha256': seq.audio_hash(available) if len(available)>432 else None,
                'prediction_context_sha256': seq.audio_hash(prediction_context),
                'blind_predicted_context_sha256': seq.audio_hash(prediction_context[cutoff:]),
                'prefix_sha256_before': seq.audio_hash(old_audio),
                'retained_prefix_sha256_after': seq.audio_hash(emitted[:len(old_audio)]),
                'chunk_sha256': seq.audio_hash(output), 'command_chunk_sha256': seq.audio_hash(chunk),
                'bias_before_norm': float(np.linalg.norm(bias_before)), 'feedback_update_rate': observer.rate,
                **observation, **stats}
        trace.append(item); biases.append(bias_before)
        if on_step:
            on_step(item)
    observed_control = observer.observed.copy()
    # Analysis-only completion, after the output ends and the delay has elapsed; no decisions follow.
    terminal = observer.update(evaluator.observe(emitted), forecasts, feedback=False, final=True)
    terminal.update(observation_time_ms=len(tokens)*20+delay_samples/16, used_for_control=False)
    return {'tokens': tokens, 'audio': emitted, 'commanded_audio': commanded, 'prediction_ppg': forecasts,
            'observed_ppg': observer.observed, 'control_observed_ppg': observed_control,
            'biases': np.asarray(biases), 'trace': trace, 'terminal_observation': terminal}


METHOD = '''<h1>観測情報t−60 ms・更新20 ms・探索200 / 400 / 800 ms</h1>
<p>各時刻tで、出力音声のうちt−60 msまでだけをPPG観測器へ渡します。
PPGの時刻だけを切り詰めるのではなく、入力波形自体を切り詰めます。最初の20 ms分だけではSTFTのreflect paddingが成立しないため、
40 msの観測が届くt=100 msからPPG補正を開始します。観測PPGは到着時に確定して保持します。</p>
<p>現在から先の200 / 400 / 800 msを編集し、先頭1トークン（20 ms）だけを確定します。
直近60 msの未観測音声を予測器の文脈へ流用せず、その区間には発行済みコマンドの予測波形を使います。
過去の出力音声・トークンは変更しません。区間先頭の10 msを前回の未出力予測とクロスフェードします。
各ステップで現在位置と差の大きい未来1位置を1パス、各16候補で探索します。λ=0.03。
補正は音素別log確率バイアス（上限±1、適用ゲイン0.5）で、更新率は100 msあたり0.25の時定数を保つよう20 msで約0.0559に換算します。</p>
<p>補正なしも同じ情報制限・探索予算とし、バイアスをゼロに固定します。
既存の非因果PPGモデルを利用可能な波形の接頭辞だけに適用する近似であり、発話全体のPPGと同じとは限りません。
末尾の観測補完は出力終了後60 ms待った解析専用処理で、制御には使いません。
目標PPG・初期トークン・F0等は事前抽出済みで、入力側までストリーミング化した実験ではありません。</p>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('outputs/jsut_window_sweep/window_040ms'))
    parser.add_argument('--ppg-dir', type=Path, default=Path('outputs/jsut_ppg_40ms128'))
    parser.add_argument('--offline-dir', type=Path, default=Path('outputs/jsut_ppg_soft_transition/condition_02_lambda_0.03'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/jsut_ppg_delay060_step020'))
    parser.add_argument('--horizons', nargs='+', type=int, default=[200, 400, 800])
    parser.add_argument('--utterances', nargs='+', default=local.DEFAULT_IDS)
    parser.add_argument('--device', default='cuda'); parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    if (len(set(args.horizons))!=len(args.horizons) or any(h<20 or h%20 for h in args.horizons)
            or len(set(args.utterances))!=len(args.utterances) or args.threads<1):
        parser.error('Invalid horizons or threads')
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error('Choose a new output directory')
    torch.set_num_threads(args.threads); args.transition_alpha = 1.
    source, digest, normalized, _, counts, selected, model, mean, std, markov = local.prepare(args)
    ref_manifest = json.loads((args.offline_dir/'manifest.json').read_text())
    if ref_manifest['vq_checkpoint_sha256']!=digest or ref_manifest['settings']['transition_weight']!=.03:
        raise ValueError('Offline reference mismatch')
    offline = [next(r for r in ref_manifest['records'] if r['utterance_id']==u) for u in args.utterances]
    generator, generator_digest = local.load_generator(args.device); extractor = JapanesePPGExtractor(args.device)
    if extractor.checkpoint_sha256!=source['ppg_model']['checkpoint_sha256']:
        raise ValueError('PPG checkpoint mismatch')
    configure_plots(); args.output_dir.mkdir(parents=True, exist_ok=True)
    save_npz(args.output_dir/'transitions.npz', **markov, alpha=np.float64(1), vq_checkpoint_sha256=np.str_(digest))
    references = []
    for ref, (record, _, _, _, _) in zip(offline, selected):
        folder = args.output_dir/'references'/ref['utterance_id']; folder.mkdir(parents=True)
        for name, path in [('original', args.ppg_dir/record['variants']['original']['audio']),
                           ('before', args.offline_dir/ref['directory']/'before.wav'),
                           ('offline', args.offline_dir/ref['directory']/'after.wav')]:
            shutil.copyfile(path, folder/f'{name}.wav'); references.append({'source':str(path.resolve()), 'sha256':sha256_file(path)})
    manifest = {'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                'settings':{'commit_ms':20,'acoustic_information_delay_ms':60,'observer_input_is_truncated':True,
                            'blind_context_source':'issued model prediction, not actual waveform',
                            'positions_per_step':2,'passes_per_step':1,'force_current_position':True,
                            'transition_weight':.03,'feedback_gain':.5,'feedback_log_bias_clip':1,
                            'feedback_update_rate_per_100ms':.25,'crossfade_ms':10},
                'vq_checkpoint_sha256':digest,'generator_checkpoint_sha256':generator_digest,
                'ppg_checkpoint_sha256':extractor.checkpoint_sha256,
                'offline_manifest_sha256':sha256_file(args.offline_dir/'manifest.json'),
                'references':references,'records':[],'aggregates':[]}
    rows, conditions = [], []
    for horizon in args.horizons:
        for mode in seq.MODES:
            condition=f'horizon_{horizon:03d}ms_{mode}'; conditions.append(condition)
            for record, fixed, initial, target, gain in selected:
                directory=args.output_dir/condition/Path(record['feature_path']).with_suffix(''); directory.mkdir(parents=True)
                evaluator=seq.PrefixEvaluator(generator,extractor,normalized,fixed,gain,args.device,model,mean,std)
                tick=time.monotonic()
                with (directory/'trace.jsonl').open('w') as log:
                    def on_step(item):
                        log.write(json.dumps(item)+'\n'); log.flush()
                        if item['committed_until_token']%10==0 or item['committed_until_token']==len(initial):
                            print(f'{condition} {record["utterance_id"]} t={item["decision_time_ms"]}ms cutoff={item["information_cutoff_ms"]}ms JS(plan)={item["final_planning_js"]:.5f}',flush=True)
                    result=delayed_search(initial,target,normalized,np.flatnonzero(counts),markov['transition_probs'],evaluator,
                                          horizon_tokens=horizon//20,mode=mode,on_step=on_step)
                sf.write(directory/'output.wav',result['audio'],16000,subtype='PCM_16')
                saved,sr=sf.read(directory/'output.wav',dtype='float32')
                if sr!=16000 or not np.array_equal(saved,result['audio']): raise ValueError('Saved samples changed')
                final_ppg=extractor.extract(load_audio(directory/'output.wav')[0])
                distances=local.posterior_score(target,final_ppg)
                observed_distance=local.posterior_score(target,result['observed_ppg'])
                n=len(distances)
                row={'condition':condition,'horizon_ms':horizon,'mode':mode,'utterance_id':record['utterance_id'],
                     'directory':directory.relative_to(args.output_dir).as_posix(), 'mean_js':float(distances.mean()),
                     'fixed_lag_js':float(observed_distance.mean()),'ppg_frames':n,
                     'control_observed_frames':len(result['control_observed_ppg']),
                     'control_observed_js':float(local.posterior_score(target,result['control_observed_ppg']).mean()),
                     'argmax_agreement':float((target[:n].argmax(1)==final_ppg[:n].argmax(1)).mean()),
                     'edited_tokens':int(np.count_nonzero(initial!=result['tokens'])),'token_count':len(initial),
                     'edges':len(initial)-1,'boundaries':len(initial)-1,
                     'evaluations':sum(x['evaluations'] for x in result['trace']),'seconds':time.monotonic()-tick,
                     'feedback_updates':sum(x['observed_until_frame']>x['observed_from_frame'] for x in result['trace']) if mode=='feedback' else 0,
                     **local.transition_statistics(result['tokens'],markov['transition_probs'],markov['transition_counts']),
                     **seq.boundary_statistics(result['audio'],320)}
                save_npz(directory/'result.npz',initial_tokens=initial,edited_tokens=result['tokens'],final_ppg=final_ppg,
                         target_ppg=target,observed_ppg=result['observed_ppg'],control_observed_ppg=result['control_observed_ppg'],
                         committed_prediction_ppg=result['prediction_ppg'],commanded_audio=result['commanded_audio'],
                         feedback_biases=result['biases'],time=frame_times(len(final_ppg)),js=distances,
                         pitch=fixed['pitch'],loudness=fixed['loudness'],spk_emb=fixed['spk_emb'])
                write_json(directory/'terminal_observation.json',result['terminal_observation'])
                before=np.load(args.ppg_dir/record['variants']['reconstructed_ema']['ppg'])['ppg']
                seq.plot_utterance(directory/'ppg.png',target,before,final_ppg,result['observed_ppg'],result['trace'])
                write_json(directory/'summary.json',row); rows.append(row)
                manifest['records']=rows
                manifest['aggregates']=seq.write_report(args.output_dir,rows,offline,conditions,method_html=METHOD)
                write_json(args.output_dir/'manifest.json',manifest)
                print(f'{condition} {record["utterance_id"]}: final JS={row["mean_js"]:.6f} NLL={row["transition_nll"]:.4f}',flush=True)
    print(f'Report: {args.output_dir}/index.html',flush=True)
    return 0


if __name__=='__main__':
    main()
