"""Offline PPG-guided editing of all RVQ stages, with a first-stage-only transition prior."""

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
from tqdm import tqdm

from . import evaluate_generated_audio as metrics
from .compare_jsut_audio import audio_player
from .compare_trajectory_reconstruction import trajectory_statistics, summarize_trajectories
from .japanese_ppg import JapanesePPGExtractor, sha256_file
from .local_ppg_search import TokenEvaluator, load_generator, posterior_score, search_tokens, transition_nll, transition_statistics
from .model_jsut_tokens import CachedWindowBatches, checked_feature, load_run, save_npz, write_json
from .sequential_ppg_search import save_csv
from .trajectory_rvq import load_rvq
from .trajectory_vq import WindowDataset, fit_markov, overlap_average, score_markov


class RVQEvaluator(TokenEvaluator):
    def __init__(self, model, mean, std, generator, extractor, conditions, gain, device):
        super().__init__(generator, extractor, None, conditions, gain, device)
        self.model, self.mean, self.std = model, mean, std

    def trajectories(self, sequences):
        with torch.inference_mode():
            return np.stack([overlap_average(self.model.decode_tokens(
                torch.as_tensor(z, dtype=torch.long, device=self.device)).cpu().numpy().reshape(len(z),2,12),
                output_length=len(z))*self.std+self.mean for z in sequences])


def search_rvq(initial, target, evaluate, codebooks, probabilities, weight, *, passes=2,
               positions=24, neighbors=12, global_candidates=4, batch_size=4, on_step=None):
    """Coordinate search over every stage; only column zero enters the NLL.

    Later-stage updates leave first-stage NLL constant, so the existing JS-only
    coordinate search is exactly equivalent to the full objective for those updates.
    Each candidate changes one ID; the other seven IDs are not automatically re-encoded.
    """
    if not np.isfinite(weight) or weight < 0 or passes < 1:
        raise ValueError('Invalid transition weight or pass count')
    tokens = np.asarray(initial, dtype=np.int64).copy()
    if tokens.ndim != 2 or not len(tokens) or tokens.shape[1] != len(codebooks):
        raise ValueError('Expected nonempty time-by-stage tokens')
    if tokens.min() < 0 or tokens.max() >= codebooks.shape[1]:
        raise ValueError('Invalid RVQ token IDs')
    baseline = np.asarray(evaluate(tokens[None]))[0]
    trace = []; evaluations = 1
    for pass_index in range(passes):
        accepted = 0
        for stage in range(tokens.shape[1]):
            stage_base = tokens.copy()
            def evaluate_stage(sequences):
                proposals = np.repeat(stage_base[None],len(sequences),axis=0)
                proposals[:,:,stage] = sequences
                return evaluate(proposals)
            constant_nll = transition_nll(tokens[:,0],probabilities)
            offset = evaluations
            def step(item):
                nll = item['current_transition_nll'] if stage == 0 else constant_nll
                entry = {'pass':pass_index+1,'stage':stage+1,'position':item['position'],
                         'old_code':item['old_code'],'new_code':item['new_code'],
                         'accepted':item['accepted'],'candidate_count':item['candidate_count'],
                         'current_js':item['current_js'],'first_stage_nll':nll,
                         'current_objective':item['current_js']+weight*nll,
                         'evaluations':offset+item['evaluations']}
                trace.append(entry)
                if on_step: on_step(entry)
            result = search_tokens(tokens[:,stage],target,evaluate_stage,codebooks[stage,:,None,:],
                                   np.arange(codebooks.shape[1]),passes=1,positions=positions,
                                   neighbors=neighbors,global_candidates=global_candidates,batch_size=batch_size,
                                   transition_probs=probabilities if stage == 0 else None,
                                   transition_weight=weight if stage == 0 else 0,on_step=step)
            tokens[:,stage] = result['tokens']
            accepted += sum(r['accepted'] for r in result['trace'])
            evaluations += result['evaluations']
        if not accepted: break
    final = np.asarray(evaluate(tokens[None]))[0]
    return {'tokens':tokens,'baseline_ppg':baseline,'ppg':final,'trace':trace,'evaluations':evaluations+1,
            'baseline_js':float(posterior_score(target,baseline).mean()),
            'final_js':float(posterior_score(target,final).mean())}


def first_stage_sequences(model, config, relatives, mean, std, device):
    sequences = []
    for relative in tqdm(relatives,desc='Encode first-stage transitions'):
        ema = checked_feature(config,relative)['ema']
        windows = CachedWindowBatches(WindowDataset([ema],mean,std,2,1),4096,device)
        codes = []
        with torch.inference_mode():
            centers = model.codebooks[0]
            for batch in windows:
                latent = model.ae.encoder(batch)
                distance = latent.square().sum(1,keepdim=True)+centers.square().sum(1)-2*latent@centers.T
                codes.append(distance.argmin(1).cpu().numpy())
        sequences.append(np.concatenate(codes))
    return sequences


def dynamics(ema,std):
    velocity = np.diff(ema/std,axis=0)*50
    acceleration = np.diff(velocity,axis=0)*50
    return {'velocity_rms_sd_per_s':float(np.sqrt(np.square(velocity).mean())),
            'acceleration_rms_sd_per_s2':float(np.sqrt(np.square(acceleration).mean()))}


def report(output, manifest, summaries, audio_rows):
    audio = metrics.aggregate(audio_rows)
    save_csv(output/'audio_aggregates.csv',audio)
    aggregated = []
    for weight in manifest['settings']['weights']:
        group = [r for r in summaries if r['weight']==weight]
        if not group: continue
        row = {'weight':weight,'utterances':len(group)}
        for key in ('baseline_js','final_js','baseline_nll','final_nll','final_unseen_transition_fraction',
                    'final_self_transition_fraction','velocity_rms_sd_per_s','acceleration_rms_sd_per_s2'):
            row[key] = float(np.mean([r[key] for r in group]))
        row['edited_stage1'] = sum(r['edited_per_stage'][0] for r in group)
        row['edited_other_stages'] = sum(sum(r['edited_per_stage'][1:]) for r in group)
        row['evaluations'] = sum(r['evaluations'] for r in group)
        aggregated.append(row)
    if aggregated: save_csv(output/'search_aggregates.csv',aggregated)
    table = ''.join(f'<tr><td>{r["condition"]}</td><td>{100*r["cer"]:.2f}</td><td>{r["mcd_db"]:.4f}</td><td>{r["stoi"]:.4f}</td><td>{r["f0_rmse_hz"]:.2f}</td></tr>' for r in audio)
    searches = ''.join(f'<tr><td>{r["weight"]:g}</td><td>{r["final_js"]:.5f}</td><td>{r["final_nll"]:.4f}</td><td>{100*r["final_unseen_transition_fraction"]:.2f}</td><td>{r["edited_stage1"]}</td><td>{r["edited_other_stages"]}</td><td>{r["velocity_rms_sd_per_s"]:.2f}</td><td>{r["acceleration_rms_sd_per_s2"]:.2f}</td></tr>' for r in aggregated)
    cards = []
    for record in manifest['audio_records']:
        folder = record['directory']
        variants = [('original','元音声'),('original_ema','元軌道合成'),('before','RVQ8・編集前')]
        variants += [(f'lambda_{w:g}',f'λ={w:g}') for w in manifest['settings']['weights'] if any(r['utterance_id']==record['utterance_id'] and r['weight']==w for r in summaries)]
        players = ''.join(f'<div><h3>{label}</h3>{audio_player(folder+"/"+v+".wav")}</div>' for v,label in variants)
        transcripts = ''.join(f'<li>{r["condition"]}: {html.escape(r["asr_text"])} (CER {100*r["cer"]:.1f}%)</li>' for r in audio_rows if r['utterance_id']==record['utterance_id'])
        cards.append(f'<section><h2>{record["utterance_id"]}</h2><p>{html.escape(record["text"])}</p><div class="players">{players}</div><ul>{transcripts}</ul></section>')
    page = '''<!doctype html><html lang="ja"><meta charset="utf-8"><title>RVQ1段目の遷移ソフト制約</title>
<style>body{font-family:system-ui;max-width:1400px;margin:30px auto;padding:16px}.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}audio{width:100%}td,th{padding:8px;border-bottom:1px solid #ccc}section{border:1px solid #ccc;margin:20px 0;padding:16px}</style>
<h1>RVQ8：1段目だけの遷移ソフト制約</h1>
<p>オフラインの全発話PPG局所探索。全8段を編集し、目的関数は元音声PPGとの平均JS＋λ×1段目の平均遷移NLL。
2〜8段の遷移制約はなし。各候補で1つのIDだけを変更し、残りの段を再量子化しない。
訓練4500発話でRVQ専用遷移表を学習、加算平滑化α=1、自己遷移を含み発話間の遷移は数えない。</p>
<p>全条件で同じ2パス×8段×最大24時刻、各時刻12近傍＋4広域候補、バッチ4。パス内に採用がなければ早期終了。
段1→8の順に探索。候補は段ごとの潜在コードベクトル距離から選び、採用前に単独再評価する。
同一の予算規則だが採用経路や再検証回数は条件によって変わる。これは観測遅延を設けたオンラインFB実験ではない。</p>
<p>音声は既存テスト5発話。F0・音量・話者・ゲイン固定。MCD/STOI/F0の参照は元軌道合成、CERはWhisper large-v3とJSUT正解文。
PPG JS、遷移NLL、速度・加速度の小ささだけでは明瞭度や物理的妥当性を保証しない。速度と加速度は訓練SDで標準化した軌道の差分。</p>
<p><a href="manifest.json">全設定・記録</a> / <a href="summary.csv">探索CSV</a> / <a href="audio_aggregates.csv">音声集計</a> / <a href="prior_diagnostics.json">遷移推定の診断</a></p>
<h2>探索結果（発話平均）</h2><table><tr><th>λ</th><th>JS ↓</th><th>段1 NLL ↓</th><th>未観測遷移 %</th><th>段1編集数</th><th>他段編集数</th><th>速度RMS</th><th>加速度RMS</th></tr>'''
    page += searches+'</table><h2>音声評価</h2><table><tr><th>条件</th><th>CER % ↓</th><th>MCD dB ↓</th><th>STOI ↑</th><th>F0 RMSE Hz ↓</th></tr>'+table+'</table>'+''.join(cards)
    page += "<script>document.querySelectorAll('audio').forEach(p=>p.onplay=()=>document.querySelectorAll('audio').forEach(q=>{if(q!==p)q.pause()}));</script></html>"
    (output/'index.html').write_text(page,encoding='utf-8')
    return aggregated,audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,default=Path('outputs/jsut_rvq/model'))
    parser.add_argument('--baseline-dir',type=Path,default=Path('outputs/jsut_rvq/comparison_verified'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/jsut_rvq_first_stage_transition'))
    parser.add_argument('--weights',type=float,nargs='+',default=[0,.01,.03,.1])
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args()
    if (not np.isfinite(args.weights).all() or min(args.weights)<0 or len(set(args.weights))!=len(args.weights)
            or (args.output_dir.exists() and any(args.output_dir.iterdir()))): parser.error('Invalid weights or existing output')
    torch.set_num_threads(4)
    model,mean,std = load_rvq(args.model_dir,8,args.device)
    rvq_config = json.loads((args.model_dir/'config.json').read_text())
    config,splits,_,rmean,rstd = load_run(Path(rvq_config['reference_dir']),'cpu')
    np.testing.assert_array_equal(mean,rmean);np.testing.assert_array_equal(std,rstd)
    baseline = json.loads((args.baseline_dir/'manifest.json').read_text())
    if (sha256_file(args.model_dir/'codebooks.npz') != baseline['codebooks_sha256']
            or json.loads((args.model_dir/'splits.json').read_text()) != splits): raise ValueError('RVQ provenance mismatch')
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    sequences = {name:first_stage_sequences(model,config,relatives,mean,std,args.device) for name,relatives in splits.items()}
    markov = fit_markov(sequences['train'],128,1.)
    save_npz(args.output_dir/'transitions.npz',**markov,alpha=np.float64(1),codebooks_sha256=np.str_(baseline['codebooks_sha256']))
    diagnostics = {'fit_split':'train','fit_utterances':len(splits['train']),
                   'observed_transition_types':int(np.count_nonzero(markov['transition_counts'])),
                   'possible_transition_types':128*128,'training_edges':int(markov['transition_counts'].sum()),
                   'split_scores':{name:score_markov(seq,markov) for name,seq in sequences.items()},
                   'split_unseen_transition_fraction':{name:float(sum(np.count_nonzero(markov['transition_counts'][z[:-1],z[1:]]==0) for z in seq)/sum(len(z)-1 for z in seq)) for name,seq in sequences.items()}}
    write_json(args.output_dir/'prior_diagnostics.json',diagnostics)
    generator,generator_hash = load_generator(args.device)
    original_manifest = json.loads(Path('outputs/jsut_reconstruction_ablation/comparison/manifest.json').read_text())
    if generator_hash != original_manifest['generator_sha256']: raise ValueError('Generator changed')
    extractor = JapanesePPGExtractor(args.device)
    if extractor.checkpoint_sha256 != original_manifest['ppg_sha256']: raise ValueError('PPG changed')
    recognizer = metrics.WhisperRecognizer(args.device)
    manifest = {'settings':{'weights':args.weights,'stages':8,'prior_stage':1,'alpha':1,'passes':2,'positions_per_stage_per_pass':24,
                            'neighbors':12,'global_candidates':4,'batch_size':4,'min_improvement':1e-5,
                            'target':'original natural audio PPG','mode':'offline whole utterance',
                            'candidate_distance':'latent code vector squared Euclidean','higher_stage_reencoding':False},
                'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'source_sha256':sha256_file(Path(__file__)), 'search_source_sha256':sha256_file(Path(__file__).with_name('local_ppg_search.py')),
                'codebooks_sha256':baseline['codebooks_sha256'],'ae_sha256':rvq_config['ae_sha256'],
                'baseline_manifest_sha256':sha256_file(args.baseline_dir/'manifest.json'),
                'transition_sha256':sha256_file(args.output_dir/'transitions.npz'),
                'generator_sha256':generator_hash,'ppg_sha256':extractor.checkpoint_sha256,
                'asr_model':metrics.ASR_MODEL,'asr_revision':metrics.ASR_REVISION,
                'asr_settings':original_manifest['asr_settings'],'audio_metric_reference':'original trajectory synthesis',
                'cer_reference':'JSUT gold text','audio_records':baseline['audio_records'],'records':[]}
    with (args.baseline_dir/'audio_metrics.csv').open() as stream: baseline_rows = list(csv.DictReader(stream))
    numeric = [k for k in baseline_rows[0] if k not in ('condition','utterance_id','audio_sha256','reference_text','asr_text','normalized_reference','normalized_hypothesis')]
    for row in baseline_rows:
        for key in numeric: row[key] = json.loads(row[key]) if row[key] else None
    if baseline['asr_model'] != metrics.ASR_MODEL or baseline['asr_revision'] != metrics.ASR_REVISION or baseline['asr_settings'] != original_manifest['asr_settings']:
        raise ValueError('ASR settings changed')
    audio_rows = [r for r in baseline_rows if r['condition'] in ('original','original_ema','rvq8')]
    summaries = []; trajectory_totals = {'before':[]}
    source_audio = json.loads((Path(rvq_config['reference_dir'])/'audio_comparison/manifest.json').read_text())
    for record in baseline['audio_records']:
        folder = record['directory']; relative = folder+'.npz'
        if relative not in splits['test']: raise ValueError('Expected test utterance')
        directory = args.output_dir/folder; directory.mkdir(parents=True)
        for name in ('original','original_ema','rvq8'):
            path = args.baseline_dir/folder/f'{name}.wav'
            expected = next(r['audio_sha256'] for r in audio_rows if r['utterance_id']==record['utterance_id'] and r['condition']==name)
            if sha256_file(path) != expected: raise ValueError('Baseline waveform changed')
            shutil.copyfile(path,directory/('before.wav' if name=='rvq8' else f'{name}.wav'))
        initial = np.load(args.baseline_dir/'tokens'/relative)['tokens']
        test_index = splits['test'].index(relative)
        if not np.array_equal(initial[:,0],sequences['test'][test_index]): raise ValueError('First-stage tokens differ')
        fixed = dict(np.load(Path(rvq_config['reference_dir'])/'audio_comparison'/folder/'conditions.npz'))
        gain = next(r['synthesis_common_gain'] for r in source_audio['records'] if r['utterance_id']==record['utterance_id'])
        evaluator = RVQEvaluator(model,mean,std,generator,extractor,fixed,gain,args.device)
        before = evaluator.synthesize(initial[None])[0,0].cpu().numpy()
        saved_before = metrics.read_audio(directory/'before.wav')
        reproduction_error = float(np.max(abs(before-saved_before)))
        if before.shape != saved_before.shape or reproduction_error > 3/32768: raise ValueError('Baseline synthesis mismatch')
        natural = metrics.read_audio(directory/'original.wav')
        target = extractor.extract(torch.from_numpy(natural[None,None]))
        reference = metrics.read_audio(directory/'original_ema.wav'); reference_features = metrics.world_features(reference)
        original_ema = checked_feature(config,relative)['ema']
        trajectory_totals['before'].append(trajectory_statistics(original_ema,evaluator.trajectories(initial[None])[0],std))
        codebooks = model.codebooks.cpu().numpy()
        for weight in args.weights:
            tick = time.monotonic(); name = f'lambda_{weight:g}'
            with (directory/f'{name}_trace.jsonl').open('w') as stream:
                def step(item):
                    stream.write(json.dumps(item)+'\n');stream.flush()
                result = search_rvq(initial,target,evaluator,codebooks,markov['transition_probs'],weight,on_step=step)
            final_audio = evaluator.synthesize(result['tokens'][None])[0,0].cpu().numpy()
            path = directory/f'{name}.wav';sf.write(path,final_audio,16000,subtype='PCM_16')
            saved = metrics.read_audio(path);saved_ppg = extractor.extract(torch.from_numpy(saved[None,None]))
            saved_js = float(posterior_score(target,saved_ppg).mean())
            if abs(saved_js-result['final_js'])>1e-5: raise ValueError('Saved WAV PPG differs from search')
            before_stats,after_stats = [transition_statistics(z[:,0],markov['transition_probs'],markov['transition_counts']) for z in (initial,result['tokens'])]
            ema = evaluator.trajectories(result['tokens'][None])[0]
            summary = {'utterance_id':record['utterance_id'],'weight':weight,
                       'baseline_js':result['baseline_js'],'final_js':result['final_js'],
                       'baseline_nll':before_stats['transition_nll'],'final_nll':after_stats['transition_nll'],
                       'final_unseen_transition_fraction':after_stats['unseen_transition_fraction'],
                       'final_self_transition_fraction':after_stats['self_transition_fraction'],
                       'edited_per_stage':np.count_nonzero(initial!=result['tokens'],axis=0).tolist(),
                       'evaluations':result['evaluations'],'seconds':time.monotonic()-tick,
                       'baseline_audio_max_error':reproduction_error,'saved_wav_js':saved_js,
                       **dynamics(ema,std)}
            summaries.append(summary)
            trajectory_totals.setdefault(name,[]).append(trajectory_statistics(original_ema,ema,std))
            save_npz(directory/f'{name}_result.npz',initial_tokens=initial,edited_tokens=result['tokens'],
                     target_ppg=target,before_ppg=result['baseline_ppg'],after_ppg=saved_ppg,edited_ema=ema)
            text = recognizer.transcribe(saved)
            audio_rows.append({'condition':name,'utterance_id':record['utterance_id'],'audio_sha256':sha256_file(path),
                               'reference_text':record['text'],'asr_text':text,**metrics.character_errors(record['text'],text),
                               **metrics.acoustic_scores(reference,saved,reference_features)})
            save_csv(args.output_dir/'summary.csv',summaries);save_csv(args.output_dir/'audio_metrics.csv',audio_rows)
            manifest['records'] = summaries
            manifest['search_aggregates'],manifest['audio_aggregates'] = report(args.output_dir,manifest,summaries,audio_rows)
            manifest['trajectory_metrics'] = summarize_trajectories(trajectory_totals)
            manifest['elapsed_seconds'] = time.monotonic()-started
            write_json(args.output_dir/'manifest.json',manifest)
            print(f'{record["utterance_id"]} λ={weight:g}: JS {result["baseline_js"]:.5f}→{saved_js:.5f}, NLL {summary["baseline_nll"]:.4f}→{summary["final_nll"]:.4f}, CER={audio_rows[-1]["cer"]:.4f}, edits={summary["edited_per_stage"]}',flush=True)
    print(f'Report: {args.output_dir}/index.html',flush=True)


if __name__ == '__main__': main()
