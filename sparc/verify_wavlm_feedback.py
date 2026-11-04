"""Audit completed feature-feedback runs and independently re-extract saved audio features."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from . import evaluate_generated_audio as metrics
from .feedback_features import FeatureObjective,FeatureObserver
from .japanese_ppg import sha256_file
from .local_ppg_search import transition_nll
from .model_jsut_tokens import write_json
from .sequential_ppg_search import audio_hash,save_csv
from .trajectory_rvq import load_rvq
from .trajectory_vq import overlap_average


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=Path('outputs/jsut_wavlm_feedback'))
    parser.add_argument('--baseline-dir',type=Path,default=Path('outputs/jsut_rvq/comparison_verified'))
    parser.add_argument('--model-dir',type=Path,default=Path('outputs/jsut_rvq/model'))
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();root=args.run_dir
    torch.set_num_threads(4)
    m=json.loads((root/'manifest.json').read_text())
    if not m.get('complete'):raise ValueError('Run is incomplete')
    for name,digest in m['sources'].items():
        if sha256_file(Path(__file__).with_name(name))!=digest:raise ValueError(f'Source changed: {name}')
    for path,digest in m['dependencies'].items():
        if sha256_file(Path(path))!=digest:raise ValueError(f'Dependency changed: {path}')
    objectives={name:FeatureObjective(name,args.device) for name in ('ppg','wavlm')}
    for name,obj in objectives.items():
        if obj.metadata!=m['features'][name]:raise ValueError('Feature model changed')
    model,mean,std=load_rvq(args.model_dir,8,'cpu')
    prior=np.load(Path(m['settings']['prior_dir'])/'transitions.npz')['transition_probs']
    records=m['records'];unique={(r['condition'],r['utterance_id']) for r in records}
    if len(unique)!=len(records):raise ValueError('Duplicate completed condition')
    objective_count=2 if m['settings']['objective']=='all' else 1
    condition_count=(len(m['settings']['weights']) if m['settings']['mode'] in ('all','offline') else 0)
    condition_count+=(3 if m['settings']['mode'] in ('all','delayed') else 0)
    if len(records)!=len(m['inputs'])*objective_count*condition_count:raise ValueError('Missing conditions')
    unedited_pairs=0
    for source in m['inputs']:
        pair=[r for r in records if r['utterance_id']==source['utterance_id'] and r['control']=='unedited']
        if len(pair)==2:
            if pair[0]['audio_sha256']!=pair[1]['audio_sha256']:raise ValueError('Unedited renderer differs between objectives')
            unedited_pairs+=1
    max_replay=0.;max_feature_error=0.;decisions=0;coordinates=0;baseline_hashes={};baseline_feature_rows=[]
    for source in m['inputs']:
        folder=source['directory']
        for variant in ('original','original_ema','before'):
            old='rvq8' if variant=='before' else variant
            if sha256_file(root/folder/f'{variant}.wav')!=sha256_file(args.baseline_dir/folder/f'{old}.wav'):
                raise ValueError('Baseline audio changed')
        baseline_hashes[source['utterance_id']]=sha256_file(root/folder/'before.wav')
        targets=np.load(root/folder/'targets.npz')
        for variant in ('before','original_ema','original'):
            wave=metrics.read_audio(root/folder/f'{variant}.wav')
            row={'condition':variant,'utterance_id':source['utterance_id']}
            for name,obj in objectives.items():
                value=obj.extract(wave[None],fresh=True)[0]
                n=min(len(value),len(targets[name]))
                row['ppg_js' if name=='ppg' else 'wavlm_distance']=float(obj.distances(targets[name][:n],value[:n]).mean())
                if variant=='original':np.testing.assert_allclose(value,targets[name],atol=1e-4,rtol=0)
            baseline_feature_rows.append(row)
    for row in records:
        assert row['evaluations']==row['synthesis_candidates']
        if row.get('execution_sources') not in [m['sources'],*m.get('source_history',[])]:
            raise ValueError('Condition execution source is not recorded in source history')
        path=root/row['audio'];folder=path.parent
        if sha256_file(path)!=row['audio_sha256'] or sha256_file(folder/'result.npz')!=row['result_sha256']:
            raise ValueError('Waveform or result checksum changed')
        if sha256_file(folder/'trace.jsonl')!=row['trace_sha256']:raise ValueError('Trace checksum changed')
        source=next(r for r in m['inputs'] if r['utterance_id']==row['utterance_id'])
        result=np.load(folder/'result.npz');targets=np.load(root/source['directory']/'targets.npz')
        original=np.load(args.baseline_dir/'tokens'/f'{source["directory"]}.npz')['tokens']
        np.testing.assert_array_equal(original,result['initial_tokens'])
        tokens=result['edited_tokens'];assert tokens.shape==original.shape and tokens.dtype==np.int64
        assert tokens.min()>=0 and tokens.max()<128
        assert row['edited_per_stage']==np.count_nonzero(tokens!=original,axis=0).tolist()
        with torch.inference_mode():windows=model.decode_tokens(torch.from_numpy(tokens)).numpy().reshape(-1,2,12)
        ema=overlap_average(windows,output_length=len(tokens))*std+mean
        error=float(np.max(abs(ema-result['edited_ema'])));max_replay=max(error,max_replay)
        if error>2e-5:raise ValueError('Token replay changed the trajectory')
        assert abs(transition_nll(tokens[:,0],prior)-row['first_stage_nll'])<1e-12
        wave=metrics.read_audio(path)
        for name,obj in objectives.items():
            features=obj.extract(wave[None],fresh=True)[0]
            saved=result[name+'_features'];assert features.shape==saved.shape
            error=float(np.max(abs(features-saved)));max_feature_error=max(error,max_feature_error)
            if error>1e-4:raise ValueError('Saved features do not match saved WAV')
            np.testing.assert_array_equal(result[name+'_times'],obj.times(len(features)))
            n=min(len(features),len(targets[name]))
            distance=float(obj.distances(targets[name][:n],features[:n]).mean())
            key='ppg_js' if name=='ppg' else 'wavlm_distance'
            if abs(distance-row[key])>1e-5:raise ValueError('Saved distance does not reproduce')
        trace=[json.loads(line) for line in (folder/'trace.jsonl').read_text().splitlines()]
        replay_tokens=original.copy()
        def replay_edits(edits,start,end):
            for edit in edits:
                position,stage=edit['position'],edit['stage']-1
                assert start<=position<end and 0<=stage<8
                assert replay_tokens[position,stage]==edit['old_code']
                replay_tokens[position,stage]=edit['new_code']
        if row['mode']=='offline':
            for a,b in zip(trace,trace[1:]):assert b['objective']<=a['objective']+1e-8
            for item in trace:assert item['objective']<=item['previous_objective']+1e-8
            coordinates+=len(trace)
            replay_edits(trace,0,len(tokens))
        else:
            assert len(trace)==len(tokens)
            forecasts=result['prediction_features'];biases=result['biases'];obj=objectives[row['objective']]
            observer=FeatureObserver(obj)
            for i,item in enumerate(trace):
                assert item['start_token']==i and item['decision_time_ms']==i*20
                cutoff=max(0,i*320-960);assert item['information_cutoff_samples']==cutoff
                assert item['observer_input_samples']<=cutoff
                if item['observer_input_samples']:
                    assert item['observer_input_samples']==cutoff
                    assert item['observer_input_sha256']==audio_hash(wave[:cutoff])
                    prefix=obj.extract(wave[None,:cutoff],fresh=True)[0]
                    observed=observer.update(prefix,forecasts,row['control']=='feedback')
                    assert observed['observed_from_frame']==item['observed_from_frame']
                    assert observed['observed_until_frame']==item['observed_until_frame']
                np.testing.assert_allclose(observer.bias,biases[i],rtol=0,atol=1e-7)
                assert item['prefix_sha256_before']==audio_hash(wave[:i*320])
                assert item['prefix_sha256_before']==item['retained_prefix_sha256_after']
                assert item['chunk_sha256']==audio_hash(wave[i*320:(i+1)*320])
                context=result['commanded_audio'][:i*320].copy();context[:cutoff]=wave[:cutoff]
                assert item['prediction_context_sha256']==audio_hash(context)
                assert item['blind_predicted_context_sha256']==audio_hash(context[cutoff:])
                assert abs(np.linalg.norm(biases[i])-item['bias_before_norm'])<1e-10
                ready=item['observed_until_frame']
                if ready:assert obj.times(ready)[-1]<cutoff/16000
                for edit in item['edits']:assert edit['objective']<=edit['previous_objective']+1e-8
                replay_edits(item['edits'],i,item['planning_end_token'])
                coordinates+=len(item['edits']);decisions+=1
            assert not row['terminal_observation']['used_for_control']
            assert row['terminal_observation']['observation_time_ms']==len(tokens)*20+60
            if row['control']!='feedback':np.testing.assert_array_equal(biases,np.zeros_like(biases))
            if row['control']=='unedited':np.testing.assert_array_equal(tokens,original)
            np.testing.assert_allclose(observer.observed,result['control_observed_features'],rtol=0,atol=1e-4)
        np.testing.assert_array_equal(replay_tokens,tokens)
    with (root/'audio_metrics.csv').open() as stream:audio_rows=list(csv.DictReader(stream))
    natural_rows=[];natural_references={}
    for row in audio_rows:
        source=next(s for s in m['inputs'] if s['utterance_id']==row['utterance_id'])
        if row['condition'] in ('before','original','original_ema'):path=root/source['directory']/f'{row["condition"]}.wav'
        else:path=root/next(r['audio'] for r in records if r['condition']==row['condition'] and r['utterance_id']==row['utterance_id'])
        assert sha256_file(path)==row['audio_sha256']
        actual=metrics.character_errors(row['reference_text'],row['asr_text'])
        assert actual['character_errors']==int(row['character_errors'])
        utterance=row['utterance_id']
        if utterance not in natural_references:
            natural=metrics.read_audio(root/source['directory']/'original.wav')
            natural_references[utterance]=(natural,metrics.world_features(natural))
        natural,ref_features=natural_references[utterance]
        natural_rows.append({'condition':row['condition'],'utterance_id':utterance,
                             'cer':actual['cer'],'character_errors':actual['character_errors'],
                             'reference_characters':actual['reference_characters'],
                             **metrics.acoustic_scores(natural,metrics.read_audio(path),ref_features)})
    natural_aggregates=metrics.aggregate(natural_rows)
    save_csv(root/'audio_metrics_natural_reference.csv',natural_rows)
    save_csv(root/'audio_aggregates_natural_reference.csv',natural_aggregates)
    comparisons=[]
    for condition in ['before','original_ema','original',*dict.fromkeys(r['condition'] for r in records)]:
        feature_rows=([r for r in baseline_feature_rows if r['condition']==condition]
                      if condition in ('before','original_ema','original') else [r for r in records if r['condition']==condition])
        audio=next(r for r in m['audio_aggregates'] if r['condition']==condition)
        comparisons.append({'condition':condition,'utterances':len(feature_rows),
                            'wavlm_distance':float(np.mean([r['wavlm_distance'] for r in feature_rows])),
                            'ppg_js':float(np.mean([r['ppg_js'] for r in feature_rows])),
                            'cer':audio['cer'],'mcd_db':audio['mcd_db'],'stoi':audio['stoi']})
    save_csv(root/'comparison_summary.csv',comparisons)
    natural_comparisons=[]
    for row in comparisons:
        audio=next(r for r in natural_aggregates if r['condition']==row['condition'])
        natural_comparisons.append({**row,'cer':audio['cer'],'mcd_db':audio['mcd_db'],'stoi':audio['stoi']})
    save_csv(root/'comparison_summary_natural_reference.csv',natural_comparisons)
    save_csv(root/'baseline_feature_metrics.csv',baseline_feature_rows)
    verification={'conditions_checked':len(records),'delayed_decisions_checked':decisions,'coordinates_checked':coordinates,
                  'candidate_evaluations':sum(r['evaluations'] for r in records),
                  'sum_condition_seconds':sum(r['seconds'] for r in records),
                  'device':torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else args.device,
                  'cuda_version':torch.version.cuda,
                  'source_and_dependency_hashes_match':True,'baseline_waveforms_unchanged':True,
                  'cutoff_input_and_prefix_hashes_match':True,'committed_audio_immutable':True,
                  'non_feedback_bias_zero':True,'independently_reextracted_saved_features':True,
                  'prefix_features_and_feedback_biases_replayed':True,
                  'condition_execution_source_history_checked':True,
                  'unedited_objective_pairs_byte_identical':unedited_pairs,
                  'max_absolute_saved_feature_error':max_feature_error,'max_absolute_cpu_token_replay_error':max_replay,
                  'verification_source_sha256':sha256_file(Path(__file__))}
    write_json(root/'verification.json',verification)
    page=(root/'index.html').read_text()
    marker='<p id="verification-summary">'
    if marker in page:page=page[:page.index(marker)]+page[page.index('</p>',page.index(marker))+4:]
    before=comparisons[0]
    insert=(marker+f'編集前の{len(m["inputs"])}発話平均：WavLM距離 {before["wavlm_distance"]:.5f}、PPG JS {before["ppg_js"]:.5f}。'
            '<a href="comparison_summary.csv">編集前を含む比較CSV</a> / <a href="verification.json">独立検証結果</a></p>')
    insert+=('<p id="natural-reference-summary">主評価のMCD/STOI/F0は元軌道合成参照。補助評価として '
             '<a href="comparison_summary_natural_reference.csv">元音声参照の比較CSV</a> / '
             '<a href="audio_aggregates_natural_reference.csv">元音声参照の全音響指標</a> も保存しました。</p>')
    marker2='<p id="natural-reference-summary">'
    if marker2 in page:page=page[:page.index(marker2)]+page[page.index('</p>',page.index(marker2))+4:]
    page=page.replace('<h2>特徴量・計算時間',insert+'<h2>特徴量・計算時間',1)
    (root/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
