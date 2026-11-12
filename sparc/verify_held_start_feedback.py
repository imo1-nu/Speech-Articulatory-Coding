"""Recheck held initialization, saved control decisions and raw-prefix WavLM observations."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from .evaluate_generated_audio import read_audio
from .feedback_features import FeatureObjective, unit
from .held_start_feedback import audit_control, held_initialization, distance_key
from .japanese_ppg import sha256_file
from .model_jsut_tokens import write_json
from .rvq_transition_search import RVQEvaluator
from .trajectory_rvq import load_rvq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=Path('outputs/jsut_held_start_wavlm_feedback'))
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args();torch.set_num_threads(4)
    manifest = json.loads((args.run_dir/'manifest.json').read_text())
    if not manifest['complete']:raise ValueError('Incomplete experiment')
    for name,expected in manifest['sources'].items():
        historical = subprocess.check_output(['git','show',f'{manifest["source_commit"]}:sparc/{name}'])
        if hashlib.sha256(historical).hexdigest()!=expected:raise ValueError('Execution source differs from recorded commit')
    for path,expected in manifest['dependencies'].items():
        if sha256_file(Path(path))!=expected:raise ValueError('Input dependency changed')
    model,mean,std = load_rvq(Path('outputs/jsut_rvq/model'),8,args.device)
    kind = manifest['settings'].get('objective','wavlm');key = distance_key(kind)
    error_key = 'maximum_'+key+'_error'
    obj = FeatureObjective(kind,args.device)
    if obj.metadata!=manifest['feature']:raise ValueError('Feature model changed')
    verification = {'utterances':len(manifest['inputs']),'waveforms_checked':0,'control_decisions_checked':0,
                    'observer_prefixes_reextracted':0,'accepted_edits_replayed':0,
                    'maximum_full_waveform_feature_error':0.,'maximum_observed_feature_error':0.,
                    'maximum_feedback_bias_error':0.,'maximum_token_decode_error':0.,
                    error_key:0.,'initial_trajectory_strictly_constant':True,
                    'initial_tokens_reencoded_from_first_position_only':True,
                    'historical_execution_sources_match_commit':True,'fixed_phonation_conditions_unchanged':True}
    for source in manifest['inputs']:
        directory = args.run_dir/source['directory']
        initial = dict(np.load(directory/'initialization.npz'))
        held,codes = held_initialization(initial['first_position'],len(initial['held_ema']),model,mean,std,args.device)
        np.testing.assert_array_equal(held,initial['held_ema']);np.testing.assert_array_equal(codes,initial['initial_tokens'])
        np.testing.assert_array_equal(initial['first_position'],initial['original_ema'][0])
        config = json.loads(Path('outputs/jsut_rvq/model/config.json').read_text())
        fixed = np.load(Path(config['reference_dir'])/'audio_comparison'/source['directory']/'conditions.npz')
        for key in ('pitch','loudness','spk_emb'):np.testing.assert_array_equal(initial[key],fixed[key])
        target = np.load(directory/'target.npz')['features']
        natural = obj.extract(read_audio(directory/'original.wav')[None],fresh=True)[0]
        np.testing.assert_array_equal(target,natural)
        for row in [r for r in manifest['records'] if r['utterance_id']==source['utterance_id']]:
            path = args.run_dir/row['audio']
            if sha256_file(path)!=row['audio_sha256']:raise ValueError('Saved audio changed')
            wave = read_audio(path);values = obj.extract(wave[None],fresh=True)[0]
            n = min(len(target),len(values));distance = float(obj.distances(target[:n],values[:n]).mean())
            verification[error_key] = max(verification[error_key],abs(distance-row[key]))
            verification['waveforms_checked']+=1
            if row['condition'] not in ('unedited','no_feedback','feedback'):continue
            folder = path.parent;result = dict(np.load(folder/'result.npz'))
            if sha256_file(folder/'result.npz')!=row['result_sha256'] or sha256_file(folder/'trace.jsonl')!=row['trace_sha256']:
                raise ValueError('Saved audit data changed')
            np.testing.assert_array_equal(result['initial_tokens'],codes)
            feature_error = float(np.max(np.abs(values-result[kind+'_features'])))
            verification['maximum_full_waveform_feature_error'] = max(verification['maximum_full_waveform_feature_error'],feature_error)
            steps = [json.loads(line) for line in (folder/'trace.jsonl').read_text().splitlines()]
            audit = audit_control({**result,'tokens':result['edited_tokens'],'audio':wave,'trace':steps,
                                   'terminal_observation':{'used_for_control':False}},codes,row['condition'],kind)
            verification['control_decisions_checked']+=audit['decisions_checked']
            verification['accepted_edits_replayed']+=audit['accepted_edits_replayed']
            # Independently reconstruct the delayed feedback update from freshly extracted raw prefixes.
            bias = np.zeros(obj.dim);observed = np.empty((0,obj.dim),dtype=np.float32)
            for index,step in enumerate(steps):
                count = step['observer_input_samples']
                if count:
                    prefix = obj.extract(wave[None,:count],fresh=True)[0]
                    start,stop = len(observed),len(prefix)
                    if stop>start:
                        if kind=='ppg':
                            actual = prefix[start:stop].mean(0);forecast = result['prediction_features'][start:stop].mean(0)
                            delta = np.clip(np.log(actual+.001)-np.log(forecast+.001),-1,1)
                        else:
                            delta = (unit(prefix[start:stop])-unit(result['prediction_features'][start:stop])).mean(0)
                            delta /= max(1,float(np.linalg.norm(delta)))
                        rate = 1-.75**((stop-start)/(.1*obj.rate))
                        if row['condition']=='feedback':
                            bias = (1-rate)*bias+rate*delta
                            if kind=='ppg':bias = bias.clip(-1,1)
                            else:bias /= max(1,float(np.linalg.norm(bias)))
                        observed = np.concatenate((observed,prefix[start:stop]))
                    verification['observer_prefixes_reextracted']+=1
                error = float(np.max(np.abs(bias-result['biases'][index])))
                verification['maximum_feedback_bias_error'] = max(verification['maximum_feedback_bias_error'],error)
            if len(observed):
                error = float(np.max(np.abs(observed-result['control_observed_features'])))
                verification['maximum_observed_feature_error'] = max(verification['maximum_observed_feature_error'],error)
            decoder = RVQEvaluator(model,mean,std,None,None,{},1,args.device)
            decoded = decoder.trajectories(result['edited_tokens'][None])[0]
            error = float(np.max(np.abs(decoded-result['edited_ema'])))
            verification['maximum_token_decode_error'] = max(verification['maximum_token_decode_error'],error)
        print(f'Verified {source["utterance_id"]}',flush=True)
    for key in ('maximum_full_waveform_feature_error','maximum_observed_feature_error','maximum_feedback_bias_error',
                'maximum_token_decode_error',error_key):
        if verification[key]>1e-5:raise ValueError(f'Independent verification failed: {key}')
    verification['verification_source_sha256'] = sha256_file(Path(__file__))
    write_json(args.run_dir/'verification.json',verification)
    print(json.dumps(verification,indent=2),flush=True)


if __name__ == '__main__':main()
