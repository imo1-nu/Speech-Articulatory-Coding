"""Simulate irrevocable chunk emission with finite-horizon token search and delayed PPG feedback."""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import time

import numpy as np
import soundfile as sf
import torch

from . import local_ppg_search as local
from .compare_jsut_audio import audio_player
from .japanese_ppg import JapanesePPGExtractor, frame_times, js_divergence, load_audio, sha256_file
from .model_jsut_tokens import configure_plots, save_npz, write_json


MODES = ('unedited', 'no_feedback', 'feedback')
LABELS = {'unedited': '逐次・無編集', 'no_feedback': '逐次・PPG補正なし', 'feedback': '逐次・PPG補正あり'}
SAMPLES_PER_TOKEN = 320
PPG_PER_TOKEN = 2


def pcm16(audio):
    return (np.floor(np.asarray(audio)*32768).clip(-32768, 32767)/32768).astype(np.float32)


def audio_hash(audio):
    return hashlib.sha256(np.asarray(audio, dtype='<f4').tobytes()).hexdigest()


def stitch_prediction(predicted, emitted, continuation, crossfade_samples):
    """Keep emitted samples byte-exact, blending only the still-unemitted future."""
    output = np.asarray(predicted, dtype=np.float32).copy()
    start = len(emitted)
    if output.ndim != 2 or start >= output.shape[1]:
        raise ValueError('Prediction must extend the emitted prefix')
    output[:, :start] = emitted
    count = min(crossfade_samples, len(continuation), output.shape[1]-start)
    if count:
        weight = np.linspace(0, 1, count, dtype=np.float32)
        output[:, start:start+count] = (continuation[:count]*(1-weight)
                                      + output[:, start:start+count]*weight)
    output = pcm16(output)
    if not np.array_equal(output[:, :start], np.broadcast_to(emitted, (len(output), start))):
        raise ValueError('An emitted sample was changed')
    return output


def corrected_posterior(posterior, bias, gain):
    if not gain or not np.any(bias):
        return posterior
    logits = np.log(np.maximum(posterior.astype(np.float64), 1e-12)) + gain*bias
    logits -= logits.max(axis=-1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=-1, keepdims=True)


class DelayedObserver:
    """Freeze PPG estimates at a fixed delay; calibrate from observed-versus-forecast phone mass."""
    def __init__(self, lag_frames=10, rate=.25, clip=1.0):
        self.lag_frames, self.rate, self.clip = lag_frames, rate, clip
        self.observed = np.empty((0, 35), dtype=np.float32)
        self.bias = np.zeros(35, dtype=np.float64)

    def update(self, prefix_ppg, committed_prediction, *, feedback, final=False):
        ready = len(prefix_ppg) if final else max(0, len(prefix_ppg)-self.lag_frames)
        start = len(self.observed)
        if ready < start or ready > len(committed_prediction):
            raise ValueError('Observation availability is inconsistent')
        innovation = np.zeros(35)
        if ready > start:
            actual = prefix_ppg[start:ready]
            forecast = committed_prediction[start:ready]
            innovation = np.log(actual.mean(0)+.001)-np.log(forecast.mean(0)+.001)
            innovation = np.clip(innovation, -self.clip, self.clip)
            if feedback:
                self.bias = ((1-self.rate)*self.bias + self.rate*innovation).clip(-self.clip, self.clip)
            self.observed = np.concatenate((self.observed, actual))
        return {'observed_from_frame': start, 'observed_until_frame': ready,
                'innovation_norm': float(np.linalg.norm(innovation)),
                'bias_norm': float(np.linalg.norm(self.bias)), 'eos_flush': final}


def local_transition_nll(tokens, start, probabilities):
    # Include the fixed-prefix -> first-editable edge, excluding earlier constant edges.
    left = max(0, start-1)
    return local.transition_nll(tokens[left:], probabilities)


def planning_loss(ppg, target, tokens, start, probabilities, weight, bias, gain):
    stop = min(len(ppg), len(target))
    begin = start*PPG_PER_TOKEN
    if begin >= stop:
        raise ValueError('No future frames to score')
    distances = js_divergence(target[begin:stop], corrected_posterior(ppg[begin:stop], bias, gain))
    js = float(distances.mean())
    nll = local_transition_nll(tokens, start, probabilities)
    return js+weight*nll, js, nll, distances


def plan_step(tokens, start, target, evaluate, normalized, allowed, probabilities, bias, *,
              optimize=True, weight=.03, gain=.5, passes=2, positions=5, batch_size=4,
              min_improvement=1e-5, include_current=False):
    tokens = tokens.copy()
    audio, ppg = evaluate(tokens[None])
    audio, ppg = audio[0], ppg[0]
    score, js, nll, distances = planning_loss(ppg, target, tokens, start, probabilities, weight, bias, gain)
    initial_score, initial_js, initial_nll = score, js, nll
    evaluations, edits = 1, []
    for pass_index in range(passes if optimize else 0):
        priorities = []
        for index in range(start, len(tokens)):
            relative = (index-start)*2
            priorities.append(distances[max(0, relative-5):min(len(distances), relative+9)].mean())
        chosen = np.argsort(-np.asarray(priorities), kind='stable')[:positions]+start
        if include_current:
            chosen = np.asarray([start, *[int(i) for i in chosen if i != start]])[:positions]
        accepted = 0
        for position in chosen:
            old = int(tokens[position])
            codes = local.candidate_codes(old, normalized, allowed, 12, 4)
            best_score, best_code = score, old
            for offset in range(0, len(codes), batch_size):
                group = codes[offset:offset+batch_size]
                proposals = np.repeat(tokens[None], len(group), axis=0)
                proposals[:, position] = group
                _, predicted = evaluate(proposals)
                evaluations += len(group)
                for code, trial_tokens, trial_ppg in zip(group, proposals, predicted):
                    trial_score, _, _, _ = planning_loss(trial_ppg, target, trial_tokens, start, probabilities, weight, bias, gain)
                    if trial_score < best_score-min_improvement:
                        best_score, best_code = trial_score, int(code)
            previous = score
            if best_code != old:
                proposal = tokens.copy(); proposal[position] = best_code
                verified_audio, verified_ppg = evaluate(proposal[None]); evaluations += 1
                verified = planning_loss(verified_ppg[0], target, proposal, start, probabilities, weight, bias, gain)
                if verified[0] < score-min_improvement:
                    tokens, audio, ppg = proposal, verified_audio[0], verified_ppg[0]
                    score, js, nll, distances = verified
                    accepted += 1
            edits.append({'pass': pass_index+1, 'position': int(position), 'old_code': old,
                          'new_code': int(tokens[position]), 'accepted': int(tokens[position]) != old,
                          'previous_objective': previous, 'objective': score, 'planning_js': js,
                          'planning_transition_nll': nll, 'candidate_count': len(codes)})
        if not accepted:
            break
    return tokens, audio, ppg, {'initial_objective': initial_score, 'final_objective': score,
                               'initial_planning_js': initial_js, 'final_planning_js': js,
                               'initial_planning_nll': initial_nll, 'final_planning_nll': nll,
                               'evaluations': evaluations, 'edits': edits}


class PrefixEvaluator:
    def __init__(self, generator, extractor, normalized, conditions, gain, device, model, mean, std):
        self.generator, self.extractor = generator, extractor
        self.windows, self.conditions, self.gain, self.device = normalized, conditions, gain, device
        self.model, self.mean, self.std = model, mean, std

    def at(self, end, emitted, continuation, crossfade_samples):
        conditions = {key: (value[:end] if key in ('pitch', 'loudness') else value)
                      for key, value in self.conditions.items()}
        base = local.TokenEvaluator(self.generator, self.extractor, self.windows, conditions, self.gain,
                                    self.device, vq_model=self.model, mean=self.mean, std=self.std)
        def evaluate(sequences):
            if sequences.shape[1] != end:
                raise ValueError('Candidate reaches beyond planning horizon')
            raw = base.synthesize(sequences)[:, 0].cpu().numpy()
            audio = stitch_prediction(raw, emitted, continuation, crossfade_samples)
            with torch.inference_mode():
                tensor = torch.from_numpy(audio[:, None]).to(self.device)
                ppg = self.extractor.model(self.extractor.frontend(tensor)).softmax(1).transpose(1, 2).cpu().numpy()
            return audio, ppg
        return evaluate

    def observe(self, audio):
        return self.extractor.extract(torch.from_numpy(audio[None, None]))


def sequential_search(initial, target, normalized, allowed, probabilities, evaluator, *,
                      horizon_tokens, commit_tokens=5, lag_frames=10, mode='no_feedback',
                      crossfade_samples=160, positions=5, passes=2, weight=.03,
                      feedback_gain=.5, feedback_rate=.25, feedback_clip=1.0, on_step=None):
    if mode not in MODES or horizon_tokens < commit_tokens or commit_tokens < 1 or lag_frames < 0:
        raise ValueError('Invalid sequential configuration')
    tokens = initial.copy()
    emitted = np.empty(0, dtype=np.float32)
    continuation = np.empty(0, dtype=np.float32)
    forecasts = np.empty((0, 35), dtype=np.float32)
    observer = DelayedObserver(lag_frames, feedback_rate, feedback_clip)
    trace, biases = [], []
    for start in range(0, len(tokens), commit_tokens):
        end = min(len(tokens), start+horizon_tokens)
        commit = min(len(tokens), start+commit_tokens)
        prefix_tokens = tokens[:start].copy()
        prefix_audio = emitted.copy()
        evaluate = evaluator.at(end, emitted, continuation, crossfade_samples)
        bias_before = observer.bias.copy()
        planned, audio, ppg, stats = plan_step(tokens[:end], start, target, evaluate,
                normalized, allowed, probabilities, bias_before, optimize=mode != 'unedited',
                weight=weight, gain=feedback_gain if mode == 'feedback' else 0,
                positions=positions, passes=passes)
        tokens[:end] = planned
        if not np.array_equal(tokens[:start], prefix_tokens):
            raise ValueError('An emitted token was edited')
        sample_start, sample_commit = start*SAMPLES_PER_TOKEN, commit*SAMPLES_PER_TOKEN
        emitted = np.concatenate((emitted, audio[sample_start:sample_commit]))
        if not np.array_equal(emitted[:len(prefix_audio)], prefix_audio):
            raise ValueError('An emitted waveform prefix was rewritten')
        continuation = audio[sample_commit:].copy()
        forecasts = np.concatenate((forecasts, ppg[start*2:commit*2]))
        observation = observer.update(evaluator.observe(emitted), forecasts,
                    feedback=mode == 'feedback', final=commit == len(tokens))
        if observation['observed_until_frame'] > commit*2-(0 if commit == len(tokens) else lag_frames):
            raise ValueError('Feedback used unavailable future observations')
        item = {'start_token': start, 'planning_end_token': end, 'committed_until_token': commit,
                'prefix_sha256_before': audio_hash(prefix_audio),
                'retained_prefix_sha256_after': audio_hash(emitted[:len(prefix_audio)]),
                'chunk_sha256': audio_hash(emitted[sample_start:sample_commit]),
                'bias_before_norm': float(np.linalg.norm(bias_before)),
                **observation, **stats}
        trace.append(item); biases.append(bias_before)
        if on_step:
            on_step(item)
    return {'tokens': tokens, 'audio': emitted, 'prediction_ppg': forecasts,
            'observed_ppg': observer.observed, 'biases': np.asarray(biases), 'trace': trace}


def boundary_statistics(audio, commit_samples=1600):
    boundaries = np.arange(commit_samples, len(audio), commit_samples)
    jumps = np.abs(audio[boundaries]-audio[boundaries-1])
    return {'boundary_mean_abs_jump': float(jumps.mean()) if len(jumps) else 0,
            'boundary_max_abs_jump': float(jumps.max()) if len(jumps) else 0}


def plot_utterance(path, target, before, after, online, trace):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(13, 11))
    for axis, ppg, label in zip(axes[:3], [target, after, online],
                               ['Target: original audio', 'Actually emitted audio: full-utterance PPG', 'Frozen fixed-lag observations']):
        axis.imshow(ppg.T, aspect='auto', vmin=0, vmax=1, cmap='viridis', interpolation='nearest',
                    extent=(0, len(ppg)/100, 34.5, -.5))
        axis.set_yticks(range(35), labels=local.PHONEMES, fontsize=6); axis.set_title(label)
    for ppg, label in [(before, 'Before editing: full synthesis'), (after, 'Emitted output')]:
        d = local.posterior_score(target, ppg)
        axes[3].plot(frame_times(len(d)), d, label=label, linewidth=1)
    for step in trace[:-1]:
        axes[3].axvline(step['committed_until_token']/50, color='black', alpha=.08)
    axes[3].set(xlabel='Time (s)', ylabel='Raw JS (nats)', ylim=(0, np.log(2))); axes[3].legend()
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def write_report(output, rows, offline, conditions, method_html=None):
    import matplotlib.pyplot as plt
    aggregates = []
    for condition in conditions:
        group = [r for r in rows if r['condition'] == condition]
        if not group:
            continue
        aggregate = {'condition': condition, 'horizon_ms': group[0]['horizon_ms'], 'mode': group[0]['mode'],
                     'utterances': len(group), 'edited_tokens': sum(r['edited_tokens'] for r in group),
                     'evaluations': sum(r['evaluations'] for r in group), 'seconds': sum(r['seconds'] for r in group)}
        for key, weights in [('mean_js', 'ppg_frames'), ('fixed_lag_js', 'ppg_frames'),
                             ('transition_nll', 'edges'), ('unseen_transition_fraction', 'edges'),
                             ('boundary_mean_abs_jump', 'boundaries')]:
            aggregate[key] = float(np.average([r[key] for r in group], weights=[r[weights] for r in group]))
        aggregates.append(aggregate)
    save_csv(output/'summary.csv', rows); save_csv(output/'aggregates.csv', aggregates)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for mode in MODES:
        group = [r for r in aggregates if r['mode'] == mode]
        axes[0].plot([r['horizon_ms'] for r in group], [r['mean_js'] for r in group], marker='o', label=mode)
        axes[1].plot([r['horizon_ms'] for r in group], [r['transition_nll'] for r in group], marker='o', label=mode)
    reference = float(np.average([r['final_js'] for r in offline], weights=[r['common_ppg_frames'] for r in offline]))
    axes[0].axhline(reference, linestyle='--', color='gray', label='Offline reference (different search budget)')
    for axis in axes: axis.set_xlabel('Planning horizon (ms)'); axis.legend(fontsize=7)
    axes[0].set_ylabel('Final raw mean JS (nats)'); axes[1].set_ylabel('Transition NLL (nats/edge)')
    fig.tight_layout(); fig.savefig(output/'comparison.png', dpi=150); plt.close(fig)
    table = ''.join(f'<tr><td>{r["horizon_ms"]}</td><td>{LABELS[r["mode"]]}</td><td>{r["mean_js"]:.5f}</td>'
                    f'<td>{r["transition_nll"]:.4f}</td><td>{100*r["unseen_transition_fraction"]:.2f}%</td>'
                    f'<td>{r["edited_tokens"]}</td><td>{r["evaluations"]}</td></tr>' for r in aggregates)
    cards = []
    for ref in offline:
        utterance = ref['utterance_id']; players = ''
        for name, label in [('original', '元音声'), ('before', '編集前・一括合成'), ('offline', '一括局所探索・参考')]:
            players += f'<div><h3>{label}</h3>{audio_player("references/"+utterance+"/"+name+".wav")}</div>'
        for row in rows:
            if row['utterance_id'] != utterance:
                continue
            players += (f'<div><h3>{row["horizon_ms"]} ms・{LABELS[row["mode"]]}</h3>{audio_player(row["directory"]+"/output.wav")}'
                        f'<p>JS {row["mean_js"]:.5f} / 遷移NLL {row["transition_nll"]:.4f}</p>'
                        f'<a href="{row["directory"]}/ppg.png">PPG図</a> / <a href="{row["directory"]}/trace.jsonl">履歴</a></div>')
        cards.append(f'<section><h2>{html.escape(utterance)}</h2><p>{html.escape(ref["text"])}</p><div class="players">{players}</div></section>')
    content = '''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>逐次PPG制御の比較</title>
<style>body{font-family:system-ui;max-width:1300px;margin:30px auto;padding:0 18px}.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:18px}section{border:1px solid #ccc;padding:18px;margin:24px 0}audio,img{width:100%}td,th{padding:9px;border-bottom:1px solid #ddd}table{border-collapse:collapse}h3{font-size:1rem}</style>
<h1>確定幅100 ms・探索範囲200 / 400 / 800 ms</h1><p>λ=0.03。音声は100 msずつ確定し、過去の音声・トークンは変更しません。
区間の先頭10 msを前回の未出力予測とクロスフェードします。各ステップで5位置×2パス・各16候補を評価します。
探索範囲の外の音声・PPGを候補の予測に使いません。目標PPGと初期トークン・F0等は事前抽出済みで、入力側までオンライン化した実験ではありません。</p>
<p>PPG補正ありでは、出力済み音声から100 ms遅延でPPGを確定し、出力前の予測との差から音素別log確率バイアスを更新します。
次の探索でそのバイアスを予測PPGに適用します（更新率0.25・上限±1・適用ゲイン0.5）。
補正なしは同じ予測器・同じ確定済み波形の文脈を使い、バイアスをゼロに固定します。FBの比較は、この予測補正の有無です。
既存の非因果PPGモデルを固定遅延で近似的に使っており、遅延後の値も発話全体のPPGと同じとは限りません。末尾100 msはEOSで確定します。</p>
<p>主指標は、実際に出力・保存した音声全体から再抽出した未補正PPGのJSです。
同じ合成モデルを予測と出力に使うシミュレーションで、外部の音響変動は加えていません。
一括探索の参考値は総探索予算が異なり、性能の上限ではありません。</p><p><a href="summary.csv">発話別CSV</a> / <a href="aggregates.csv">集計CSV</a> / <a href="manifest.json">設定</a></p>
<table><tr><th>範囲(ms)</th><th>条件</th><th>平均JS ↓</th><th>遷移NLL ↓</th><th>未観測遷移</th><th>編集数</th><th>評価回数</th></tr>'''
    content += table+'</table><img src="comparison.png" alt="逐次処理の比較">'+''.join(cards)
    if method_html is not None:
        start = content.index('<h1>')
        end = content.index('<p>主指標', start)
        content = content[:start] + method_html + content[end:]
    content += "<script>document.querySelectorAll('audio').forEach(p=>p.onplay=()=>document.querySelectorAll('audio').forEach(q=>{if(q!==p)q.pause()}));</script></html>"
    (output/'index.html').write_text(content, encoding='utf-8')
    return aggregates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('outputs/jsut_window_sweep/window_040ms'))
    parser.add_argument('--ppg-dir', type=Path, default=Path('outputs/jsut_ppg_40ms128'))
    parser.add_argument('--offline-dir', type=Path, default=Path('outputs/jsut_ppg_soft_transition/condition_02_lambda_0.03'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/jsut_ppg_sequential'))
    parser.add_argument('--horizons', nargs='+', type=int, default=[200, 400, 800])
    parser.add_argument('--utterances', nargs='+', default=local.DEFAULT_IDS)
    parser.add_argument('--device', default='cuda'); parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    if (len(set(args.horizons)) != len(args.horizons) or any(h < 100 or h % 20 for h in args.horizons)
            or len(set(args.utterances)) != len(args.utterances) or args.threads < 1):
        parser.error('Invalid horizons, utterances, or threads')
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error('Output directory must be new or empty')
    torch.set_num_threads(args.threads)
    args.transition_alpha = 1.0
    source, digest, normalized, windows, counts, selected, model, mean, std, markov = local.prepare(args)
    offline_manifest = json.loads((args.offline_dir/'manifest.json').read_text())
    if (offline_manifest['vq_checkpoint_sha256'] != digest
            or offline_manifest['settings']['transition_weight'] != .03
            or offline_manifest['ppg_checkpoint_sha256'] != source['ppg_model']['checkpoint_sha256']):
        raise ValueError('Offline reference model mismatch')
    offline = [next(r for r in offline_manifest['records'] if r['utterance_id'] == u) for u in args.utterances]
    generator, generator_digest = local.load_generator(args.device)
    extractor = JapanesePPGExtractor(args.device)
    if extractor.checkpoint_sha256 != source['ppg_model']['checkpoint_sha256']:
        raise ValueError('PPG checkpoint mismatch')
    configure_plots(); args.output_dir.mkdir(parents=True, exist_ok=True)
    save_npz(args.output_dir/'transitions.npz', **markov, alpha=np.float64(1), vq_checkpoint_sha256=np.str_(digest))
    references = []
    for ref, (record, _, _, _, _) in zip(offline, selected):
        folder = args.output_dir/'references'/ref['utterance_id']; folder.mkdir(parents=True)
        for name, path in [('original', args.ppg_dir/record['variants']['original']['audio']),
                           ('before', args.offline_dir/ref['directory']/'before.wav'),
                           ('offline', args.offline_dir/ref['directory']/'after.wav')]:
            shutil.copyfile(path, folder/f'{name}.wav')
            references.append({'source': str(path.resolve()), 'sha256': sha256_file(path)})
    settings = {'commit_ms': 100, 'observation_lag_ms': 100, 'crossfade_ms': 10,
                'transition_weight': .03, 'transition_alpha': 1,
                'positions_per_pass_per_step': 5, 'passes_per_step': 2, 'neighbors': 12, 'global_candidates': 4,
                'feedback_update_rate': .25, 'feedback_log_bias_clip': 1., 'feedback_gain': .5,
                'feedback_definition': 'delayed observed vs pre-emission forecast phone-mass log-ratio calibration',
                'objective': 'mean JS over editable horizon + .03 * mean transition NLL over horizon including incoming edge',
                'target_available_offline': True, 'initial_features_available_offline': True,
                'plant': 'same deterministic SPARC synthesizer as predictor, immutable chunk emission',
                'future_audio_beyond_horizon_used': False, 'offline_search_budget_differs': True}
    manifest = {'arguments': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                'settings': settings, 'vq_checkpoint_sha256': digest, 'generator_checkpoint_sha256': generator_digest,
                'ppg_checkpoint_sha256': extractor.checkpoint_sha256,
                'offline_manifest_sha256': sha256_file(args.offline_dir/'manifest.json'), 'references': references,
                'records': [], 'aggregates': []}
    rows, conditions = [], []
    for horizon in args.horizons:
        for mode in MODES:
            condition = f'horizon_{horizon:03d}ms_{mode}'; conditions.append(condition)
            for record, fixed, initial, target, gain in selected:
                directory = args.output_dir/condition/Path(record['feature_path']).with_suffix('')
                directory.mkdir(parents=True)
                evaluator = PrefixEvaluator(generator, extractor, normalized, fixed, gain, args.device, model, mean, std)
                tick = time.monotonic()
                with (directory/'trace.jsonl').open('w') as log:
                    def on_step(item):
                        log.write(json.dumps(item)+'\n'); log.flush()
                        print(f'{condition} {record["utterance_id"]} commit={item["committed_until_token"]*20}ms '
                              f'JS(plan)={item["final_planning_js"]:.5f} evals={item["evaluations"]}', flush=True)
                    result = sequential_search(initial, target, normalized, np.flatnonzero(counts), markov['transition_probs'],
                                               evaluator, horizon_tokens=horizon//20, mode=mode, on_step=on_step)
                waveform = result['audio']; sf.write(directory/'output.wav', waveform, 16000, subtype='PCM_16')
                saved, sr = sf.read(directory/'output.wav', dtype='float32')
                if sr != 16000 or not np.array_equal(saved, waveform):
                    raise ValueError('Saved WAV differs from irrevocably emitted samples')
                final_ppg = extractor.extract(load_audio(directory/'output.wav')[0])
                distance = local.posterior_score(target, final_ppg)
                online_distance = local.posterior_score(target, result['observed_ppg'])
                n = len(distance)
                stats = local.transition_statistics(result['tokens'], markov['transition_probs'], markov['transition_counts'])
                row = {'condition': condition, 'horizon_ms': horizon, 'mode': mode, 'utterance_id': record['utterance_id'],
                       'directory': directory.relative_to(args.output_dir).as_posix(),
                       'mean_js': float(distance.mean()), 'fixed_lag_js': float(online_distance.mean()),
                       'argmax_agreement': float((target[:n].argmax(1)==final_ppg[:n].argmax(1)).mean()),
                       'edited_tokens': int(np.count_nonzero(initial != result['tokens'])), 'token_count': len(initial),
                       'ppg_frames': n, 'edges': len(initial)-1, 'boundaries': (len(initial)-1)//5,
                       'evaluations': sum(x['evaluations'] for x in result['trace']), 'seconds': time.monotonic()-tick,
                       'feedback_updates': sum(x['observed_until_frame'] > x['observed_from_frame'] and not x['eos_flush']
                                               for x in result['trace']) if mode == 'feedback' else 0,
                       **stats, **boundary_statistics(waveform)}
                before = np.load(args.ppg_dir/record['variants']['reconstructed_ema']['ppg'])['ppg']
                save_npz(directory/'result.npz', initial_tokens=initial, edited_tokens=result['tokens'],
                         target_ppg=target, final_ppg=final_ppg, fixed_lag_ppg=result['observed_ppg'],
                         committed_prediction_ppg=result['prediction_ppg'], feedback_biases=result['biases'],
                         time=frame_times(len(final_ppg)), js=distance, fixed_lag_js=online_distance,
                         phonemes=np.asarray(local.PHONEMES), pitch=fixed['pitch'], loudness=fixed['loudness'], spk_emb=fixed['spk_emb'])
                plot_utterance(directory/'ppg.png', target, before, final_ppg, result['observed_ppg'], result['trace'])
                write_json(directory/'summary.json', row); rows.append(row)
                manifest['records'] = rows
                manifest['aggregates'] = write_report(args.output_dir, rows, offline, conditions)
                write_json(args.output_dir/'manifest.json', manifest)
                print(f'{condition} {record["utterance_id"]}: final JS={row["mean_js"]:.6f} '
                      f'NLL={row["transition_nll"]:.4f} edits={row["edited_tokens"]}', flush=True)
    print(f'Report: {args.output_dir}/index.html', flush=True)
    return 0


if __name__ == '__main__':
    main()
