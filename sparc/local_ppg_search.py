"""Greedy discrete token editing using whole-utterance Japanese PPG feedback."""

import argparse
import csv
import html
import json
import os
from pathlib import Path
import shutil
import time

from huggingface_hub import hf_hub_download
import numpy as np
import soundfile as sf
import torch

from .japanese_ppg import (
    HOP, MODEL_ID, MODEL_REVISION, PHONEMES, SOURCE_COMMIT,
    JapanesePPGExtractor, frame_times, js_divergence, sha256_file,
)
from .model_jsut_tokens import checked_feature, configure_plots, infer_utterance, load_run, save_npz, write_json
from .trajectory_vq import fit_markov, overlap_average


SPARC_REVISION = "2e6a07d4b35022d4366a2896a6293fe16aaed78a"
DEFAULT_IDS = ["BASIC5000_0053", "BASIC5000_0050", "BASIC5000_0066"]


def posterior_score(target, posterior):
    length = min(len(target), len(posterior))
    if not length:
        raise ValueError("Empty PPG comparison")
    return js_divergence(target[:length], posterior[:length])


def candidate_codes(token, decoded_normalized, allowed, neighbors, global_candidates):
    """Near decoded windows plus deterministic farthest-point coverage; no transition model."""
    allowed = np.asarray(allowed, dtype=np.int64)
    remaining = allowed[allowed != token]
    distances = ((decoded_normalized[remaining] - decoded_normalized[token]) ** 2).mean(axis=(1, 2))
    order = np.argsort(distances, kind="stable")
    chosen = list(remaining[order[:neighbors]])
    # Spread extra candidates across the remaining decoded codebook, rather than relying on IDs.
    for _ in range(global_candidates):
        pool = np.asarray([k for k in remaining if k not in chosen], dtype=np.int64)
        if not len(pool):
            break
        anchors = np.asarray([token, *chosen], dtype=np.int64)
        coverage = ((decoded_normalized[pool, None] - decoded_normalized[anchors][None]) ** 2).mean(axis=(2, 3)).min(axis=1)
        chosen.append(int(pool[np.argmax(coverage)]))
    return np.asarray(chosen, dtype=np.int64)


def priority_positions(distances, token_length, maximum):
    # Each 40 ms window starts at token_index * 20 ms; include 50 ms context on both sides.
    priority = np.asarray([distances[max(0, 2*i-5):min(len(distances), 2*i+9)].mean()
                           for i in range(token_length)])
    order = np.argsort(-priority, kind="stable")
    return order[:min(maximum, token_length)]


def transition_nll(tokens, probabilities):
    """Mean transition surprise in nats; no start/end prior or cross-utterance edges."""
    if probabilities is None or len(tokens) < 2:
        return 0.0
    return float(-np.log(probabilities[tokens[:-1], tokens[1:]]).mean())


def transition_statistics(tokens, probabilities, counts):
    pairs = counts[tokens[:-1], tokens[1:]]
    runs = np.diff(np.r_[0, np.flatnonzero(tokens[1:] != tokens[:-1]) + 1, len(tokens)])
    return {'transition_nll': transition_nll(tokens, probabilities),
            'unseen_transition_fraction': float((pairs == 0).mean()) if len(pairs) else 0.0,
            'self_transition_fraction': float((tokens[:-1] == tokens[1:]).mean()) if len(pairs) else 0.0,
            'mean_run_tokens': float(runs.mean()), 'max_run_tokens': int(runs.max())}


def search_tokens(initial, target, evaluate, decoded_normalized, allowed, *,
                  passes=2, positions=24, neighbors=12, global_candidates=4,
                  batch_size=4, min_improvement=1e-5, on_step=None,
                  transition_probs=None, transition_weight=0.0):
    """Budgeted coordinate descent, accepting only independently verified discrete improvements.

    evaluate receives (B, token_count) and returns (B, PPG_frames, 35).
    Every proposal changes one token. Optional mean transition NLL is added to mean PPG JS.
    The combined objective must improve; PPG JS alone may increase when the weight is positive.
    """
    if (passes < 1 or positions < 1 or neighbors < 0 or global_candidates < 0
            or neighbors + global_candidates < 1 or batch_size < 1
            or not np.isfinite(min_improvement) or min_improvement <= 0
            or not np.isfinite(transition_weight) or transition_weight < 0):
        raise ValueError("Invalid search budget")
    if transition_probs is not None:
        transition_probs = np.asarray(transition_probs, dtype=np.float64)
        if (transition_probs.shape != (len(decoded_normalized), len(decoded_normalized))
                or not np.isfinite(transition_probs).all() or np.any(transition_probs <= 0)
                or not np.allclose(transition_probs.sum(1), 1, atol=1e-10, rtol=0)):
            raise ValueError('Expected positive normalized smoothed transition probabilities')
    elif transition_weight:
        raise ValueError('Positive transition weight requires transition probabilities')
    tokens = np.asarray(initial, dtype=np.int64).copy()
    if (tokens.ndim != 1 or not len(tokens) or tokens.min() < 0
            or tokens.max() >= len(decoded_normalized)):
        raise ValueError("Invalid initial tokens")
    current = np.asarray(evaluate(tokens[None]))[0]
    baseline = current.copy()
    distance = posterior_score(target, current)
    js_score = float(distance.mean())
    nll = transition_nll(tokens, transition_probs)
    score = js_score + transition_weight * nll
    baseline_score, baseline_js, baseline_nll = score, js_score, nll
    evaluations, trace = 1, []
    for pass_index in range(passes):
        accepted = 0
        for position in priority_positions(distance, len(tokens), positions):
            old = int(tokens[position])
            codes = candidate_codes(old, decoded_normalized, allowed, neighbors, global_candidates)
            best_score, best_code = score, old
            for offset in range(0, len(codes), batch_size):
                group = codes[offset:offset + batch_size]
                proposals = np.repeat(tokens[None], len(group), axis=0)
                proposals[:, position] = group
                results = np.asarray(evaluate(proposals))
                if len(results) != len(group):
                    raise ValueError("Evaluator returned the wrong candidate batch size")
                evaluations += len(group)
                for code, posterior, proposal in zip(group, results, proposals):
                    trial_score = (float(posterior_score(target, posterior).mean())
                                   + transition_weight * transition_nll(proposal, transition_probs))
                    if trial_score < best_score - min_improvement:
                        best_score, best_code = trial_score, int(code)
            previous_score, previous_js, previous_nll = score, js_score, nll
            verified_score, verified_js, verified_nll = None, None, None
            if best_code != old:
                proposal = tokens.copy()
                proposal[position] = best_code
                verified = np.asarray(evaluate(proposal[None]))[0]
                evaluations += 1
                verified_distance = posterior_score(target, verified)
                verified_js = float(verified_distance.mean())
                verified_nll = transition_nll(proposal, transition_probs)
                verified_score = verified_js + transition_weight * verified_nll
                if verified_score < score - min_improvement:
                    tokens, current, distance, score = proposal, verified, verified_distance, verified_score
                    js_score, nll = verified_js, verified_nll
                    accepted += 1
            item = {"pass": pass_index + 1, "position": int(position), "time_seconds": int(position) / 50,
                    "old_code": old, "new_code": int(tokens[position]), "candidate_count": len(codes),
                    "accepted": int(tokens[position]) != old, "previous_js": previous_js,
                    "previous_transition_nll": previous_nll, "previous_objective": previous_score,
                    "best_batch_objective": best_score, "verified_js": verified_js,
                    "verified_transition_nll": verified_nll, "verified_objective": verified_score,
                    "current_js": js_score, "current_transition_nll": nll,
                    "current_objective": score, "transition_weight": transition_weight,
                    "evaluations": evaluations}
            trace.append(item)
            if on_step:
                on_step(item)
        if not accepted:
            break
    return {"tokens": tokens, "ppg": current, "baseline_ppg": baseline,
            "baseline_js": baseline_js, "final_js": js_score,
            "baseline_transition_nll": baseline_nll, "final_transition_nll": nll,
            "baseline_objective": baseline_score, "final_objective": score,
            "evaluations": evaluations, "trace": trace}


class TokenEvaluator:
    def __init__(self, generator, extractor, windows, conditions, gain, device, *, vq_model=None, mean=None, std=None):
        self.generator, self.extractor = generator, extractor
        self.windows, self.conditions, self.gain, self.device = windows, conditions, gain, device
        self.vq_model, self.mean, self.std = vq_model, mean, std

    def trajectories(self, sequences):
        if self.vq_model is not None:
            # Decode each full sequence as in the stored baseline. A codebook lookup table
            # changes GEMM rounding with batch shape and can measurably change the waveform.
            with torch.inference_mode():
                return np.stack([overlap_average(self.vq_model.decoder(
                    self.vq_model.codebook(torch.as_tensor(z, dtype=torch.long))).numpy().reshape(len(z), 2, 12))
                    * self.std + self.mean for z in sequences])
        return np.stack([overlap_average(self.windows[z], output_length=len(z)) for z in sequences])

    def synthesize(self, sequences):
        ema = self.trajectories(sequences)
        count = len(sequences)
        art = np.concatenate((ema, np.repeat(self.conditions['pitch'][None], count, axis=0),
                              np.repeat(self.conditions['loudness'][None], count, axis=0)), axis=2)
        speaker = torch.as_tensor(self.conditions['spk_emb'], device=self.device).float()[None].expand(count, -1)
        with torch.inference_mode():
            audio = self.generator(torch.from_numpy(art).to(self.device).transpose(1, 2), speaker)
            # Match the actual PCM_16 files during candidate evaluation, with a fixed gain.
            audio = torch.floor(audio * self.gain * 32768).clamp(-32768, 32767) / 32768
        if not torch.isfinite(audio).all():
            raise ValueError("Nonfinite synthesized audio")
        return audio

    def __call__(self, sequences):
        with torch.inference_mode():
            audio = self.synthesize(sequences)
            mel = self.extractor.frontend(audio)
            result = self.extractor.model(mel).softmax(1).transpose(1, 2).cpu().numpy()
        if result.shape != (len(sequences), audio.shape[-1] // HOP, len(PHONEMES)):
            raise ValueError("Invalid candidate PPG dimensions")
        return result


def load_generator(device):
    # Only the synthesizer is needed: source features and the speaker embedding are already stored.
    from .generator import HiFiGANGenerator
    path = hf_hub_download("cheoljun95/Speech-Articulatory-Coding", "model_multiling.ckpt", revision=SPARC_REVISION)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(checkpoint['config']['generator_configs'])
    config['spk_emb_size'] = checkpoint['config']['spk_emb_size']
    generator = HiFiGANGenerator(**config)
    generator.load_state_dict(checkpoint['state_dict']['generator'], strict=True)
    generator.remove_weight_norm()
    return generator.eval().requires_grad_(False).to(device), sha256_file(path)


def plot_result(path, target, before, after, initial, edited, trace):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 1, figsize=(13, 13), gridspec_kw={"height_ratios": [3, 3, 3, 1.5, 1]})
    for axis, ppg, label in zip(axes[:3], [target, before, after], ["Target: original audio", "Before token editing", "After token editing"]):
        axis.imshow(ppg.T, aspect='auto', vmin=0, vmax=1, interpolation='nearest',
                    extent=(0, len(ppg)/100, 34.5, -0.5), cmap='viridis')
        axis.set_yticks(range(35), PHONEMES, fontsize=6)
        axis.set_title(label)
        axis.set_ylabel('Phone')
    for posterior, label in [(before, 'Before'), (after, 'After')]:
        distances = posterior_score(target, posterior)
        axes[3].plot(frame_times(len(distances)), distances, label=label, linewidth=1)
    for index in np.flatnonzero(initial != edited):
        axes[3].axvline(index / 50, color='black', alpha=.15)
    axes[3].set(xlabel='Time (s); vertical lines: edited tokens', ylabel='JS (nats)', ylim=(0, np.log(2)))
    axes[3].legend()
    axes[4].plot(range(len(trace) + 1), [float(posterior_score(target, before).mean())] + [x['current_js'] for x in trace])
    axes[4].set(xlabel='Token positions evaluated', ylabel='Mean JS')
    if trace and trace[0].get('transition_weight', 0):
        other = axes[4].twinx()
        other.plot(range(len(trace) + 1), [trace[0]['previous_objective']] + [x['current_objective'] for x in trace],
                   color='tab:orange', label='Combined objective')
        other.set_ylabel('JS + lambda * transition NLL', color='tab:orange')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(output, records, transition_weight=0.0):
    from .compare_jsut_audio import audio_player
    cards = []
    for record in records:
        folder = record['directory']
        players = ''.join(f'<div><h3>{label}</h3>{audio_player(folder+"/"+name+".wav")}</div>'
                          for name, label in [('original', '元音声（目標）'), ('original_ema', '元軌道から合成'),
                                              ('before', 'トークン編集前'), ('after', 'トークン編集後')])
        cards.append(f'<section><h2>{html.escape(record["utterance_id"])}</h2><p>{html.escape(record["text"])}</p>'
                     f'<p>平均JS：{record["baseline_js"]:.5f} → {record["final_js"]:.5f}'
                     f'（{record["relative_improvement_percent"]:.1f}%減少）、'
                     f'編集 {record["edited_tokens"]}/{record["token_count"]} トークン</p>'
                     + (f'<p>遷移NLL：{record["baseline_transition_nll"]:.4f} → {record["final_transition_nll"]:.4f} nats/遷移、'
                        f'合計損失：{record["baseline_objective"]:.5f} → {record["final_objective"]:.5f}</p>'
                        if 'baseline_transition_nll' in record else '') +
                     f'<div class="players">{players}</div><a href="{folder}/comparison.png">図を拡大</a>'
                     f'<img src="{folder}/comparison.png" loading="lazy"><p><a href="{folder}/result.npz">結果NPZ</a> / '
                     f'<a href="{folder}/trace.jsonl">探索履歴</a></p></section>')
    content = '''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>トークン局所探索とPPG</title><style>body{font-family:system-ui;max-width:1200px;margin:30px auto;padding:0 20px}section{border:1px solid #ccc;padding:18px;margin:24px 0}.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:15px}audio,img{width:100%}h3{font-size:1rem}</style>
<h1>40 ms・128コード：トークン局所探索</h1><p>元音声のPPGを目標に、1トークンずつ置換しました。F0・音量・話者情報・合成ゲイン・系列長は固定です。
同じ絶対時刻の共通区間で平均JSを評価します。編集数・軌道の正則化は使っていません。
予算内の探索であり、全コード・全位置を探索し尽くした結果や、音質改善の保証ではありません。</p><p><a href="summary.csv">集計CSV</a> / <a href="manifest.json">実験設定</a></p>'''
    content += (f'<p>λ={transition_weight:g}。採用基準は「平均JS＋λ×平均遷移負対数確率」の低下です。'
                '遷移スコアを改善するために、一部の編集ではPPG差が増える場合があります。</p>'
                if transition_weight else '<p>λ=0：遷移確率の罰則を使わず、平均JSが下がる変更だけを採用しました。</p>')
    content += ''.join(cards)
    content += "<script>document.querySelectorAll('audio').forEach(p=>p.onplay=()=>document.querySelectorAll('audio').forEach(q=>{if(q!==p)q.pause()}));</script></html>"
    (output/'index.html').write_text(content, encoding='utf-8')


def prepare(args):
    manifest = json.loads((args.ppg_dir/'manifest.json').read_text())
    config, splits, model, mean, std = load_run(args.run_dir, torch.device('cpu'))
    digest = sha256_file(args.run_dir/'best.pt')
    if (config['window_frames'] != 2 or config['codebook_size'] != 128 or model.token_stride != 1
            or manifest['vq_checkpoint_sha256'] != digest or Path(manifest['run_dir']).resolve() != args.run_dir.resolve()):
        raise ValueError('Expected matching 40 ms / 128-code / 50 Hz input artifacts')
    if (manifest['ppg_model']['model_revision'] != MODEL_REVISION
            or tuple(manifest['ppg_model']['phonemes']) != PHONEMES):
        raise ValueError('PPG model metadata mismatch')
    comparison = json.loads((args.run_dir/'audio_comparison/manifest.json').read_text())
    if comparison['checkpoint_sha256'] != digest:
        raise ValueError('Stale synthesis conditions')
    with torch.inference_mode():
        normalized = model.decoder(model.codebook.weight).numpy().reshape(128, 2, 12)
    windows = normalized * std + mean
    # Fit only within training utterances; no validation/test sequences enter the prior.
    training_sequences = []
    for relative in splits['train']:
        token_path = args.run_dir/'tokens'/relative
        if token_path.exists():
            with np.load(token_path, allow_pickle=False) as data:
                if str(data['checkpoint_sha256']) != digest or str(data['split']) != 'train':
                    raise ValueError('Stale training tokens')
                training_tokens = data['tokens']
        else:
            features = checked_feature(config, relative)
            training_tokens, _, _ = infer_utterance(model, features['ema'], mean, std, 4096, torch.device('cpu'))
        training_sequences.append(training_tokens.copy())
    markov = fit_markov(training_sequences, 128, args.transition_alpha)
    counts = markov['token_counts']
    selected = []
    for utterance in args.utterances:
        matches = [r for r in manifest['records'] if r['utterance_id'] == utterance]
        if len(matches) != 1:
            raise ValueError(f'Expected one PPG record for {utterance}')
        record = matches[0]
        relative = Path(record['feature_path'])
        if relative.as_posix() not in splits['test'] or record['split'] != 'test':
            raise ValueError('Expected test utterance')
        checked_feature(config, relative.as_posix())
        directory = args.run_dir/'audio_comparison'/relative.with_suffix('')
        with np.load(directory/'conditions.npz', allow_pickle=False) as data:
            conditions = dict(data)
        with np.load(args.run_dir/'tokens'/relative, allow_pickle=False) as data:
            if str(data['checkpoint_sha256']) != digest:
                raise ValueError('Stale test tokens')
            tokens = data['tokens'].copy()
        reconstructed = overlap_average(windows[tokens])
        if str(conditions['checkpoint_sha256']) != digest or not np.allclose(reconstructed, conditions['reconstruction'], atol=2e-5, rtol=1e-5):
            raise ValueError('Tokens do not match synthesis conditions')
        for name in ['pitch', 'loudness']:
            if conditions[name].shape != (len(tokens), 1) or not np.isfinite(conditions[name]).all():
                raise ValueError(f'Invalid fixed {name}')
        if conditions['spk_emb'].shape != (64,) or not np.isfinite(conditions['spk_emb']).all():
            raise ValueError('Invalid fixed speaker embedding')
        item = next(r for r in comparison['records'] if r['utterance_id'] == utterance)
        gain = item['synthesis_common_gain']
        for variant in record['variants'].values():
            if sha256_file(args.ppg_dir/variant['audio']) != variant['audio_sha256']:
                raise ValueError('PPG input audio checksum mismatch')
        with np.load(args.ppg_dir/record['variants']['original']['ppg'], allow_pickle=False) as data:
            if str(data['model_revision']) != MODEL_REVISION or tuple(data['phonemes']) != PHONEMES:
                raise ValueError('Invalid target PPG provenance')
            target = data['ppg'].copy()
            if not np.array_equal(data['time'], frame_times(len(target))):
                raise ValueError('Invalid target time grid')
        selected.append((record, conditions, tokens, target, gain))
    return manifest, digest, normalized, windows, counts, selected, model, mean, std, markov


def main(argv=None, *, prepared=None, models=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('outputs/jsut_window_sweep/window_040ms'))
    parser.add_argument('--ppg-dir', type=Path, default=Path('outputs/jsut_ppg_40ms128'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/jsut_ppg_local_search'))
    parser.add_argument('--utterances', nargs='+', default=DEFAULT_IDS)
    parser.add_argument('--passes', type=int, default=2)
    parser.add_argument('--positions', type=int, default=24)
    parser.add_argument('--neighbors', type=int, default=12)
    parser.add_argument('--global-candidates', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--min-improvement', type=float, default=1e-5)
    parser.add_argument('--transition-weight', type=float, default=0.0)
    parser.add_argument('--transition-alpha', type=float, default=1.0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    if (args.passes < 1 or args.positions < 1 or args.batch_size < 1 or args.threads < 1
            or args.neighbors < 0 or args.global_candidates < 0
            or args.neighbors + args.global_candidates < 1 or args.min_improvement <= 0
            or not np.isfinite([args.min_improvement, args.transition_weight, args.transition_alpha]).all()
            or args.transition_weight < 0 or args.transition_alpha <= 0):
        parser.error('Invalid search budget')
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error('Output directory is not empty; choose a new --output-dir')
    if len(set(args.utterances)) != len(args.utterances):
        parser.error('Duplicate utterance IDs')
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    source_manifest, digest, normalized, windows, counts, selected, vq_model, mean, std, markov = (
        prepare(args) if prepared is None else prepared)
    print(f'Prepared {len(selected)} utterances; {np.count_nonzero(counts)}/128 training codes', flush=True)
    if models is None:
        generator, generator_digest = load_generator(args.device)
        extractor = JapanesePPGExtractor(args.device)
    else:
        generator, generator_digest, extractor = models
    if extractor.checkpoint_sha256 != source_manifest['ppg_model']['checkpoint_sha256']:
        parser.error('Target PPG checkpoint mismatch')
    configure_plots()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_npz(args.output_dir/'transitions.npz', **markov, alpha=np.float64(args.transition_alpha),
             checkpoint_sha256=np.str_(digest))
    manifest = {'settings': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                'objective': 'mean PPG JS + transition_weight * mean transition NLL; common absolute times',
                'target': 'original audio PPG', 'transition_probabilities_used': args.transition_weight > 0,
                'regularization': 'mean transition NLL' if args.transition_weight else 'none',
                'transition_model': {'fit_split': 'train', 'utterances': len(json.loads((args.run_dir/'splits.json').read_text())['train']),
                                     'alpha': args.transition_alpha, 'includes_self_transitions': True,
                                     'cross_utterance_transitions': False, 'file': 'transitions.npz',
                                     'sha256': sha256_file(args.output_dir/'transitions.npz')},
                'candidate_selection': 'nearest decoded windows plus farthest-point coverage; observed training codes only',
                'training_code_counts': counts.tolist(), 'vq_checkpoint_sha256': digest,
                'ppg_model_id': MODEL_ID, 'ppg_revision': MODEL_REVISION, 'ppg_source_commit': SOURCE_COMMIT,
                'ppg_checkpoint_sha256': extractor.checkpoint_sha256,
                'synthesis_revision': SPARC_REVISION, 'synthesis_checkpoint_sha256': generator_digest,
                'torch_version': torch.__version__, 'records': []}
    write_json(args.output_dir/'manifest.json', manifest)
    records = []
    for record, conditions, tokens, target, gain in selected:
        utterance = record['utterance_id']
        directory = args.output_dir/Path(record['feature_path']).with_suffix('')
        directory.mkdir(parents=True)
        evaluator = TokenEvaluator(generator, extractor, windows, conditions, gain, args.device,
                                   vq_model=vq_model, mean=mean, std=std)
        # Reproduce the saved baseline audio before editing; catch mismatched synthesis models/conditions.
        baseline_audio = evaluator.synthesize(tokens[None])[0,0].cpu().numpy()
        saved_audio, sr = sf.read(args.ppg_dir/record['variants']['reconstructed_ema']['audio'], dtype='float32')
        if sr != 16000 or saved_audio.shape != baseline_audio.shape:
            raise ValueError('Baseline audio shape mismatch')
        audio_error = float(np.max(np.abs(saved_audio - baseline_audio)))
        if audio_error > 3/32768:
            raise ValueError(f'Baseline synthesis differs from the saved audio: {audio_error}')
        tick = time.monotonic()
        with (directory/'trace.jsonl').open('w') as trace_file:
            def on_step(item):
                trace_file.write(json.dumps(item)+'\n'); trace_file.flush()
                print(f'{utterance} pass {item["pass"]} position {item["position"]}: JS={item["current_js"]:.6f}, NLL={item["current_transition_nll"]:.4f}, objective={item["current_objective"]:.6f}, accepted={item["accepted"]}, evaluations={item["evaluations"]}', flush=True)
            result = search_tokens(tokens, target, evaluator, normalized, np.flatnonzero(counts),
                                   passes=args.passes, positions=args.positions, neighbors=args.neighbors,
                                   global_candidates=args.global_candidates, batch_size=args.batch_size,
                                   min_improvement=args.min_improvement, on_step=on_step,
                                   transition_probs=markov['transition_probs'], transition_weight=args.transition_weight)
        final_audio = evaluator.synthesize(result['tokens'][None])[0,0].cpu().numpy()
        sf.write(directory/'before.wav', baseline_audio, 16000, subtype='PCM_16')
        sf.write(directory/'after.wav', final_audio, 16000, subtype='PCM_16')
        # Verify PPG scores against the bytes the user will actually hear.
        from .japanese_ppg import load_audio
        saved_posterior = extractor.extract(load_audio(directory/'after.wav')[0])
        saved_score = float(posterior_score(target, saved_posterior).mean())
        if abs(saved_score-result['final_js']) > 1e-5:
            raise ValueError('Saved WAV PPG score differs from accepted result')
        for variant in ['original', 'original_ema']:
            shutil.copyfile(args.ppg_dir/record['variants'][variant]['audio'], directory/f'{variant}.wav')
        edited = int(np.count_nonzero(tokens != result['tokens']))
        before_d, after_d = [posterior_score(target, p) for p in [result['baseline_ppg'], result['ppg']]]
        n = len(before_d)
        summary = {'utterance_id': utterance, 'text': record['text'],
                   'directory': directory.relative_to(args.output_dir).as_posix(),
                   'baseline_js': result['baseline_js'], 'final_js': result['final_js'],
                   'transition_weight': args.transition_weight,
                   'baseline_objective': result['baseline_objective'], 'final_objective': result['final_objective'],
                   **{f'{stage}_{key}': value for stage, seq in [('baseline', tokens), ('final', result['tokens'])]
                      for key, value in transition_statistics(seq, markov['transition_probs'], markov['transition_counts']).items()},
                   'relative_improvement_percent': 100*(1-result['final_js']/result['baseline_js']) if result['baseline_js'] else 0,
                   'baseline_argmax_agreement': float((target[:n].argmax(1)==result['baseline_ppg'][:n].argmax(1)).mean()),
                   'final_argmax_agreement': float((target[:n].argmax(1)==result['ppg'][:n].argmax(1)).mean()),
                   'edited_tokens': edited, 'token_count': len(tokens), 'accepted_edits': sum(x['accepted'] for x in result['trace']),
                   'evaluations': result['evaluations'], 'common_ppg_frames': n,
                   'baseline_audio_max_absolute_error': audio_error, 'saved_wav_mean_js': saved_score,
                   'seconds': time.monotonic()-tick, 'fixed_synthesis_gain': gain}
        save_npz(directory/'result.npz', initial_tokens=tokens, edited_tokens=result['tokens'],
                 initial_ema=evaluator.trajectories(tokens[None])[0], edited_ema=evaluator.trajectories(result['tokens'][None])[0],
                 target_ppg=target, before_ppg=result['baseline_ppg'], after_ppg=result['ppg'],
                 target_time=frame_times(len(target)), time=frame_times(len(result['ppg'])),
                 before_js=before_d, after_js=after_d, comparison_time=frame_times(n), phonemes=np.asarray(PHONEMES),
                 token_time=conditions['time'], pitch=conditions['pitch'], loudness=conditions['loudness'], spk_emb=conditions['spk_emb'],
                 transition_weight=np.float64(args.transition_weight), transition_alpha=np.float64(args.transition_alpha),
                 vq_checkpoint_sha256=np.str_(digest), ppg_checkpoint_sha256=np.str_(extractor.checkpoint_sha256))
        plot_result(directory/'comparison.png', target, result['baseline_ppg'], result['ppg'], tokens, result['tokens'], result['trace'])
        write_json(directory/'summary.json', summary)
        records.append(summary)
        manifest['records'] = records
        manifest['elapsed_seconds'] = time.monotonic()-started
        write_json(args.output_dir/'manifest.json', manifest)
        with (args.output_dir/'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary)); writer.writeheader(); writer.writerows(records)
        write_report(args.output_dir, records, args.transition_weight)
        print(f'{utterance}: {summary["baseline_js"]:.6f} -> {summary["final_js"]:.6f}; {edited}/{len(tokens)} edited', flush=True)
    print(f'Report: {args.output_dir}/index.html', flush=True)
    return 0


if __name__ == '__main__':
    main()
