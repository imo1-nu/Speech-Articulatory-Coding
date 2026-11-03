"""Residual quantization of a frozen continuous trajectory AE (train split only)."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from .continuous_trajectory_ae import load_ae
from .japanese_ppg import sha256_file
from .model_jsut_tokens import CachedWindowBatches, checked_feature, load_run, write_json
from .trajectory_vq import WindowDataset


class ResidualTrajectoryAE(nn.Module):
    def __init__(self, ae, codebooks, stages=None):
        super().__init__()
        self.ae = ae.requires_grad_(False)
        self.window_frames = ae.window_frames
        self.stages = len(codebooks) if stages is None else stages
        if not 1 <= self.stages <= len(codebooks):
            raise ValueError('Invalid RVQ stage count')
        self.register_buffer('codebooks', torch.as_tensor(codebooks, dtype=torch.float32))

    def quantize(self, latent):
        residual = latent
        quantized = torch.zeros_like(latent)
        tokens = []
        for centers in self.codebooks[:self.stages]:
            distances = residual.square().sum(1, keepdim=True) + centers.square().sum(1) - 2 * residual @ centers.T
            indices = distances.argmin(1)
            selected = centers[indices]
            quantized = quantized + selected
            residual = residual - selected
            tokens.append(indices)
        return quantized, torch.stack(tokens, dim=1)

    def decode_tokens(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] != self.stages or tokens.dtype != torch.long:
            raise ValueError('Expected int64 tokens with one column per stage')
        if tokens.numel() and (tokens.min() < 0 or tokens.max() >= self.codebooks.shape[1]):
            raise ValueError('Token ID out of range')
        latent = sum(self.codebooks[i][tokens[:,i]] for i in range(self.stages))
        return self.ae.decoder(latent)

    def forward(self, windows):
        return self.ae.decoder(self.quantize(self.ae.encoder(windows))[0])


def train_rvq(reference_dir, ae_dir, output_dir, device='cpu', stages=8, size=128):
    config, splits, _, mean, std = load_run(reference_dir, 'cpu')
    ae, amean, astd, ae_config = load_ae(ae_dir, device)
    np.testing.assert_array_equal(mean, amean)
    np.testing.assert_array_equal(std, astd)
    if (json.loads((ae_dir/'splits.json').read_text()) != splits
            or ae_config['reference_checkpoint_sha256'] != sha256_file(reference_dir/'best.pt')):
        raise ValueError('AE and reference data differ')
    signature = {'ae_sha256': sha256_file(ae_dir/'best.pt'), 'stages': stages, 'codebook_size': size,
                 'seed': 42, 'algorithm': 'sequential MiniBatchKMeans on train-only frozen AE latent residuals',
                 'batch_size': 4096, 'max_iter': 100, 'n_init': 3,
                 'reference_checkpoint_sha256': sha256_file(reference_dir/'best.pt')}
    if output_dir.exists() and any(output_dir.iterdir()):
        saved = json.loads((output_dir/'config.json').read_text())
        if (any(saved.get(k) != v for k,v in signature.items()) or not (output_dir/'codebooks.npz').exists()
                or json.loads((output_dir/'splits.json').read_text()) != splits):
            raise ValueError('Incompatible or incomplete RVQ run; choose a new directory')
        return
    arrays = [checked_feature(config, r)['ema'] for r in tqdm(splits['train'], desc='RVQ train features')]
    dataset = WindowDataset(arrays, mean, std, ae.window_frames, 1)
    loader = CachedWindowBatches(dataset, 8192, device)
    with torch.inference_mode():
        residual = torch.cat([ae.encoder(batch).cpu() for batch in loader]).numpy()
    codebooks = []; history = []
    with threadpool_limits(limits=4):
        for stage in range(stages):
            km = MiniBatchKMeans(n_clusters=size, batch_size=4096, max_iter=100, n_init=3,
                                 random_state=42+stage, reassignment_ratio=.01)
            km.fit(residual)
            labels = km.predict(residual)
            residual = residual - km.cluster_centers_[labels]
            codebooks.append(km.cluster_centers_.copy())
            history.append({'stage': stage+1, 'train_latent_residual_mse': float(np.square(residual).mean()),
                            'train_code_counts': np.bincount(labels, minlength=size).tolist(),
                            'iterations': int(km.n_iter_)})
            print(f'RVQ stage {stage+1}: residual MSE={history[-1]["train_latent_residual_mse"]:.7f}', flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir/'config.json', {**signature, 'ae_dir': str(ae_dir.resolve()),
               'reference_dir': str(reference_dir.resolve()), 'train_windows': len(residual),
               'training': history, 'encoder_decoder_frozen': True})
    write_json(output_dir/'splits.json', splits)
    np.savez(output_dir/'codebooks.npz', codebooks=np.stack(codebooks))


def load_rvq(directory, stages, device='cpu'):
    config = json.loads((directory/'config.json').read_text())
    ae_dir = Path(config['ae_dir'])
    if sha256_file(ae_dir/'best.pt') != config['ae_sha256']:
        raise ValueError('Frozen AE checkpoint changed')
    ae, mean, std, _ = load_ae(ae_dir, device)
    books = np.load(directory/'codebooks.npz')['codebooks']
    return ResidualTrajectoryAE(ae, books, stages).eval().to(device), mean, std


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, default=Path('outputs/jsut_window_sweep/window_040ms'))
    parser.add_argument('--ae-dir', type=Path, default=Path('outputs/jsut_reconstruction_ablation/continuous_ae'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/jsut_rvq/model'))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    train_rvq(args.reference_dir, args.ae_dir, args.output_dir, args.device)


if __name__ == '__main__':
    main()
