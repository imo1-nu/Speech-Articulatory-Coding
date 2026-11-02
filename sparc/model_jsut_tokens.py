"""Train a trajectory VQ-VAE, tokenize JSUT, and analyze direct token transitions."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .trajectory_vq import (
    CHANNELS, TrajectoryVQVAE, WindowDataset, fit_markov, normalization,
    overlap_average, score_markov, split_utterances, usage_statistics, window_padding,
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def atomic_save(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_npz(path, **arrays):
    atomic_save(path, lambda stream: np.savez_compressed(stream, **arrays))


def checkpoint_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_feature(path):
    with np.load(path, allow_pickle=False) as stored:
        data = dict(stored)
    values = data["ema"]
    if (values.ndim != 2 or values.shape[1] != 12 or not len(values)
            or not np.isfinite(values).all()):
        raise ValueError(f"Invalid finite (T, 12) EMA: {path}")
    if float(data["frame_rate"]) != 50 or not np.array_equal(data["channels"], CHANNELS):
        raise ValueError(f"Expected 50 Hz features in the standard SPARC channel order: {path}")
    time = data["time"]
    if (time.shape != (len(values),) or not np.isfinite(time).all()
            or not np.allclose(np.diff(time), 0.02, atol=1e-7, rtol=0)):
        raise ValueError(f"Expected one uniformly spaced time value per EMA frame: {path}")
    for name in ("pitch", "periodicity", "loudness"):
        if name in data and (data[name].shape != (len(values), 1)
                             or not np.isfinite(data[name]).all()):
            raise ValueError(f"Invalid aligned {name}: {path}")
    data["ema"] = values.astype(np.float32)
    return data


def load_run(run_dir, device):
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    splits = json.loads((run_dir / "splits.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model = TrajectoryVQVAE(config["codebook_size"], config["latent_dim"],
                           config.get("window_frames", 5), config.get("token_stride", 1)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return (config, splits, model, checkpoint["mean"].numpy(), checkpoint["std"].numpy())


def checked_feature(config, relative):
    source = Path(config["features_dir"]) / relative
    signature = config["files"][relative]
    stat = source.stat()
    if stat.st_size != signature["size"] or stat.st_mtime_ns != signature["mtime_ns"]:
        raise ValueError(f"Features changed since training: {source}; create a new run")
    return load_feature(source)


def data_loader(dataset, batch_size, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, generator=generator)


class CachedWindowBatches:
    """Cache fixed windows on the selected device to avoid per-frame Python loading."""

    def __init__(self, dataset, batch_size, device, shuffle=False, seed=42):
        width = dataset.window_frames
        self.windows = torch.cat([values.unfold(0, width, dataset.stride).permute(0, 2, 1).reshape(-1, width * 12)
                                  for values in dataset.values]).to(device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        order = (torch.randperm(len(self.windows), generator=self.generator).to(self.windows.device)
                 if self.shuffle else None)
        for start in range(0, len(self.windows), self.batch_size):
            end = start + self.batch_size
            yield self.windows[start:end] if order is None else self.windows[order[start:end]]


def epoch_pass(model, loader, device, commitment, optimizer=None):
    model.train(optimizer is not None)
    totals = {name: 0.0 for name in ("reconstruction_loss", "codebook_loss", "commitment_loss")}
    counts = np.zeros(model.codebook.num_embeddings, dtype=np.int64)
    frames = 0
    with torch.set_grad_enabled(optimizer is not None):
        for windows in loader:
            windows = windows.to(device)
            result = model(windows)
            loss = (result["reconstruction_loss"] + result["codebook_loss"]
                    + commitment * result["commitment_loss"])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            frames += len(windows)
            for name in totals:
                totals[name] += result[name].item() * len(windows)
            counts += np.bincount(result["tokens"].detach().cpu().numpy(), minlength=len(counts))
    return {**{name: value / frames for name, value in totals.items()}, **usage_statistics(counts)}


def infer_utterance(model, ema, mean, std, batch_size, device):
    dataset = WindowDataset([ema], mean, std, model.window_frames, model.token_stride)
    tokens, windows, distances = [], [], []
    with torch.inference_mode():
        for batch in data_loader(dataset, batch_size):
            batch = batch.to(device)
            latent = model.encoder(batch)
            quantized, indices = model.quantize(latent)
            tokens.append(indices.cpu().numpy())
            windows.append(model.decoder(quantized).cpu().numpy().reshape(-1, model.window_frames, 12))
            distances.append((latent - quantized).square().sum(-1).cpu().numpy())
    windows = np.concatenate(windows)
    reconstructed = overlap_average(windows, stride=model.token_stride, output_length=len(ema)) * std + mean
    return np.concatenate(tokens), reconstructed, np.concatenate(distances)


def train(args, device):
    feature_dir = args.features_dir.resolve()
    paths = []
    for subset in sorted(set(args.subsets)):
        if subset != Path(subset).name or subset in (".", ".."):
            raise ValueError("Subset names must be immediate child directories")
        subset_paths = sorted((feature_dir / subset).glob("*.npz"))
        if not subset_paths:
            raise ValueError(f"No features found for subset: {subset}")
        paths.extend(path.relative_to(feature_dir).as_posix() for path in subset_paths)
    if args.limit is not None:
        paths = paths[:args.limit]
    splits = split_utterances(paths, args.seed)
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise ValueError(f"Run directory is not empty: {args.run_dir}; choose a new --run-dir")

    print(f"Loading {len(paths)} utterances...", file=sys.stderr, flush=True)
    arrays = {}
    signatures = {}
    for relative in tqdm(paths, desc="Loading features", unit="utterance"):
        source = feature_dir / relative
        arrays[relative] = load_feature(source)["ema"]
        stat = source.stat()
        signatures[relative] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                "frames": len(arrays[relative])}
    mean, std = normalization([arrays[path] for path in splits["train"]])
    window_frames = args.window_ms // 20
    stride = 50 // args.token_rate
    left, right = window_padding(window_frames)
    datasets = {name: WindowDataset([arrays[path] for path in group], mean, std, window_frames, stride)
                for name, group in splits.items()}
    if len(datasets["train"]) < args.codebook_size:
        raise ValueError("Training frames must be at least the codebook size")
    config = {"format_version": 1, "features_dir": str(feature_dir), "files": signatures,
              "subsets": args.subsets, "seed": args.seed, "frame_rate": 50,
              "window_frames": window_frames, "window_ms": args.window_ms,
              "token_rate": args.token_rate, "token_stride": stride,
              "window_left_frames": left, "window_right_frames": right,
              "window_padding": "edge", "window_alignment": "centered_even_right",
              "cache_windows": args.cache_windows,
              "channels": list(CHANNELS), "codebook_size": args.codebook_size,
              "latent_dim": args.latent_dim, "batch_size": args.batch_size,
              "learning_rate": args.learning_rate, "commitment": args.commitment,
              "max_epochs": args.epochs, "patience": args.patience,
              "codebook_initialization": "sampled_training_encoder_outputs",
              "torch_version": str(torch.__version__), "numpy_version": np.__version__,
              "device": str(device), "threads": args.threads}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.run_dir / "config.json", config)
    write_json(args.run_dir / "splits.json", splits)
    save_npz(args.run_dir / "normalization.npz", mean=mean, std=std, channels=np.asarray(CHANNELS))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = TrajectoryVQVAE(args.codebook_size, args.latent_dim, window_frames, stride).to(device)
    # Match the initial codebook scale to encoder outputs using training data only.
    initial_indices = np.random.default_rng(args.seed).choice(
        len(datasets["train"]), args.codebook_size, replace=False
    )
    with torch.no_grad():
        initial = torch.stack([datasets["train"][index] for index in initial_indices]).to(device)
        model.codebook.weight.copy_(model.encoder(initial))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.cache_windows:
        loaders = {name: CachedWindowBatches(dataset, args.batch_size, device,
                                             shuffle=(name == "train"), seed=args.seed)
                   for name, dataset in datasets.items()}
    else:
        loaders = {name: data_loader(dataset, args.batch_size, name == "train", args.seed)
                   for name, dataset in datasets.items()}
    train_loader, validation_loader = loaders["train"], loaders["validation"]
    best = float("inf")
    best_epoch = 0
    stale = 0
    with (args.run_dir / "history.jsonl").open("w", encoding="utf-8") as history:
        for epoch in range(1, args.epochs + 1):
            training = epoch_pass(model, train_loader, device, args.commitment, optimizer)
            validation = epoch_pass(model, validation_loader, device, args.commitment)
            record = {"epoch": epoch, "train": training, "validation": validation}
            history.write(json.dumps(record, allow_nan=False) + "\n")
            history.flush()
            score = validation["reconstruction_loss"]
            print(f"Epoch {epoch}: train MSE={training['reconstruction_loss']:.6f}, "
                  f"validation MSE={score:.6f}, "
                  f"validation codes={validation['used_codes']}/{args.codebook_size}", flush=True)
            if score < best - 1e-5:
                best, best_epoch, stale = score, epoch, 0
                checkpoint = {"state_dict": model.state_dict(), "mean": torch.from_numpy(mean),
                              "std": torch.from_numpy(std), "epoch": epoch}
                atomic_save(args.run_dir / "best.pt", lambda stream: torch.save(checkpoint, stream))
            else:
                stale += 1
            if stale >= args.patience:
                break
    _, _, model, _, _ = load_run(args.run_dir, device)
    test = epoch_pass(model, loaders["test"], device, args.commitment)
    write_json(args.run_dir / "training_metrics.json", {
        "best_epoch": best_epoch, "best_validation_window_mse": best, "test": test,
        "split_utterances": {name: len(group) for name, group in splits.items()},
        "split_frames": {name: sum(len(arrays[path]) for path in group) for name, group in splits.items()},
        "split_tokens": {name: len(dataset) for name, dataset in datasets.items()},
        "note": "Window MSE is in training-standardized coordinates; no phonetic labels are used.",
    })
    print(f"Best epoch: {best_epoch}; test window MSE={test['reconstruction_loss']:.6f}")


def plot_reconstruction(path, time, original, reconstructed):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 2, figsize=(12, 12), sharex=True)
    for channel, axis in enumerate(axes.flat):
        axis.plot(time, original[:, channel], label="Input", linewidth=1)
        axis.plot(time, reconstructed[:, channel], label="VQ reconstruction", linewidth=1)
        axis.set_ylabel(CHANNELS[channel])
    axes[0, 0].legend(fontsize=8)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def configure_plots():
    import matplotlib

    matplotlib.use("Agg")


def tokenize(args, device):
    configure_plots()
    config, splits, model, mean, std = load_run(args.run_dir, device)
    digest = checkpoint_hash(args.run_dir / "best.pt")
    size = config["codebook_size"]
    counts = {split: np.zeros(size, dtype=np.int64) for split in splits}
    errors = {split: np.zeros(12, dtype=np.float64) for split in splits}
    baseline = {split: np.zeros(12, dtype=np.float64) for split in splits}
    frame_counts = dict.fromkeys(splits, 0)
    representatives = [None] * size
    representative_distances = np.full(size, np.inf)
    width = model.window_frames
    left, right = window_padding(width)
    representative_windows = np.zeros((size, width, 12), dtype=np.float32)
    plot_dir = args.run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    for split, relatives in splits.items():
        for number, relative in enumerate(tqdm(relatives, desc=f"Tokenize {split}", unit="utterance")):
            data = checked_feature(config, relative)
            original = data["ema"]
            tokens, reconstructed, distances = infer_utterance(
                model, original, mean, std, args.batch_size, device
            )
            auxiliary = {key: data[key] for key in ("pitch", "periodicity", "loudness", "text", "source")
                         if key in data}
            save_npz(args.run_dir / "tokens" / relative, tokens=tokens.astype(np.int64),
                     time=data["time"], token_time=data["time"][::model.token_stride],
                     frame_rate=np.float64(50), token_rate=np.float64(50 / model.token_stride),
                     token_stride=np.int64(model.token_stride), token_len=np.int64(len(tokens)),
                     ft_len=np.int64(len(original)),
                     channels=np.asarray(CHANNELS), reconstruction=reconstructed,
                     split=np.str_(split), feature_path=np.str_(relative),
                     checkpoint_sha256=np.str_(digest), **auxiliary)
            counts[split] += np.bincount(tokens, minlength=size)
            errors[split] += ((original.astype(np.float64) - reconstructed) ** 2).sum(axis=0)
            baseline[split] += ((original.astype(np.float64) - mean) ** 2).sum(axis=0)
            frame_counts[split] += len(original)
            if number == 0:
                plot_reconstruction(plot_dir / f"reconstruction_{split}.png", data["time"],
                                    original, reconstructed)
            if split == "train":
                padded = np.pad(original, ((left, right), (0, 0)), mode="edge")
                for token in np.unique(tokens):
                    positions = np.flatnonzero(tokens == token)
                    token_index = int(positions[np.argmin(distances[positions])])
                    frame = token_index * model.token_stride
                    if distances[token_index] < representative_distances[token]:
                        representative_distances[token] = distances[token_index]
                        representative_windows[token] = padded[frame:frame + width]
                        representatives[token] = {"token": int(token), "feature_path": relative,
                                                  "frame": frame, "time": float(data["time"][frame])}
        print(f"Tokenized {split}: {len(relatives)} utterances, {frame_counts[split]} frames", flush=True)
    metrics = {}
    for split in splits:
        mse = errors[split] / frame_counts[split]
        metrics[split] = {**usage_statistics(counts[split]), "frames": frame_counts[split],
                          "tokens": int(counts[split].sum()), "token_rate": 50 / model.token_stride,
                          "rmse_per_channel": np.sqrt(mse).tolist(),
                          "normalized_mse": float(np.mean(mse / std.astype(float) ** 2)),
                          "train_mean_baseline_normalized_mse": float(np.mean(
                              baseline[split] / frame_counts[split] / std.astype(float) ** 2))}
    write_json(args.run_dir / "reconstruction_metrics.json", {
        "checkpoint_sha256": digest, "channels": list(CHANNELS), "splits": metrics,
        "note": "Utterance reconstructions average overlapping decoded windows, excluding padding.",
    })
    write_json(args.run_dir / "representatives.json", representatives)
    with torch.inference_mode():
        decoded = model.decoder(model.codebook.weight).cpu().numpy().reshape(size, width, 12) * std + mean
    save_npz(args.run_dir / "codebook_trajectories.npz", decoded=decoded,
             representative=representative_windows,
             has_representative=np.asarray([item is not None for item in representatives]),
             training_token_counts=counts["train"], channels=np.asarray(CHANNELS),
             relative_time=np.arange(-left, width - left) / 50, checkpoint_sha256=np.str_(digest))
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    top = np.argsort(-counts["train"])[:6]
    for axis, token in zip(axes.flat, top):
        if representatives[token] is None:
            axis.set_visible(False)
            continue
        colors = plt.get_cmap("tab20").colors
        for channel in range(12):
            axis.plot(np.arange(-left, width - left) * 20, representative_windows[token, :, channel],
                      color=colors[channel])
        axis.set_title(f"Token {token}; train n={counts['train'][token]}")
        axis.set_xlabel("Offset (ms)")
    for axis in list(axes.flat)[len(top):]:
        axis.set_visible(False)
    fig.legend(CHANNELS, loc="upper center", ncol=6, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(plot_dir / "representative_trajectories.png", dpi=140)
    plt.close(fig)


def transitions(args):
    configure_plots()
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    splits = json.loads((args.run_dir / "splits.json").read_text(encoding="utf-8"))
    digest = checkpoint_hash(args.run_dir / "best.pt")
    size = config["codebook_size"]
    sequences = {}
    for split, relatives in splits.items():
        sequences[split] = []
        for relative in tqdm(relatives, desc=f"Read {split} tokens", unit="utterance"):
            # Refuse stale tokens or changed input features rather than mixing experiments.
            checked_feature(config, relative)
            with np.load(args.run_dir / "tokens" / relative, allow_pickle=False) as data:
                tokens = data["tokens"]
                if (str(data["checkpoint_sha256"]) != digest or str(data["split"]) != split
                        or str(data["feature_path"]) != relative
                        or tokens.shape != ((config["files"][relative]["frames"]
                                             + config.get("token_stride", 1) - 1)
                                            // config.get("token_stride", 1),)
                        or not np.issubdtype(tokens.dtype, np.integer)
                        or tokens.min() < 0 or tokens.max() >= size):
                    raise ValueError(f"Stale or invalid tokens: {relative}; rerun tokenize")
                sequences[split].append(tokens.copy())
    model = fit_markov(sequences["train"], size, args.alpha)
    total = int(model["transition_counts"].sum())
    if not total:
        raise ValueError("Training utterances contain no within-utterance transitions")
    save_npz(args.run_dir / "transitions.npz", **model, alpha=np.float64(args.alpha),
             checkpoint_sha256=np.str_(digest))
    metrics = {"alpha": args.alpha, "checkpoint_sha256": digest,
               "model": "first_order_observed_markov_chain", "includes_self_transitions": True,
               "training_usage": usage_statistics(model["token_counts"]),
               "unused_tokens": np.flatnonzero(model["token_counts"] == 0).tolist(),
               "tokens_without_outgoing_transitions": np.flatnonzero(
                   model["transition_counts"].sum(axis=1) == 0).tolist(),
               "self_transition_fraction": float(np.trace(model["transition_counts"]) / total),
               "splits": {split: score_markov(group, model) for split, group in sequences.items()}}
    write_json(args.run_dir / "transition_metrics.json", metrics)
    import matplotlib.pyplot as plt

    plot_dir = args.run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    image = axes[0].imshow(model["transition_probs"], origin="lower", aspect="auto", vmin=0, vmax=1)
    axes[0].set(xlabel="Next token", ylabel="Current token", title="P(next | current), smoothed")
    fig.colorbar(image, ax=axes[0])
    counts = model["token_counts"]
    axes[1].bar(np.arange(size), counts / counts.sum())
    axes[1].set(xlabel="Token", ylabel="Training frequency", title="Code usage")
    fig.tight_layout()
    fig.savefig(plot_dir / "transitions_and_usage.png", dpi=160)
    plt.close(fig)
    for split, result in metrics["splits"].items():
        print(f"{split}: {result['transitions']} transitions, "
              f"perplexity={result['perplexity']}, unigram={result['unigram_perplexity']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "tokenize", "transitions"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, default=Path("outputs/jsut_tokens"))
        command.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
        if name != "transitions":
            command.add_argument("--device", default="auto")
            command.add_argument("--batch-size", type=int, default=512)
        if name == "train":
            command.add_argument("--features-dir", type=Path, default=Path("outputs/jsut_ema"))
            command.add_argument("--subsets", nargs="+", default=["basic5000"])
            command.add_argument("--limit", type=int, help="Total utterance limit for a smoke test (>=3)")
            command.add_argument("--codebook-size", type=int, default=128)
            command.add_argument("--latent-dim", type=int, default=16)
            command.add_argument("--epochs", type=int, default=100)
            command.add_argument("--patience", type=int, default=10)
            command.add_argument("--learning-rate", type=float, default=0.001)
            command.add_argument("--commitment", type=float, default=0.25)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--window-ms", type=int, default=100,
                                 help="Context window duration, a positive multiple of 20 ms")
            command.add_argument("--token-rate", type=int, choices=(25, 50), default=50,
                                 help="Tokens per second; reconstructed trajectories remain at 50 Hz")
            command.add_argument("--cache-windows", action="store_true",
                                 help="Cache fixed windows on the device for faster repeated training")
        if name == "transitions":
            command.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.command == "train" and (args.window_ms < 20 or args.window_ms % 20):
        parser.error("--window-ms must be a positive multiple of 20")
    if args.command == "train" and args.window_ms // 20 < 50 // args.token_rate:
        parser.error("The window must cover the token interval (25 Hz requires at least 40 ms)")
    for name in ("threads", "batch_size", "limit", "codebook_size", "latent_dim", "epochs", "patience"):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("learning_rate", "alpha", "commitment"):
        value = getattr(args, name, None)
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if getattr(args, "seed", 0) < 0:
        parser.error("--seed must be nonnegative")
    torch.set_num_threads(args.threads)
    device = None
    if args.command != "transitions":
        try:
            device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                                  if args.device == "auto" else args.device)
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        if device.type not in ("cpu", "cuda"):
            parser.error("--device must be auto, cpu, cuda, or cuda:N")
        if device.type == "cuda" and (not torch.cuda.is_available() or (
                device.index is not None and device.index >= torch.cuda.device_count())):
            parser.error(f"CUDA device is unavailable: {device}")
        print(f"Device: {device}; threads: {args.threads}", file=sys.stderr, flush=True)
    try:
        if args.command == "train":
            train(args, device)
        elif args.command == "tokenize":
            tokenize(args, device)
        else:
            transitions(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Results: {args.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
