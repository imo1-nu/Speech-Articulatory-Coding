"""Keep SPARC's original code intact and verify synthesis reproducibility."""

from time import perf_counter

import numpy as np
import torch


EMA_NAMES = ["TDX", "TDY", "TBX", "TBY", "TTX", "TTY",
             "LIX", "LIY", "ULX", "ULY", "LLX", "LLY"]


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def encode_decode(coder, path, device):
    synchronize(device)
    begin = perf_counter()
    with torch.inference_mode():
        code = coder.encode(path)
    synchronize(device)
    encode_s = perf_counter() - begin
    if (code["ema"].ndim != 2 or code["ema"].shape[1] != 12
            or not all(np.isfinite(v).all() for v in code.values())):
        raise RuntimeError("SPARC returned invalid features")
    snapshot = {key: np.array(value, copy=True) for key, value in code.items()}
    synthesized = []
    decode_times = []
    for _ in range(2):
        begin = perf_counter()
        with torch.inference_mode():
            wav = coder.decode(**code)
        synchronize(device)
        decode_times.append(perf_counter() - begin)
        if wav.ndim != 1 or not wav.size or not np.isfinite(wav).all():
            raise RuntimeError("SPARC returned invalid synthesis")
        synthesized.append(wav)
    for key, value in code.items():
        if not np.array_equal(value, snapshot[key]):
            raise RuntimeError(f"SPARC changed condition {key} during decoding")
    difference = synthesized[0] - synthesized[1]
    checks = {
        "code_unchanged": True,
        "decode_bitwise_equal": bool(np.array_equal(*synthesized)),
        "decode_max_abs_difference": float(np.max(np.abs(difference))),
        "decode_rms_difference": float(np.sqrt(np.mean(difference ** 2))),
    }
    return code, synthesized[0], {"encode_s": encode_s, "decode_s": decode_times}, checks
