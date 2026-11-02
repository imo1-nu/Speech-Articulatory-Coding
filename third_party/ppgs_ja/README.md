# Japanese PPG adapter provenance

`sparc/japanese_ppg.py` adapts the Transformer, positional encoding, log-Mel
frontend and Japanese phoneme order from
<https://github.com/taisa1/ppgs_ja>, commit
`d2dac95fd6f40ba0f2708b4a3561c4912e2490d1`.
The upstream MIT license is reproduced in `LICENSE`.

Adapted upstream files:

- `ppgs/model/transformer.py`
- `ppgs/preprocess/spectrogram.py`
- `ppgs/preprocess/mel.py`
- `ppgs/phonemes.py`
- `ppgs/config/defaults.py`
- `ppgs/load.py`

The checkpoint is `taisa1/ppgs-ja/mel_ja.pt`, revision
`c77bee7a8110b12d1d4fc10989b457e185e2a3b3`, distributed with MIT metadata.
Weights are downloaded into the Hugging Face cache, not included in this repo.

The adapter preserves the model's parameter names and its 35 Japanese output
categories, including the final `<silent>` category. It handles unpadded
utterances up to the checkpoint's positional-encoding limit (5000 frames),
uses the upstream STFT/magnitude/log-Mel definitions, and preserves the two
FP16 frontend rounding operations. The Mel matrix multiplication and model
run in FP32, without automatic mixed precision. There is no causal mask and
no temporal interpolation or PPG sparsification.

The adapter does not import the upstream package or its optional training
dependencies. It does not use the English pretrained model or the upstream
English phoneme-similarity matrix.
