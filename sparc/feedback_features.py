"""Pinned WavLM and PPG feature objectives, with their own time grids and corrections."""

import hashlib
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

from .japanese_ppg import JapanesePPGExtractor, frame_times, js_divergence, sha256_file

WAVLM_MODEL = 'microsoft/wavlm-large'
WAVLM_REVISION = 'c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c'


def unit(values):
    values = np.asarray(values,dtype=np.float64)
    return values/np.maximum(np.linalg.norm(values,axis=-1,keepdims=True),1e-8)


class FeatureObjective:
    def __init__(self,name,device='cuda'):
        self.name,self.device = name,device
        self.calls,self.seconds = 0,0.
        self.cache = OrderedDict()
        if name == 'wavlm':
            from transformers import WavLMModel
            path = Path(snapshot_download(WAVLM_MODEL,revision=WAVLM_REVISION,
                                         allow_patterns=['config.json','pytorch_model.bin']))
            self.model = WavLMModel.from_pretrained(path).float()
            self.model.encoder.layers = self.model.encoder.layers[:10]
            self.model = self.model.eval().requires_grad_(False).to(device)
            self.dim,self.rate,self.minimum_samples = 1024,50,320
            self.metadata = {'model':WAVLM_MODEL,'revision':WAVLM_REVISION,'weights_sha256':sha256_file(path/'pytorch_model.bin'),
                             'hidden_state_index':9,'lowpass_hz':None,'normalize_waveform':True,'zero_pad_samples':160,
                             'metric':'mean framewise cosine distance','precision':'float32','waveform_statistics':'float64 population mean/std, then float32',
                             'frame_stride_samples':320,
                             'frame_center_offset_samples':39.5,'receptive_field_samples':400}
        elif name == 'ppg':
            self.extractor = JapanesePPGExtractor(device)
            self.dim,self.rate,self.minimum_samples = 35,100,433
            self.metadata = {'checkpoint_sha256':self.extractor.checkpoint_sha256,'metric':'mean Jensen-Shannon divergence',
                             'frame_stride_samples':160,'frame_center_offset_samples':80}
        else: raise ValueError('Unknown feature objective')

    def times(self,count):
        return (np.arange(count)*320+39.5)/16000 if self.name=='wavlm' else frame_times(count)

    def extract(self,audio,*,fresh=False):
        """Equal-length (B,N) waveforms; no padding other utterances into a batch."""
        audio = np.asarray(audio,dtype=np.float32)
        if audio.ndim!=2 or not np.isfinite(audio).all() or audio.shape[1]<self.minimum_samples:
            raise ValueError('Invalid or too short feature input')
        keys = [hashlib.sha256(row.tobytes()).hexdigest() for row in audio]
        if fresh:
            for key in keys:self.cache.pop(key,None)
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        if missing:
            tick = time.monotonic()
            indices = [keys.index(key) for key in missing]
            selected=audio[indices]
            if self.name=='wavlm':
                selected=selected.astype(np.float64)
                selected=((selected-selected.mean(1,keepdims=True))/selected.std(1,keepdims=True).clip(min=1e-7)).astype(np.float32)
            tensor = torch.from_numpy(selected).to(self.device)
            with torch.inference_mode():
                if self.name=='wavlm':
                    tensor = torch.nn.functional.pad(tensor,(160,160))
                    values = self.model(tensor,attention_mask=torch.ones_like(tensor,dtype=torch.long),
                                        output_hidden_states=True).hidden_states[9].cpu().numpy()
                else:
                    values = self.extractor.model(self.extractor.frontend(tensor[:,None])).softmax(1).transpose(1,2).cpu().numpy()
            if not np.isfinite(values).all(): raise ValueError('Nonfinite feature output')
            for key,value in zip(missing,values): self.cache[key] = value
            self.calls += len(missing);self.seconds += time.monotonic()-tick
        result = np.stack([self.cache[key] for key in keys])
        for key in keys: self.cache.move_to_end(key)
        while len(self.cache)>256: self.cache.popitem(last=False)
        return result

    def corrected(self,values,bias,gain):
        if self.name=='wavlm': return unit(unit(values)+gain*bias)
        if not gain or not np.any(bias): return values
        logits = np.log(np.maximum(values,1e-12))+gain*bias
        logits -= logits.max(axis=-1,keepdims=True)
        result = np.exp(logits)
        return result/result.sum(axis=-1,keepdims=True)

    def distances(self,target,prediction):
        if target.shape!=prediction.shape or target.ndim!=2 or target.shape[1]!=self.dim:
            raise ValueError('Expected aligned feature matrices')
        if self.name=='ppg': return js_divergence(target,prediction)
        return np.clip(1-np.sum(unit(target)*unit(prediction),axis=1),0,2)

    def innovation(self,actual,predicted):
        if self.name=='ppg':
            return np.clip(np.log(actual.mean(0)+.001)-np.log(predicted.mean(0)+.001),-1,1)
        delta = (unit(actual)-unit(predicted)).mean(0)
        return delta/max(1,float(np.linalg.norm(delta)))


class FeatureObserver:
    def __init__(self,objective):
        self.objective = objective
        self.observed = np.empty((0,objective.dim),dtype=np.float32)
        self.bias = np.zeros(objective.dim,dtype=np.float64)

    def update(self,prefix,forecasts,feedback):
        start,stop = len(self.observed),len(prefix)
        if stop<start or stop>len(forecasts): raise ValueError('Invalid observation availability')
        innovation = np.zeros_like(self.bias);rate=0.
        if stop>start:
            innovation = self.objective.innovation(prefix[start:stop],forecasts[start:stop])
            rate = 1-.75**((stop-start)/(.1*self.objective.rate))
            if feedback:
                self.bias = (1-rate)*self.bias+rate*innovation
                if self.objective.name=='ppg': self.bias = self.bias.clip(-1,1)
                else: self.bias /= max(1,float(np.linalg.norm(self.bias)))
            self.observed = np.concatenate((self.observed,prefix[start:stop]))
        return {'observed_from_frame':start,'observed_until_frame':stop,'innovation_norm':float(np.linalg.norm(innovation)),
                'feedback_update_rate':rate,'bias_norm':float(np.linalg.norm(self.bias))}
