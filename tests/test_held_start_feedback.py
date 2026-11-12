import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from sparc.held_start_feedback import held_initialization, audit_control, write_report
from sparc.feature_feedback_search import delayed
from sparc.feedback_features import FeatureObjective


class HeldStartTest(unittest.TestCase):
    def test_initialization_encodes_only_first_position_and_repeats_all_stages(self):
        captured = []
        def encoder(x):
            captured.append(x.numpy().copy())
            return x[:,:3]
        def quantize(x):return x,torch.tensor([[2,4,7]])
        model = SimpleNamespace(window_frames=2,ae=SimpleNamespace(encoder=encoder),quantize=quantize)
        first = np.arange(12,dtype=np.float32);mean = np.ones(12,dtype=np.float32);std = np.full(12,2,dtype=np.float32)
        held,tokens = held_initialization(first,10,model,mean,std,'cpu')
        np.testing.assert_array_equal(captured[0],np.tile((first-mean)/std,2)[None])
        np.testing.assert_array_equal(held,np.repeat(first[None],10,axis=0))
        np.testing.assert_array_equal(tokens,np.tile([2,4,7],(10,1)))
        self.assertEqual(tokens.dtype,np.int64)
        np.testing.assert_array_equal(first,np.arange(12,dtype=np.float32))
        with self.assertRaises(ValueError):held_initialization(first,1,model,mean,std,'cpu')

    def test_replay_checks_delay_frozen_output_and_edit_history(self):
        obj = FeatureObjective.__new__(FeatureObjective)
        obj.name,obj.dim,obj.rate,obj.minimum_samples = 'wavlm',2,50,320
        def extract(audio):
            means = audio.reshape(len(audio),-1,320).mean(2)
            return np.stack((1+means,1-means),axis=-1)
        obj.extract = extract
        class Evaluator:
            objective = obj
            candidates = [[np.array([1]),np.array([0])]]
            def at(self,end,context=None,continuation=None):
                def evaluate(tokens):
                    audio = np.repeat(tokens[:,:,0].astype(np.float32)*.1,320,axis=1)
                    if context is not None:audio[:,:len(context)]=context
                    return audio,extract(audio)
                return evaluate
        initial = np.zeros((8,1),dtype=np.int64);target = np.tile([1.1,.9],(8,1))
        result = delayed(initial,target,Evaluator(),np.full((2,2),.5),mode='no_feedback',horizon_tokens=4)
        audit = audit_control(result,initial,'no_feedback')
        self.assertEqual(audit['decisions_checked'],8)
        self.assertGreater(audit['accepted_edits_replayed'],0)
        result['biases'][:]=.8
        audit_control(result,initial,'feedback','ppg')
        with self.assertRaises(ValueError):audit_control(result,initial,'feedback','wavlm')
        with self.assertRaises(ValueError):audit_control(result,initial,'no_feedback','ppg')
        result['biases'][:]=1.01
        with self.assertRaises(ValueError):audit_control(result,initial,'feedback','ppg')
        result['biases'][:]=0
        result['trace'][4]['information_cutoff_samples']+=320
        with self.assertRaises(ValueError):audit_control(result,initial,'no_feedback')

    def test_ppg_report_uses_js_and_wavlm_report_remains_compatible(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)
            for settings,key,label in [({'objective':'ppg'},'ppg_js','PPG JS'),({},'wavlm_distance','WavLM距離')]:
                manifest={'settings':settings,'inputs':[],'records':[{'condition':'feedback',key:.12}]}
                write_report(path,manifest,[])
                text=(path/'index.html').read_text()
                self.assertIn(label,text)
                self.assertIn('0.12000',text)
                if key=='ppg_js':
                    self.assertNotIn('WavLM',text)
                    self.assertIn('日本語35音素PPG',text)


if __name__ == '__main__':unittest.main()
