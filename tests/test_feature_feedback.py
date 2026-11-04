import unittest
import tempfile
import csv
from pathlib import Path
from collections import OrderedDict
from types import SimpleNamespace
import numpy as np
import torch

from sparc.feedback_features import FeatureObjective,FeatureObserver
from sparc.feature_feedback_search import optimize,delayed
from sparc.sequential_ppg_search import stitch_prediction
from sparc.wavlm_feedback_experiment import save_feature_metrics


def objective(name='wavlm',dim=2):
    obj=FeatureObjective.__new__(FeatureObjective)
    obj.name,obj.dim,obj.rate,obj.minimum_samples=name,dim,50 if name=='wavlm' else 100,320 if name=='wavlm' else 433
    return obj


class FeatureFeedbackTest(unittest.TestCase):
    def test_mixed_offline_and_delayed_records_share_csv_schema(self):
        rows=[{'condition':'offline','seconds':1},
              {'condition':'delayed','seconds':2,'terminal_observation':{'used_for_control':False},
               'execution_sources':{'test':'hash'}}]
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'features.csv';save_feature_metrics(path,rows)
            with path.open() as stream:saved=list(csv.DictReader(stream))
            self.assertEqual(len(saved),2)
            self.assertEqual(list(saved[0]),['condition','seconds'])
            self.assertEqual(saved[1]['condition'],'delayed')

    def test_wavlm_preprocessing_and_fresh_extraction_bypass_cache(self):
        obj=objective('wavlm',1024);obj.device='cpu';obj.cache=OrderedDict();obj.calls=0;obj.seconds=0.
        captured=[]
        def model(tensor,**kwargs):
            captured.append(tensor.clone())
            count=(tensor.shape[1]-400)//320+1
            values=tensor[:,:count,None].repeat(1,1,1024)
            return SimpleNamespace(hidden_states=[None]*9+[values])
        obj.model=model
        audio=np.linspace(-.2,.4,640,dtype=np.float32)[None]
        first=obj.extract(audio)
        obj.extract(audio)
        self.assertEqual(len(captured),1)
        obj.extract(audio,fresh=True)
        self.assertEqual(len(captured),2)
        norm=audio.astype(np.float64);norm=((norm-norm.mean(1,keepdims=True))/norm.std(1,keepdims=True)).astype(np.float32)
        np.testing.assert_array_equal(captured[0].numpy(),np.pad(norm,((0,0),(160,160))))
        self.assertEqual(first.shape,(1,2,1024))

    def test_arrived_features_are_frozen_when_prefix_is_recomputed(self):
        obj=objective();observer=FeatureObserver(obj)
        predicted=np.array([[1.,0.],[1.,0.]])
        observer.update(np.array([[0.,1.]]),predicted,True)
        observer.update(np.array([[9.,9.],[.5,.5]]),predicted,True)
        np.testing.assert_array_equal(observer.observed[0],[0,1])
        np.testing.assert_array_equal(observer.observed[1],[.5,.5])

    def test_cosine_is_scale_invariant_and_zero_safe(self):
        obj=objective()
        np.testing.assert_allclose(obj.distances(np.eye(2),3*np.eye(2)),0)
        self.assertTrue(np.isfinite(obj.distances(np.zeros((1,2)),np.zeros((1,2)))).all())
        self.assertAlmostEqual(obj.times(1)[0],39.5/16000)
        self.assertAlmostEqual(objective('ppg').times(1)[0],.005)

    def test_signed_bias_moves_prediction_toward_observation_and_respects_rate(self):
        obj=objective();observer=FeatureObserver(obj)
        actual=np.array([[0.,1.]]);predicted=np.array([[1.,0.]])
        first=observer.update(actual,predicted,True)
        self.assertAlmostEqual(first['feedback_update_rate'],1-.75**.2)
        corrected=obj.corrected(predicted,observer.bias,.5)
        self.assertLess(obj.distances(actual,corrected)[0],obj.distances(actual,predicted)[0])
        nofb=FeatureObserver(obj);nofb.update(actual,predicted,False)
        np.testing.assert_array_equal(nofb.bias,[0,0])

    def test_coordinate_search_only_accepts_verified_improvements(self):
        obj=objective();initial=np.zeros((4,2),dtype=np.int64)
        target=np.tile([0.,1.],(4,1))
        def evaluate(sequences):
            value=sequences.sum(2)/2
            return np.zeros((len(sequences),1280)),np.stack((1-value,value),axis=-1)
        candidates=[[np.array([1]),np.array([0])] for _ in range(2)]
        result=optimize(initial,target,evaluate,obj,candidates,np.full((2,2),.5),.03,
                        passes=1,positions=4)
        np.testing.assert_array_equal(result['tokens'],np.ones((4,2)))
        self.assertLess(result['final_distance'],result['baseline_distance'])
        for item in result['trace']:self.assertLessEqual(item['objective'],item['previous_objective']+1e-12)
        np.testing.assert_array_equal(initial,np.zeros((4,2)))

    def test_batch_improvement_is_rejected_when_single_reextraction_disagrees(self):
        obj=objective();initial=np.zeros((3,1),dtype=np.int64)
        target=np.tile([0.,1.],(3,1));calls=[]
        def evaluate(sequences):
            values=sequences[:,:,0]
            return np.zeros((len(sequences),960)),np.stack((1-values,values),axis=-1)
        def verify(sequences):
            calls.append(1)
            return np.zeros((1,960)),np.tile([1.,0.],(1,3,1))
        evaluate.verify=verify
        result=optimize(initial,target,evaluate,obj,[[np.array([1]),np.array([0])]],
                        np.full((2,2),.5),0,passes=1,positions=3)
        self.assertTrue(calls)
        np.testing.assert_array_equal(result['tokens'],initial)

    def test_delayed_observer_never_reads_future_and_output_prefix_is_immutable(self):
        obj=objective()
        def extract(audio):
            means=audio.reshape(len(audio),-1,320).mean(2)
            return np.stack((1+means,1+means.mean(1,keepdims=True)+means*.1),axis=-1)
        obj.extract=extract
        class Evaluator:
            objective=obj
            candidates=[[np.array([1]),np.array([0])] for _ in range(2)]
            def at(self,end,context,continuation):
                def evaluate(sequences):
                    audio=np.repeat(.05+.1*sequences[:,:,0]+.02*sequences[:,:,1],320,axis=1).astype(np.float32)
                    audio=stitch_prediction(audio,context,continuation,160)
                    return audio,obj.extract(audio)
                return evaluate
        initial=np.zeros((9,2),dtype=np.int64)
        target=extract(np.full((1,9*320),.2,dtype=np.float32))[0]
        def run(changed):
            count=[0]
            def plant(chunk):
                result=chunk+(.1 if changed and count[0]>=2 else 0)
                count[0]+=1
                return result
            return delayed(initial,target,Evaluator(),np.full((2,2),.5),horizon_tokens=4,plant=plant)
        normal,changed=run(False),run(True)
        for a,b in zip(normal['trace'],changed['trace']):
            self.assertLessEqual(a['observer_input_samples'],a['information_cutoff_samples'])
            self.assertEqual(a['prefix_sha256_before'],a['retained_prefix_sha256_after'])
            if a['information_cutoff_samples']<=640:
                self.assertEqual(a['observer_input_sha256'],b['observer_input_sha256'])
                self.assertEqual(a['bias_before_norm'],b['bias_before_norm'])
        self.assertFalse(normal['terminal_observation']['used_for_control'])
        np.testing.assert_array_equal(initial,np.zeros((9,2)))


if __name__=='__main__':unittest.main()
