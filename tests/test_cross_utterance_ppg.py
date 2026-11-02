import unittest

import numpy as np
import torch

from sparc.cross_utterance_ppg import aggregate_control, pair_inputs, parse_pairs, retime_source
from sparc.evaluate_cross_utterance import aggregate
from sparc.trajectory_vq import TrajectoryVQVAE


class CrossUtteranceTest(unittest.TestCase):
    def test_pair_ids_are_directional_unique_and_nonidentical(self):
        self.assertEqual(parse_pairs(['A:B','B:A']),[('A','B'),('B','A')])
        for values in [['A:A'],['A:B','A:B'],['A'],['A:B:C'],[':B']]:
            with self.subTest(values=values),self.assertRaises(ValueError):parse_pairs(values)

    def test_only_continuous_source_features_are_retimed_with_endpoints_preserved(self):
        source=np.array([[0.,10.],[1.,20.],[2.,30.]],dtype=np.float32)
        retimed=retime_source(source,5)
        np.testing.assert_array_equal(retimed[:,0],[0,.5,1,1.5,2])
        np.testing.assert_array_equal(retimed[[0,-1]],source[[0,-1]])
        np.testing.assert_array_equal(source,[[0,10],[1,20],[2,30]])
        same=retime_source(source,3);same[0,0]=10
        self.assertEqual(source[0,0],0)

    def test_target_articulation_pitch_and_speaker_cannot_leak_into_search_initialization(self):
        torch.set_num_threads(1);torch.manual_seed(42)
        model=TrajectoryVQVAE(8,4,2).eval()
        source={'ema':np.arange(60,dtype=np.float32).reshape(5,12)/20,
                'pitch':np.arange(5,dtype=np.float32)[:,None]+200,
                'loudness':np.arange(5,dtype=np.float32)[:,None]/10,
                'spk_emb':np.arange(64,dtype=np.float32)}
        target={'ema':np.ones((7,12),dtype=np.float32),
                'pitch':np.full((7,1),999,dtype=np.float32),
                'loudness':np.full((7,1),999,dtype=np.float32),'spk_emb':np.ones(64,dtype=np.float32)}
        ppg=np.full((15,35),1/35,dtype=np.float32)
        first=pair_inputs(source,target,ppg,model,np.zeros(12,dtype=np.float32),np.ones(12,dtype=np.float32))
        changed={k:v*1000 for k,v in target.items()}
        second=pair_inputs(source,changed,ppg,model,np.zeros(12,dtype=np.float32),np.ones(12,dtype=np.float32))
        np.testing.assert_array_equal(first['initial_tokens'],second['initial_tokens'])
        np.testing.assert_array_equal(first['source_ema_retimed'],second['source_ema_retimed'])
        np.testing.assert_array_equal(first['target_ppg'],ppg[:14])
        self.assertEqual(first['initial_tokens'].shape,(7,))
        self.assertTrue(np.issubdtype(first['initial_tokens'].dtype,np.integer))
        np.testing.assert_array_equal(first['fixed']['spk_emb'],source['spk_emb'])
        np.testing.assert_array_equal(first['fixed']['pitch'][[0,-1]],source['pitch'][[0,-1]])

    def test_cross_cer_uses_target_and_source_references_separately(self):
        rows=[]
        for b_chars,b_errors,a_chars,a_errors in [(10,2,20,10),(30,6,10,9)]:
            rows.append({'condition':'test','character_errors':b_errors,'reference_characters':b_chars,
                         'source_character_errors':a_errors,'source_reference_characters':a_chars,
                         'mcd_db':2.,'mcd_speech_frames':10,'stoi':.7,'f0_rmse_hz':5.,'f0_rmse_cents':10.,
                         'both_voiced_frames':10,'vuv_error_rate':.1,'acoustic_frames':10,
                         'ppg_js_to_B':.2,'ppg_frames':20})
        result=aggregate(rows)[0]
        self.assertAlmostEqual(result['cer_B'],.2)
        self.assertAlmostEqual(result['cer_A'],19/30)

    def test_control_aggregates_weight_ppg_frames_and_allow_reference_without_tokens(self):
        rows=[{'condition':'test','ppg_js_to_B':.1,'ppg_js_to_source_initial':.2,'ppg_frames':10,
               'transition_nll':None,'edges':4,'token_count':5,'edited_tokens':0,'evaluations':0},
              {'condition':'test','ppg_js_to_B':.3,'ppg_js_to_source_initial':.4,'ppg_frames':30,
               'transition_nll':None,'edges':14,'token_count':15,'edited_tokens':0,'evaluations':0}]
        result=aggregate_control(rows)[0]
        self.assertAlmostEqual(result['ppg_js_to_B'],.25)
        self.assertIsNone(result['transition_nll'])


if __name__=='__main__':unittest.main()
