import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sparc.compare_held_start_features import COMMON, validate_pair, numeric_rows
from sparc.japanese_ppg import sha256_file


class HeldStartComparisonTest(unittest.TestCase):
    def make_pair(self,root):
        directories={k:root/k for k in ('wavlm','ppg')};manifests={}
        for kind,directory in directories.items():
            (directory/'u').mkdir(parents=True)
            np.savez(directory/'u/initialization.npz',**{k:np.zeros((2,2)) for k in
                ('first_position','held_ema','held_rvq_ema','initial_tokens','original_ema','pitch','loudness','spk_emb')})
            records=[]
            for condition in COMMON:
                path=directory/'u'/f'{condition}.wav';path.write_bytes(b'identical waveform')
                records.append({'utterance_id':'u','condition':condition,'audio':f'u/{condition}.wav','audio_sha256':sha256_file(path)})
            manifests[kind]={'complete':True,'settings':{'objective':kind,'initial_position_frame':0,'horizon_ms':400,
                'step_ms':20,'delay_ms':60,'transition_weight':.03},'generator_sha256':'generator',
                'dependencies':{'model/codebooks.npz':'books','prior/transitions.npz':'prior'},
                'sources':{s:'same' for s in ('feature_feedback_search.py','feedback_features.py','trajectory_rvq.py','local_ppg_search.py')},
                'inputs':[{'utterance_id':'u','directory':'u','text':'text'}],'records':records}
        return directories,manifests

    def test_pair_rejects_changed_settings_initialization_or_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            directories,manifests=self.make_pair(Path(temporary))
            check=validate_pair(directories,manifests)
            self.assertEqual(check['shared_waveform_pairs_byte_identical'],6)
            changed=copy.deepcopy(manifests);changed['ppg']['settings']['delay_ms']=40
            with self.assertRaises(ValueError):validate_pair(directories,changed)
            path=directories['ppg']/'u/initialization.npz'
            arrays=dict(np.load(path));arrays['initial_tokens'][1,1]=1;np.savez(path,**arrays)
            with self.assertRaises(AssertionError):validate_pair(directories,manifests)
            arrays['initial_tokens'][:]=0;np.savez(path,**arrays)
            (directories['ppg']/'u/unedited.wav').write_bytes(b'changed output')
            with self.assertRaises(ValueError):validate_pair(directories,manifests)

    def test_numeric_parsing_preserves_numeric_asr_text(self):
        row=numeric_rows([{'condition':'ppg_feedback','asr_text':'123','normalized_hypothesis':'123',
                          'normalized_reference':'456','cer':'0.5','f0_rmse_hz':''}])[0]
        self.assertEqual(row['normalized_hypothesis'],'123')
        self.assertEqual(row['asr_text'],'123')
        self.assertEqual(row['cer'],.5)
        self.assertIsNone(row['f0_rmse_hz'])


if __name__=='__main__':unittest.main()
