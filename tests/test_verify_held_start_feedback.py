import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from sparc import verify_held_start_feedback as verifier
from sparc.japanese_ppg import sha256_file


class HeldStartVerificationTest(unittest.TestCase):
    def test_ppg_distance_key_survives_phonation_field_loop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run=root/'run';folder=run/'u';folder.mkdir(parents=True)
            fixed={'pitch':np.zeros((2,1)),'loudness':np.zeros((2,1)),'spk_emb':np.zeros(4)}
            held=np.zeros((2,12),dtype=np.float32);tokens=np.zeros((2,8),dtype=np.int64)
            np.savez(folder/'initialization.npz',first_position=held[0],held_ema=held,initial_tokens=tokens,original_ema=held,**fixed)
            values=np.zeros((2,35),dtype=np.float32);np.savez(folder/'target.npz',features=values)
            sf.write(folder/'original.wav',np.zeros(640),16000)
            reference=root/'reference/audio_comparison/u';reference.mkdir(parents=True)
            np.savez(reference/'conditions.npz',**fixed)
            model_dir=root/'outputs/jsut_rvq/model';model_dir.mkdir(parents=True)
            (model_dir/'config.json').write_text(json.dumps({'reference_dir':str(root/'reference')}))
            metadata={'checkpoint_sha256':'fake'}
            manifest={'complete':True,'sources':{},'dependencies':{},'settings':{'objective':'ppg'},
                      'feature':metadata,'inputs':[{'utterance_id':'u','directory':'u'}],
                      'records':[{'utterance_id':'u','condition':'original','audio':'u/original.wav',
                                  'audio_sha256':sha256_file(folder/'original.wav'),'ppg_js':0.}]}
            (run/'manifest.json').write_text(json.dumps(manifest))
            objective=SimpleNamespace(metadata=metadata,extract=lambda audio,**kwargs:values[None],
                                      distances=lambda target,predicted:np.zeros(len(target)))
            previous=Path.cwd()
            try:
                os.chdir(root)
                with patch.object(verifier,'load_rvq',return_value=(None,np.zeros(12),np.ones(12))), \
                     patch.object(verifier,'held_initialization',return_value=(held,tokens)), \
                     patch.object(verifier,'FeatureObjective',return_value=objective), \
                     patch('sys.argv',['verify','--run-dir',str(run),'--device','cpu']),contextlib.redirect_stdout(io.StringIO()):
                    verifier.main()
            finally:os.chdir(previous)
            result=json.loads((run/'verification.json').read_text())
            self.assertEqual(result['waveforms_checked'],1)
            self.assertEqual(result['maximum_ppg_js_error'],0.)
            self.assertTrue(result['fixed_phonation_conditions_unchanged'])


if __name__=='__main__':unittest.main()
