import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

from sparc.continuous_trajectory_ae import ContinuousTrajectoryAE, infer_ae
from sparc.trajectory_rvq import ResidualTrajectoryAE, train_rvq
from sparc.japanese_ppg import sha256_file


class ResidualQuantizerTest(unittest.TestCase):
    def test_later_stage_quantizes_residual_and_decodes_sum(self):
        ae = ContinuousTrajectoryAE(latent_dim=2)
        books = np.array([[[0,0],[2,0]], [[0,0],[0,1]]],dtype=np.float32)
        model = ResidualTrajectoryAE(ae, books)
        latent = torch.tensor([[1.9,.9],[.1,.1]])
        quantized,tokens = model.quantize(latent)
        torch.testing.assert_close(quantized,torch.tensor([[2.,1.],[0.,0.]]))
        torch.testing.assert_close(tokens,torch.tensor([[1,1],[0,0]]))
        # Token IDs alone must reconstruct the identical latent sum.
        decoded = sum(model.codebooks[i][tokens[:,i]] for i in range(2))
        torch.testing.assert_close(decoded,quantized)
        torch.testing.assert_close(model.decode_tokens(tokens),ae.decoder(quantized))
        with self.assertRaises(ValueError): model.decode_tokens(tokens[:,:1])
        with self.assertRaises(ValueError): model.decode_tokens(tokens+2)

    def test_stage_prefixes_share_tokens_and_ae_is_unchanged(self):
        torch.manual_seed(42)
        ae = ContinuousTrajectoryAE()
        snapshot = {k:v.clone() for k,v in ae.state_dict().items()}
        books = torch.randn(4,8,16)
        full = ResidualTrajectoryAE(ae,books)
        short = ResidualTrajectoryAE(ae,books,2)
        latent = torch.randn(11,16)
        self.assertTrue(torch.equal(full.quantize(latent)[1][:,:2],short.quantize(latent)[1]))
        ema = np.random.default_rng(42).normal(size=(7,12)).astype(np.float32)
        reconstructed = infer_ae(full,ema,np.zeros(12),np.ones(12),batch_size=3)
        self.assertEqual(reconstructed.shape,ema.shape)
        self.assertTrue(np.isfinite(reconstructed).all())
        for k,v in ae.state_dict().items(): torch.testing.assert_close(v,snapshot[k],rtol=0,atol=0)

    def test_invalid_stage_counts_rejected(self):
        for count in (0,3):
            with self.assertRaises(ValueError):
                ResidualTrajectoryAE(ContinuousTrajectoryAE(),torch.zeros(2,4,16),count)

    def test_codebook_fit_uses_only_train_features_and_preserves_frozen_ae(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); reference = root/'ref'; reference.mkdir()
            ae_dir = root/'ae'; ae_dir.mkdir()
            (reference/'best.pt').write_bytes(b'reference')
            (ae_dir/'best.pt').write_bytes(b'ae')
            splits = {'train':['train.npz'],'validation':['val.npz'],'test':['test.npz']}
            (ae_dir/'splits.json').write_text(json.dumps(splits))
            ae = ContinuousTrajectoryAE(); snapshot = {k:v.clone() for k,v in ae.state_dict().items()}
            mean = np.zeros(12,dtype=np.float32); std = np.ones(12,dtype=np.float32)
            values = np.random.default_rng(42).normal(size=(40,12)).astype(np.float32)
            with patch('sparc.trajectory_rvq.load_run',return_value=({},splits,None,mean,std)), \
                 patch('sparc.trajectory_rvq.load_ae',return_value=(ae,mean,std,{'reference_checkpoint_sha256':sha256_file(reference/'best.pt')})), \
                 patch('sparc.trajectory_rvq.checked_feature',return_value={'ema':values}) as features:
                output = root/'rvq'
                train_rvq(reference,ae_dir,output,stages=2,size=2)
                self.assertEqual([c.args[1] for c in features.call_args_list],['train.npz'])
                self.assertEqual(np.load(output/'codebooks.npz')['codebooks'].shape,(2,2,16))
                features.reset_mock()
                train_rvq(reference,ae_dir,output,stages=2,size=2)
                features.assert_not_called()
            for k,v in ae.state_dict().items(): torch.testing.assert_close(v,snapshot[k],rtol=0,atol=0)


if __name__ == '__main__': unittest.main()
