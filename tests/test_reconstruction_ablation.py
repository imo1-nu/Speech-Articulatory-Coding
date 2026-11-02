import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from sparc.continuous_trajectory_ae import ContinuousTrajectoryAE,infer_ae,load_ae,train_ae
from sparc.compare_trajectory_reconstruction import k_intervals,summarize_trajectories,trajectory_statistics
from sparc.japanese_ppg import PHONEMES,sha256_file
from sparc.trajectory_vq import CHANNELS,TrajectoryVQVAE


class ReconstructionAblationTest(unittest.TestCase):
    def test_continuous_ae_matches_mlp_shapes_without_codebook_or_quantization(self):
        torch.set_num_threads(1)
        model=ContinuousTrajectoryAE()
        vq=TrajectoryVQVAE(128,16,2)
        expected={k:v.shape for k,v in vq.state_dict().items() if not k.startswith('codebook.')}
        self.assertEqual({k:v.shape for k,v in model.state_dict().items()},expected)
        self.assertFalse(hasattr(model,'quantize'))
        batch=torch.randn(7,24);output=model(batch)
        self.assertEqual(output.shape,batch.shape)
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_inference_preserves_even_window_overlap_odd_lengths_and_endpoints(self):
        class Identity(nn.Module):
            window_frames=2
            def forward(self,values):return values
        for length in [1,2,5,6]:
            ema=np.arange(length*12,dtype=np.float32).reshape(length,12)
            mean=np.full(12,2,dtype=np.float32);std=np.full(12,2,dtype=np.float32)
            reconstructed=infer_ae(Identity(),ema,mean,std,batch_size=2)
            np.testing.assert_array_equal(reconstructed,ema)

    def test_k_events_come_from_reference_and_select_longest_in_time_order(self):
        labels=['a','a','k','k','ky','ky','ky','a','k','a','k','k','k','k']
        ppg=np.zeros((len(labels),35),dtype=np.float32)
        ppg[np.arange(len(labels)),[PHONEMES.index(p) for p in labels]]=1
        self.assertEqual(k_intervals(ppg,maximum=2),[(2,7),(10,14)])
        self.assertEqual(k_intervals(np.full((10,35),1/35,dtype=np.float32)),[])

    def test_trajectory_statistics_do_not_count_between_utterance_jumps(self):
        std=np.ones(12)
        first=np.zeros((2,12));second=np.full((2,12),100.)
        values=[trajectory_statistics(first,first+1,std),trajectory_statistics(second,second+1,std)]
        result=summarize_trajectories({'test':values})['test']
        self.assertEqual(result['normalized_mse'],1)
        self.assertEqual(result['delta_mse'],0)
        self.assertIsNone(result['velocity_energy_ratio'])

    def test_training_uses_frozen_reference_normalization_and_can_reuse_completed_run(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);reference=root/'vq128';reference.mkdir();features=root/'features';features.mkdir()
            splits={'train':['train.npz'],'validation':['validation.npz'],'test':['test.npz']}
            signatures={}
            for name,value in [('train',7),('validation',8),('test',1000)]:
                path=features/f'{name}.npz'
                np.savez(path,ema=np.full((8,12),value,dtype=np.float32),time=np.arange(8)/50,
                         frame_rate=np.float64(50),channels=np.asarray(CHANNELS))
                stat=path.stat();signatures[path.name]={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'frames':8}
            config={'window_frames':2,'codebook_size':128,'latent_dim':16,'token_stride':1,
                    'features_dir':str(features),'files':signatures,'seed':42,'batch_size':4,
                    'learning_rate':.001,'max_epochs':2,'patience':1}
            (reference/'config.json').write_text(json.dumps(config));(reference/'splits.json').write_text(json.dumps(splits))
            mean=torch.full((12,),7.);std=torch.full((12,),3.)
            torch.save({'state_dict':TrajectoryVQVAE(128,16,2).state_dict(),'mean':mean,'std':std},reference/'best.pt')
            original_hash=sha256_file(reference/'best.pt')
            output=root/'ae';train_ae(reference,output,'cpu')
            model,saved_mean,saved_std,ae_config=load_ae(output)
            np.testing.assert_array_equal(saved_mean,mean.numpy());np.testing.assert_array_equal(saved_std,std.numpy())
            self.assertEqual(json.loads((output/'splits.json').read_text()),splits)
            self.assertEqual(ae_config['reference_checkpoint_sha256'],original_hash)
            self.assertEqual(sha256_file(reference/'best.pt'),original_hash)
            trained_hash=sha256_file(output/'best.pt');train_ae(reference,output,'cpu')
            self.assertEqual(sha256_file(output/'best.pt'),trained_hash)
            incomplete=root/'incomplete';incomplete.mkdir();(incomplete/'marker').write_text('keep')
            with self.assertRaises(ValueError):train_ae(reference,incomplete,'cpu')


if __name__=='__main__':unittest.main()
