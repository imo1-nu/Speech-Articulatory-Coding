"""Train an unquantized trajectory AE with the frozen VQ experiment's data and architecture."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from .japanese_ppg import sha256_file
from .model_jsut_tokens import CachedWindowBatches, atomic_save, checked_feature, load_run, save_npz, write_json
from .trajectory_vq import WindowDataset, overlap_average


class ContinuousTrajectoryAE(nn.Module):
    def __init__(self, window_frames=2, latent_dim=16):
        super().__init__()
        self.window_frames=window_frames
        self.token_stride=1
        self.encoder=nn.Sequential(nn.Linear(window_frames*12,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,latent_dim))
        self.decoder=nn.Sequential(nn.Linear(latent_dim,64),nn.ReLU(),nn.Linear(64,128),nn.ReLU(),nn.Linear(128,window_frames*12))

    def forward(self,windows):
        return self.decoder(self.encoder(windows))


def infer_ae(model,ema,mean,std,batch_size=512,device='cpu'):
    dataset=WindowDataset([ema],mean,std,model.window_frames,1)
    windows=[]
    with torch.inference_mode():
        for start in range(0,len(dataset),batch_size):
            batch=torch.stack([dataset[i] for i in range(start,min(start+batch_size,len(dataset)))]).to(device)
            windows.append(model(batch).cpu().numpy().reshape(-1,model.window_frames,12))
    return overlap_average(np.concatenate(windows),output_length=len(ema))*std+mean


def ae_epoch(model,loader,optimizer=None):
    model.train(optimizer is not None)
    total,count=0.,0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            loss=F.mse_loss(model(batch),batch)
            if not torch.isfinite(loss):raise ValueError('Nonfinite AE loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            total+=float(loss.detach())*len(batch);count+=len(batch)
    return total/count


def load_ae(directory,device='cpu'):
    config=json.loads((directory/'config.json').read_text())
    ckpt=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
    model=ContinuousTrajectoryAE(config['window_frames'],config['latent_dim'])
    model.load_state_dict(ckpt['state_dict'],strict=True)
    return model.eval().to(device),ckpt['mean'].numpy(),ckpt['std'].numpy(),config


def train_ae(reference_dir,output_dir,device='cuda'):
    config,splits,_,mean,std=load_run(reference_dir,'cpu')
    reference_hash=sha256_file(reference_dir/'best.pt')
    if config['window_frames']!=2 or config.get('token_stride',1)!=1 or config['latent_dim']!=16:
        raise ValueError('Expected the existing 40 ms / 50 Hz / latent16 reference')
    if output_dir.exists() and any(output_dir.iterdir()):
        if not (output_dir/'training_metrics.json').is_file():raise ValueError('AE directory is incomplete; choose a fresh run directory')
        _,old_mean,old_std,old=load_ae(output_dir)
        if old['reference_checkpoint_sha256']!=reference_hash or json.loads((output_dir/'splits.json').read_text())!=splits:
            raise ValueError('AE run uses different source data/model')
        np.testing.assert_array_equal(mean,old_mean);np.testing.assert_array_equal(std,old_std)
        print(f'Reusing completed AE: {output_dir}',flush=True)
        return
    arrays={}
    for relative in tqdm([r for group in splits.values() for r in group],desc='Load AE data'):
        arrays[relative]=checked_feature(config,relative)['ema']
    datasets={name:WindowDataset([arrays[r] for r in relatives],mean,std,2,1) for name,relatives in splits.items()}
    seed=config['seed'];torch.manual_seed(seed);np.random.seed(seed)
    if torch.device(device).type=='cuda':torch.cuda.manual_seed_all(seed)
    model=ContinuousTrajectoryAE(2,16).to(device)
    loaders={name:CachedWindowBatches(data,config['batch_size'],device,name=='train',seed) for name,data in datasets.items()}
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    output_dir.mkdir(parents=True,exist_ok=True)
    ae_config={'model':'continuous_trajectory_ae','window_frames':2,'latent_dim':16,'frame_rate':50,'token_stride':1,
               'encoder_widths':[24,128,64,16],'decoder_widths':[16,64,128,24],
               'reference_run_dir':str(reference_dir.resolve()),'reference_checkpoint_sha256':reference_hash,
               'seed':seed,'batch_size':config['batch_size'],'learning_rate':config['learning_rate'],
               'max_epochs':config['max_epochs'],'patience':config['patience'],'minimum_improvement':1e-5,
               'objective':'normalized window MSE only, no quantization or codebook/commitment loss',
               'normalization':'copied exactly from VQ128 training statistics','device':str(device),
               'torch_version':torch.__version__}
    write_json(output_dir/'config.json',ae_config);write_json(output_dir/'splits.json',splits)
    save_npz(output_dir/'normalization.npz',mean=mean,std=std)
    best=float('inf');stale=0;best_epoch=0
    with (output_dir/'history.jsonl').open('w') as stream:
        for epoch in range(1,config['max_epochs']+1):
            training=ae_epoch(model,loaders['train'],optimizer);validation=ae_epoch(model,loaders['validation'])
            stream.write(json.dumps({'epoch':epoch,'train_window_mse':training,'validation_window_mse':validation})+'\n');stream.flush()
            print(f'AE epoch {epoch}: train={training:.7f} validation={validation:.7f}',flush=True)
            if validation<best-1e-5:
                best,best_epoch,stale=validation,epoch,0
                state={'state_dict':model.state_dict(),'mean':torch.from_numpy(mean),'std':torch.from_numpy(std),'epoch':epoch}
                atomic_save(output_dir/'best.pt',lambda f:torch.save(state,f))
            else:stale+=1
            if stale>=config['patience']:break
    model,_,_,_=load_ae(output_dir,device)
    test=ae_epoch(model,loaders['test'])
    write_json(output_dir/'training_metrics.json',{'best_epoch':best_epoch,'best_validation_window_mse':best,
               'test_window_mse':test,'split_utterances':{k:len(v) for k,v in splits.items()},
               'split_frames':{k:sum(len(arrays[r]) for r in v) for k,v in splits.items()}})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',type=Path,default=Path('outputs/jsut_window_sweep/window_040ms'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/jsut_reconstruction_ablation/continuous_ae'))
    parser.add_argument('--device',default='cuda');parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args(argv)
    if args.threads<1:parser.error('--threads must be positive')
    torch.set_num_threads(args.threads);train_ae(args.reference_dir,args.output_dir,args.device)


if __name__=='__main__':main()
