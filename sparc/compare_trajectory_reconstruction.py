"""Compare continuous AE and frozen VQ128/VQ512 against synthesis from original trajectories."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from . import evaluate_generated_audio as metrics
from .compare_jsut_audio import audio_player
from .continuous_trajectory_ae import infer_ae,load_ae,train_ae
from .japanese_ppg import JapanesePPGExtractor,PHONEMES,frame_times,sha256_file
from .local_ppg_search import load_generator
from .model_jsut_tokens import checked_feature,configure_plots,infer_utterance,load_run,save_npz,write_json
from .sequential_ppg_search import save_csv


METHODS=('vq128','vq512','continuous_ae')
VARIANTS=('original','original_ema',*METHODS)
LABELS={'original':'元音声','original_ema':'元軌道から合成','vq128':'VQ・128コード',
        'vq512':'VQ・512コード','continuous_ae':'連続AE（量子化なし）'}


def trajectory_statistics(original,reconstruction,std):
    source=original.astype(np.float64)/std
    reconstructed=reconstruction.astype(np.float64)/std
    delta_source=np.diff(source,axis=0);delta_reconstructed=np.diff(reconstructed,axis=0)
    frequency=np.fft.rfftfreq(len(source),1/50)
    mask=(frequency>6)&(frequency<=10)
    def band(values):return float((np.abs(np.fft.rfft(values-values.mean(0),axis=0))**2/len(values))[mask].sum())
    return {'frames':len(source),'delta_frames':max(0,len(source)-1),
            'squared_error':float(np.square(source-reconstructed).sum()),
            'delta_squared_error':float(np.square(delta_source-delta_reconstructed).sum()),
            'source_velocity':float(np.square(delta_source).sum()),
            'reconstructed_velocity':float(np.square(delta_reconstructed).sum()),
            'source_band_energy':band(source),'reconstructed_band_energy':band(reconstructed)}


def summarize_trajectories(totals):
    output={}
    for method,values in totals.items():
        v={key:sum(x[key] for x in values) for key in values[0]}
        output[method]={'frames':v['frames'],'normalized_mse':v['squared_error']/(v['frames']*12),
                        'delta_mse':v['delta_squared_error']/(v['delta_frames']*12) if v['delta_frames'] else None,
                        'velocity_energy_ratio':v['reconstructed_velocity']/v['source_velocity'] if v['source_velocity'] else None,
                        'trajectory_6_to_10_hz_energy_ratio':v['reconstructed_band_energy']/v['source_band_energy'] if v['source_band_energy'] else None}
    return output


def k_intervals(reference_ppg,maximum=3):
    indices=[PHONEMES.index('k'),PHONEMES.index('ky')]
    mask=np.isin(reference_ppg.argmax(1),indices)
    edges=np.flatnonzero(np.diff(np.r_[False,mask,False]))
    events=[(int(s),int(e)) for s,e in zip(edges[::2],edges[1::2])]
    # Selection uses the fixed reference only, never a candidate's reconstruction outcome.
    return sorted(sorted(events,key=lambda x:(-(x[1]-x[0]),x[0]))[:maximum])


def synthesize(generator,ema,fixed,gain,device):
    art=np.concatenate((ema,fixed['pitch'],fixed['loudness']),axis=1).astype(np.float32)
    with torch.inference_mode():
        audio=generator(torch.from_numpy(art)[None].to(device).transpose(1,2),
                        torch.from_numpy(fixed['spk_emb'])[None].to(device))[0,0]
        audio=torch.floor(audio*gain*32768).clamp(-32768,32767)/32768
    if not torch.isfinite(audio).all():raise ValueError('Nonfinite synthesis')
    return audio.cpu().numpy()


def compatible_runs(reference_dir,vq512_dir,ae_dir):
    ref=load_run(reference_dir,'cpu');other=load_run(vq512_dir,'cpu');ae=load_ae(ae_dir,'cpu')
    config,splits,_,mean,std=ref
    if config['window_frames']!=2 or config['codebook_size']!=128 or config.get('token_stride',1)!=1:
        raise ValueError('Expected the 40 ms / 128-code / 50 Hz reference')
    other_config,other_splits,_,other_mean,other_std=other
    if (other_config['window_frames']!=2 or other_config['codebook_size']!=512
            or other_config.get('token_stride',1)!=1 or other_config['latent_dim']!=config['latent_dim']
            or other_config['files']!=config['files'] or other_config['features_dir']!=config['features_dir']
            or splits!=other_splits):raise ValueError('VQ512 data/architecture do not match VQ128')
    for key in ['seed','batch_size','learning_rate','commitment','max_epochs','patience']:
        if other_config[key]!=config[key]:raise ValueError(f'VQ training setting differs: {key}')
    if json.loads((ae_dir/'splits.json').read_text())!=splits or ae[3]['reference_checkpoint_sha256']!=sha256_file(reference_dir/'best.pt'):
        raise ValueError('Continuous AE uses a different source experiment')
    for candidate_mean,candidate_std in [(other_mean,other_std),(ae[1],ae[2])]:
        np.testing.assert_array_equal(mean,candidate_mean);np.testing.assert_array_equal(std,candidate_std)
    return ref,other,ae


def plot_k(path,event,waves,trajectories,mean,std):
    import matplotlib.pyplot as plt
    s,e=event;start=max(0,s/100-.12);end=min(len(waves['original_ema'])/16000,e/100+.15)
    fig,axes=plt.subplots(8,1,figsize=(12,16),sharex=True)
    reference_magnitude=np.abs(librosa.stft(waves['original_ema'],n_fft=512,hop_length=80,win_length=400))
    reference_level=max(float(reference_magnitude.max()),1e-8)
    for axis,variant in zip(axes[:4],('original_ema',*METHODS)):
        magnitude=np.abs(librosa.stft(waves[variant],n_fft=512,hop_length=80,win_length=400))
        db=20*np.log10(np.maximum(magnitude,1e-8)/reference_level)
        axis.imshow(db,origin='lower',aspect='auto',extent=(-.0025,(db.shape[1]-.5)*.005,0,8000),
                    vmin=-80,vmax=0,cmap='magma',interpolation='nearest')
        axis.set(ylabel='Hz',title=variant)
    time=np.arange(len(trajectories['original_ema']))/50
    for offset,index in enumerate([0,1]):
        for variant in ('original_ema',*METHODS):
            values=(trajectories[variant][:,index]-mean[index])/std[index]
            axes[4+offset].plot(time,values,label=variant,linewidth=1)
            axes[6+offset].plot(time[1:],np.diff(values)*50,label=variant,linewidth=1)
        axes[4+offset].set_ylabel(['TDX','TDY'][offset]+' (SD)')
        axes[6+offset].set_ylabel('d'+['TDX','TDY'][offset]+'/dt (SD/s)')
    axes[4].legend(fontsize=8,ncol=4)
    for axis in axes:
        axis.axvspan(s/100,e/100,color='cyan',alpha=.14);axis.set_xlim(start,end)
    axes[-1].set_xlabel('Time (s), shaded: reference PPG k/ky candidate')
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


def write_report(output,manifest,audio_rows):
    aggregates=metrics.aggregate(audio_rows)
    table=''.join(f'<tr><td>{LABELS[r["condition"]]}</td><td>{100*r["cer"]:.1f}%</td>'
                  f'<td>{r["mcd_db"]:.4f}</td><td>{r["stoi"]:.4f}</td><td>{r["f0_rmse_hz"]:.2f}</td></tr>' for r in aggregates)
    trajectory=''.join(f'<tr><td>{LABELS[name]}</td><td>{r["normalized_mse"]:.7f}</td><td>{r["delta_mse"]:.7f}</td>'
                       f'<td>{r["velocity_energy_ratio"]:.4f}</td></tr>' for name,r in manifest['test_trajectory_metrics'].items())
    cards=[]
    for record in manifest['audio_records']:
        folder=record['directory']
        players=''.join(f'<div><h3>{LABELS[v]}</h3>{audio_player(folder+"/"+v+".wav")}</div>' for v in VARIANTS)
        import html
        transcripts=''.join(f'<li>{LABELS[r["condition"]]}：{html.escape(r["asr_text"])}（CER {100*r["cer"]:.1f}%）</li>'
                            for r in audio_rows if r['utterance_id']==record['utterance_id'])
        events=[]
        for event in record['k_events']:
            clips=''.join(f'<div><h3>{LABELS[v]}</h3>{audio_player(event["directory"]+"/"+v+".wav")}</div>' for v in VARIANTS)
            events.append(f'<details><summary>k/ky候補：{event["start_seconds"]:.2f}〜{event["end_seconds"]:.2f}秒</summary>'
                          f'<div class="players">{clips}</div><a href="{event["directory"]}/diagnostic.png">図を拡大</a>'
                          f'<img src="{event["directory"]}/diagnostic.png" loading="lazy"></details>')
        cards.append(f'<section><h2>{record["utterance_id"]}</h2><p>{html.escape(record["text"])}</p>'
                     f'<div class="players">{players}</div><ul>{transcripts}</ul>'+''.join(events)+'</section>')
    content='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>軌道再構成の切り分け</title>
<style>body{font-family:system-ui;max-width:1350px;margin:30px auto;padding:0 18px}.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}section{border:1px solid #ccc;margin:24px 0;padding:18px}audio,img{width:100%}td,th{padding:9px;border-bottom:1px solid #ddd}table{border-collapse:collapse}h3{font-size:1rem}details{margin:20px 0}</style>
<h1>40 ms・50 Hz：VQ128 / VQ512 / 連続AE</h1><p>連続AEは同じ24→128→64→16→64→128→24のMLPを、同じ4500/250/250分割・学習時正規化・Adam・学習率・バッチサイズ・最大epoch・早期停止で別途学習しました。
連続AEは量子化なしの参考で、同じビットレートの比較ではありません。既存VQの量子化を推論時だけ外す方法は使っていません。</p>
<p>軌道評価はテスト250発話、音声評価は既存の試聴5発話。元軌道合成をMCD・STOI・F0評価の参照とし、F0・音量・話者・ゲインを全再構成で固定します。
CERはWhisper large-v3による認識をJSUT正解文と比較し、NFKC・小文字化・句読点等除去のみを適用します。時間伸縮・DTW・候補ごとの音量正規化はしません。</p>
<p>カ行候補は元軌道合成のPPGでk/kyが最大の連続区間から、長い順に各発話最大3個を選びます。これは正解音素境界ではありません。
図のスペクトルは同じ参照レベル・色範囲、軌道と速度は訓練SDで標準化したTDX/TDYです。
PPGのk/ky確率と予測音素は診断用で、カ行の正解率そのものではありません。</p>
<p><a href="audio_metrics.csv">音声指標CSV</a> / <a href="audio_aggregates.csv">音声集計CSV</a> / <a href="k_diagnostics.csv">k/ky診断CSV</a> / <a href="manifest.json">設定</a></p>
<h2>テスト軌道・重複平均後</h2><table><tr><th>条件</th><th>標準化MSE ↓</th><th>差分MSE ↓</th><th>速度エネルギー比</th></tr>'''
    content+=trajectory+'</table><h2>音声評価・参照は元軌道合成</h2><table><tr><th>条件</th><th>CER ↓</th><th>MCD(dB) ↓</th><th>STOI ↑</th><th>F0 RMSE(Hz) ↓</th></tr>'+table+'</table>'+''.join(cards)
    content+="<script>document.querySelectorAll('audio').forEach(p=>p.onplay=()=>document.querySelectorAll('audio').forEach(q=>{if(q!==p)q.pause()}));</script></html>"
    (output/'index.html').write_text(content,encoding='utf-8')
    return aggregates


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir',type=Path,default=Path('outputs/jsut_window_sweep/window_040ms'))
    parser.add_argument('--vq512-dir',type=Path,default=Path('outputs/jsut_codebook_sweep/codebook_0512/window_040ms'))
    parser.add_argument('--ae-dir',type=Path,default=Path('outputs/jsut_reconstruction_ablation/continuous_ae'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/jsut_reconstruction_ablation/comparison'))
    parser.add_argument('--device',default='cuda');parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args(argv)
    if args.threads<1:parser.error('--threads must be positive')
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):parser.error('Choose a new comparison directory')
    torch.set_num_threads(args.threads)
    train_ae(args.reference_dir,args.ae_dir,args.device)
    ref,other,ae=compatible_runs(args.reference_dir,args.vq512_dir,args.ae_dir)
    config,splits,model128,mean,std=ref;model512=other[2];ae_model=ae[0]
    reference_hash=sha256_file(args.reference_dir/'best.pt')
    source=json.loads((args.reference_dir/'audio_comparison/manifest.json').read_text())
    if source['checkpoint_sha256']!=reference_hash:raise ValueError('Stale synthesis reference')
    records=source['records']
    if len(records)!=5 or any(r['feature_path'] not in splits['test'] or r['split']!='test' for r in records):
        raise ValueError('Expected the existing five listening test utterances')
    reconstruction_cache={};totals={name:[] for name in METHODS}
    for relative in tqdm(splits['test'],desc='Reconstruction test250'):
        data=checked_feature(config,relative);ema=data['ema'];reconstructions={}
        for name,model in [('vq128',model128),('vq512',model512)]:
            _,reconstructions[name],_=infer_utterance(model,ema,mean,std,512,torch.device('cpu'))
        reconstructions['continuous_ae']=infer_ae(ae_model,ema,mean,std)
        for name,reconstruction in reconstructions.items():totals[name].append(trajectory_statistics(ema,reconstruction,std))
        if relative in {r['feature_path'] for r in records}:reconstruction_cache[relative]=reconstructions
    args.output_dir.mkdir(parents=True,exist_ok=True);configure_plots()
    generator,generator_hash=load_generator(args.device);extractor=JapanesePPGExtractor(args.device)
    recognizer=metrics.WhisperRecognizer(args.device)
    manifest={'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'comparison_source_sha256':sha256_file(Path(__file__)),
              'reference_vq128_sha256':reference_hash,'vq512_sha256':sha256_file(args.vq512_dir/'best.pt'),
              'ae_sha256':sha256_file(args.ae_dir/'best.pt'),'generator_sha256':generator_hash,
              'ppg_sha256':extractor.checkpoint_sha256,'asr_model':metrics.ASR_MODEL,'asr_revision':metrics.ASR_REVISION,
              'asr_settings':{'language':'ja','task':'transcribe','num_beams':1,'do_sample':False,'max_new_tokens':256,'prompt':None},
              'packages':{name:importlib.metadata.version(name) for name in ['torch','numpy','pyworld','pysptk','pystoi','transformers']},
              'audio_metric_reference':'original trajectory synthesis','cer_reference':'JSUT gold text',
              'k_selection':'reference original-EMA synthesis PPG argmax k/ky, longest three intervals per utterance',
              'test_trajectory_metrics':summarize_trajectories(totals),'audio_records':[]}
    write_json(args.output_dir/'manifest.json',manifest)
    audio_rows=[];k_rows=[];asr_cache={}
    old_cache=Path('outputs/jsut_ppg_delay060_step020/audio_metrics/manifest.json')
    if old_cache.exists():
        cached=json.loads(old_cache.read_text())
        if (cached['asr_model_id']==metrics.ASR_MODEL and cached['asr_revision']==metrics.ASR_REVISION
                and all(cached.get(k)==v for k,v in manifest['asr_settings'].items())):
            asr_cache={r['audio_sha256']:r['asr_text'] for r in cached['records']}
    for record in records:
        directory=args.output_dir/Path(record['feature_path']).with_suffix('');directory.mkdir(parents=True)
        source_dir=args.reference_dir/'audio_comparison'/Path(record['feature_path']).with_suffix('')
        fixed=dict(np.load(source_dir/'conditions.npz'))
        original=checked_feature(config,record['feature_path'])['ema'];reconstructions=reconstruction_cache[record['feature_path']]
        waves={};ppgs={};trajectories={'original_ema':original,**reconstructions}
        shutil.copyfile(source_dir/'original.wav',directory/'original.wav')
        for variant in ('original_ema',*METHODS):
            wave=synthesize(generator,trajectories[variant],fixed,record['synthesis_common_gain'],args.device)
            sf.write(directory/f'{variant}.wav',wave,16000,subtype='PCM_16')
            reloaded,sr=sf.read(directory/f'{variant}.wav',dtype='float32')
            if sr!=16000 or not np.array_equal(wave,reloaded):raise ValueError('Saved PCM differs from synthesis')
            waves[variant]=wave;ppgs[variant]=extractor.extract(torch.from_numpy(wave[None,None]))
        baseline,sr=sf.read(source_dir/'original_ema.wav',dtype='float32')
        if sr!=16000 or baseline.shape!=waves['original_ema'].shape or np.max(abs(baseline-waves['original_ema']))>3/32768:
            raise ValueError('Original-EMA synthesis was not reproduced')
        old_vq,sr=sf.read(source_dir/'reconstructed_ema.wav',dtype='float32')
        if sr!=16000 or old_vq.shape!=waves['vq128'].shape or np.max(abs(old_vq-waves['vq128']))>3/32768:
            raise ValueError('VQ128 baseline was not reproduced')
        waves['original']=metrics.read_audio(directory/'original.wav')
        reference_features=metrics.world_features(waves['original_ema'])
        for variant in VARIANTS:
            digest=sha256_file(directory/f'{variant}.wav')
            text=asr_cache.get(digest)
            if text is None:text=recognizer.transcribe(waves[variant]);asr_cache[digest]=text
            row={'condition':variant,'utterance_id':record['utterance_id'],'audio_sha256':digest,
                 'reference_text':record['text'],'asr_text':text,**metrics.character_errors(record['text'],text),
                 **metrics.acoustic_scores(waves['original_ema'],waves[variant],reference_features)}
            audio_rows.append(row)
            print(record['utterance_id'],variant,f'CER={row["cer"]:.4f} MCD={row["mcd_db"]:.4f} STOI={row["stoi"]:.4f}',flush=True)
        events=[];k_indices=[PHONEMES.index('k'),PHONEMES.index('ky')]
        for index,(s,e) in enumerate(k_intervals(ppgs['original_ema'])):
            event_dir=directory/f'k_{index:02d}';event_dir.mkdir()
            lo=max(0,int((s/100-.12)*16000));hi=min(len(waves['original_ema']),int((e/100+.15)*16000))
            for variant in VARIANTS:sf.write(event_dir/f'{variant}.wav',waves[variant][lo:hi],16000,subtype='PCM_16')
            plot_k(event_dir/'diagnostic.png',(s,e),waves,trajectories,mean,std)
            events.append({'start_seconds':s/100,'end_seconds':e/100,'directory':event_dir.relative_to(args.output_dir).as_posix()})
            for variant in ('original_ema',*METHODS):
                posterior=ppgs[variant][s:e]
                k_rows.append({'utterance_id':record['utterance_id'],'event':index,'condition':variant,
                               'start_seconds':s/100,'end_seconds':e/100,
                               'mean_k_ky_probability':float(posterior[:,k_indices].sum(1).mean()),
                               'top_phone':PHONEMES[int(np.bincount(posterior.argmax(1),minlength=35).argmax())]})
        save_npz(directory/'reconstructions.npz',original_ema=original,**reconstructions,
                 time=fixed['time'],pitch=fixed['pitch'],loudness=fixed['loudness'],spk_emb=fixed['spk_emb'],
                 **{f'{v}_ppg':p for v,p in ppgs.items()})
        manifest['audio_records'].append({'utterance_id':record['utterance_id'],'text':record['text'],
                                          'directory':directory.relative_to(args.output_dir).as_posix(),'k_events':events,
                                          'original_ema_reproduction_max_error':float(np.max(abs(baseline-waves['original_ema']))),
                                          'vq128_reproduction_max_error':float(np.max(abs(old_vq-waves['vq128'])))})
        save_csv(args.output_dir/'audio_metrics.csv',audio_rows)
        if k_rows:save_csv(args.output_dir/'k_diagnostics.csv',k_rows)
        aggregates=write_report(args.output_dir,manifest,audio_rows);save_csv(args.output_dir/'audio_aggregates.csv',aggregates)
        manifest['audio_aggregates']=aggregates;write_json(args.output_dir/'manifest.json',manifest)
    print(f'Report: {args.output_dir}/index.html',flush=True)


if __name__=='__main__':main()
