"""Time-aligned RVQ coordinate search and immutable delayed feature feedback."""

import time
import numpy as np

from .feedback_features import FeatureObserver
from .local_ppg_search import candidate_codes, transition_nll
from .rvq_transition_search import RVQEvaluator
from .sequential_ppg_search import audio_hash, pcm16, stitch_prediction


class AcousticEvaluator:
    def __init__(self,model,mean,std,generator,objective,fixed,gain,device):
        self.model,self.mean,self.std = model,mean,std
        self.generator,self.objective,self.fixed,self.gain,self.device = generator,objective,fixed,gain,device
        books = model.codebooks.cpu().numpy()
        self.candidates = [[candidate_codes(k,book[:,None,:],np.arange(len(book)),12,4)
                            for k in range(len(book))] for book in books]
        self.synthesis_seconds=0.;self.synthesis_candidates=0

    def at(self,end,context=None,continuation=None):
        fixed = {k:v[:end] if k in ('pitch','loudness') else v for k,v in self.fixed.items()}
        base = RVQEvaluator(self.model,self.mean,self.std,self.generator,None,fixed,self.gain,self.device)
        def evaluate(sequences,*,fresh=False):
            if sequences.shape[1]!=end: raise ValueError('Candidate extends beyond horizon')
            tick=time.monotonic()
            audio = base.synthesize(sequences)[:,0].cpu().numpy()
            if context is not None: audio=stitch_prediction(audio,context,continuation,160)
            self.synthesis_seconds+=time.monotonic()-tick;self.synthesis_candidates+=len(sequences)
            return audio,self.objective.extract(audio,fresh=fresh)
        evaluate.verify=lambda sequences:evaluate(sequences,fresh=True)
        return evaluate


def score(objective,target,features,tokens,start,probabilities,weight,bias,gain):
    stop=min(len(target),len(features))
    times=objective.times(stop)
    mask=times>=start*.02-1e-12
    if not mask.any(): raise ValueError('No future feature frames to score')
    left=int(np.searchsorted(times,start*.02-.05))
    prediction=objective.corrected(features[left:stop],bias,gain)
    distances=np.zeros(stop,dtype=np.float64)
    distances[left:]=objective.distances(target[left:stop],prediction)
    value=float(distances[mask].mean())
    nll=transition_nll(tokens[max(0,start-1):,0],probabilities)
    return value+weight*nll,value,nll,distances,times


def optimize(initial,target,evaluate,objective,candidates,probabilities,weight,*,start=0,passes=2,
             positions=24,batch_size=4,include_current=False,bias=None,gain=0,on_step=None):
    tokens=np.asarray(initial,dtype=np.int64).copy()
    if tokens.ndim!=2 or start<0 or start>=len(tokens) or passes<1 or positions<1 or batch_size<1:
        raise ValueError('Invalid search inputs')
    if not np.isfinite(weight) or weight<0: raise ValueError('Invalid transition weight')
    bias=np.zeros(objective.dim) if bias is None else bias
    audio,features=evaluate(tokens[None]);audio,features=audio[0],features[0]
    initial_features=features.copy()
    current=score(objective,target,features,tokens,start,probabilities,weight,bias,gain)
    baseline=current;trace=[];evaluations=1
    for pass_index in range(passes):
        accepted=0
        for stage in range(tokens.shape[1]):
            priorities=[]
            for i in range(start,len(tokens)):
                mask=(current[4]>=i*.02-.05)&(current[4]<i*.02+.09)
                priorities.append(float(current[3][mask].mean()) if mask.any() else 0.)
            selected=np.argsort(-np.asarray(priorities),kind='stable')[:positions]+start
            if include_current: selected=np.asarray([start,*[int(i) for i in selected if i!=start]])[:positions]
            for position in selected:
                old=int(tokens[position,stage]);best_code=old;best=current[0]
                codes=candidates[stage][old]
                for offset in range(0,len(codes),batch_size):
                    group=codes[offset:offset+batch_size]
                    proposals=np.repeat(tokens[None],len(group),axis=0);proposals[:,position,stage]=group
                    _,values=evaluate(proposals);evaluations+=len(group)
                    for code,proposal,value in zip(group,proposals,values):
                        trial=score(objective,target,value,proposal,start,probabilities,weight,bias,gain)
                        if trial[0]<best-1e-5: best,best_code=trial[0],int(code)
                previous=current[0]
                if best_code!=old:
                    proposal=tokens.copy();proposal[position,stage]=best_code
                    wave,value=getattr(evaluate,'verify',evaluate)(proposal[None]);evaluations+=1
                    verified=score(objective,target,value[0],proposal,start,probabilities,weight,bias,gain)
                    if verified[0]<current[0]-1e-5:
                        tokens,audio,features,current=proposal,wave[0],value[0],verified;accepted+=1
                item={'pass':pass_index+1,'stage':stage+1,'position':int(position),'old_code':old,
                      'new_code':int(tokens[position,stage]),'accepted':int(tokens[position,stage])!=old,
                      'previous_objective':previous,'objective':current[0],'distance':current[1],
                      'first_stage_nll':current[2],'evaluations':evaluations}
                trace.append(item)
                if on_step:on_step(item)
        if not accepted:break
    return {'tokens':tokens,'audio':audio,'features':features,'before_features':initial_features,
            'baseline_distance':baseline[1],'final_distance':current[1],
            'baseline_objective':baseline[0],'final_objective':current[0],
            'evaluations':evaluations,'trace':trace}


def delayed(initial,target,evaluator,probabilities,*,horizon_tokens=20,mode='feedback',weight=.03,
            on_step=None,plant=None):
    if mode not in ('unedited','no_feedback','feedback') or horizon_tokens<2:
        raise ValueError('Invalid delayed control settings')
    objective=evaluator.objective;observer=FeatureObserver(objective)
    tokens=np.asarray(initial,dtype=np.int64).copy()
    emitted=np.empty(0,dtype=np.float32);commanded=emitted.copy();continuation=emitted.copy()
    forecasts=np.empty((0,objective.dim),dtype=np.float32)
    trace=[];biases=[];evaluations=0
    for start in range(len(tokens)):
        cutoff=max(0,start*320-960);available=emitted[:cutoff].copy()
        observation={'observed_from_frame':len(observer.observed),'observed_until_frame':len(observer.observed),
                     'innovation_norm':0.,'feedback_update_rate':0.,'bias_norm':float(np.linalg.norm(observer.bias))}
        if len(available)>=objective.minimum_samples:
            actual=objective.extract(available[None])[0]
            observation=observer.update(actual,forecasts,mode=='feedback')
        observed_times=objective.times(len(observer.observed))
        if len(observed_times) and observed_times[-1]>=cutoff/16000:raise ValueError('Future observation leakage')
        context=commanded.copy();context[:cutoff]=available
        bias=observer.bias.copy();biases.append(bias)
        end=min(len(tokens),start+horizon_tokens);evaluate=evaluator.at(end,context,continuation)
        frozen=tokens[:start].copy()
        if mode=='unedited':
            wave,features=evaluate(tokens[None,:end]);wave,features=wave[0],features[0]
            result={'tokens':tokens[:end].copy(),'audio':wave,'features':features,'evaluations':1,'trace':[]}
        else:
            result=optimize(tokens[:end],target,evaluate,objective,evaluator.candidates,probabilities,weight,
                            start=start,passes=1,positions=2,include_current=True,bias=bias,
                            gain=.5 if mode=='feedback' else 0)
        tokens[:end]=result['tokens'];evaluations+=result['evaluations']
        if not np.array_equal(frozen,tokens[:start]):raise ValueError('Committed tokens changed')
        commit=(start+1)*320;chunk=result['audio'][start*320:commit].copy()
        commanded=np.concatenate((commanded,chunk))
        old=emitted.copy();output=chunk if plant is None else pcm16(plant(chunk.copy()))
        if output.shape!=chunk.shape:raise ValueError('Plant changed chunk length')
        emitted=np.concatenate((emitted,output));continuation=result['audio'][commit:].copy()
        if not np.array_equal(old,emitted[:len(old)]):raise ValueError('Emitted audio changed')
        features=result['features'];times=objective.times(len(features))
        committed_count=int(np.count_nonzero(times<commit/16000))
        if committed_count<len(forecasts):raise ValueError('Forecast time grid changed')
        forecasts=np.concatenate((forecasts,features[len(forecasts):committed_count]))
        item={'start_token':start,'planning_end_token':end,'decision_time_ms':start*20,
              'information_cutoff_samples':cutoff,'observer_input_samples':len(available) if len(available)>=objective.minimum_samples else 0,
              'observer_input_sha256':audio_hash(available) if len(available)>=objective.minimum_samples else None,
              'prefix_sha256_before':audio_hash(old),'retained_prefix_sha256_after':audio_hash(emitted[:len(old)]),
              'prediction_context_sha256':audio_hash(context),'blind_predicted_context_sha256':audio_hash(context[cutoff:]),
              'command_chunk_sha256':audio_hash(chunk),'chunk_sha256':audio_hash(output),
              'evaluations':evaluations,'bias_before_norm':float(np.linalg.norm(bias)),
              'edits':result['trace'],**observation}
        trace.append(item)
        if on_step:on_step(item)
    control_observed=observer.observed.copy()
    terminal=observer.update(objective.extract(emitted[None])[0],forecasts,False)
    terminal.update({'used_for_control':False,'observation_time_ms':len(tokens)*20+60})
    return {'tokens':tokens,'audio':emitted,'commanded_audio':commanded,'prediction_features':forecasts,
            'observed_features':observer.observed,'control_observed_features':control_observed,
            'biases':np.asarray(biases),'trace':trace,'terminal_observation':terminal,'evaluations':evaluations}
