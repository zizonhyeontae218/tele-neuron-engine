"""Transition-aware paired-bank CEM; production Session is the physics simulator."""
from copy import deepcopy
import numpy as np
from tele_neuron.endophalon.readout import RidgeReadout
from tele_neuron.endophalon3.isa import Instruction, OPS, LEVELS, random_stream, evaluate
from .model import Model, Bank
from .session import Session

TRAIN=range(30000,30024)
VALIDATION=range(30024,30032)
TEST=range(30032,30064)  # Previously observed; descriptive comparison only.
CONFIRMATION=range(40000,40064)  # Frozen before this training run.


def observe(model,streams,pulses=None,reset=False):
    streams=tuple(streams);pulses=pulses or model.config.stream.pulses
    session=Session(model,streams);initial_x=session.positions.copy();initial_v=session.velocities.copy()
    inputs=[random_stream('AB',model.config.physics.seed,s,pulses) for s in streams]
    x=[];energy=[];jumps=[];ages=[];age=np.full(len(streams),1000)
    for k in range(pulses):
        changed=np.array([s.levels[k]!=s.levels[k-1] for s in inputs]) if k else np.zeros(len(streams),bool)
        if reset:
            session.positions[changed]=initial_x[changed];session.velocities[changed]=initial_v[changed]
        age=np.where(changed,0,age+1);ages.append(age.copy())
        result=session.step([Instruction('AB'[s.levels[k]],OPS[s.ops[k]],int(s.a[k]),int(s.b[k])) for s in inputs])
        x.append([r.features for r in result]);energy.append([r.kinetic_before for r in result]);jumps.append([r.switching_energy for r in result])
    return {'features':np.array(x).transpose(1,0,2),'energy':np.array(energy).T,'switching_energy':np.array(jumps).T,
            'age':np.array(ages).T,'levels':np.stack([s.levels for s in inputs]),'ops':np.stack([s.ops for s in inputs]),
            'targets':np.stack([evaluate(s.ops,s.a,s.b)[0] for s in inputs]),'streams':np.array(streams)}


def subset(data,streams):
    mask=np.isin(data['streams'],list(streams));return {key:value[mask] for key,value in data.items()}


def flatten(data,washout):
    return {key:(value[:,washout:].reshape(-1,value.shape[-1]) if key=='features' else value[:,washout:].ravel()) for key,value in data.items() if key!='streams'}


def fit_heads(model,data,streams):
    train=flatten(subset(data,streams),model.config.washout);banks={}
    for bit,level in enumerate('AB'):
        mask=train['levels']==bit
        head=RidgeReadout(model.configs[level].ridge_alpha).fit(train['features'][mask],train['targets'][mask,None])
        banks[level]=Bank(model.banks[level].raw_config,head)
    return Model(banks,model.provenance)


def metrics(model,data,streams):
    d=flatten(subset(data,streams),model.config.washout);scores=np.empty(len(d['targets']));by_level={}
    for bit,level in enumerate('AB'):
        mask=d['levels']==bit;raw=model.banks[level].head.scores(d['features'][mask])[:,0];target=d['targets'][mask];ops=d['ops'][mask];age=d['age'][mask]
        scores[mask]=raw;hit=(raw>=.5)==target
        row={'accuracy':float(hit.mean()),'mse':float(np.mean((raw-target)**2)),'n':int(mask.sum()),
             'per_op':{op:{'accuracy':float(hit[ops==OPS.index(op)].mean()),'n':int((ops==OPS.index(op)).sum())} for op in LEVELS[level]}}
        for k,name in [(0,'first_after_switch'),(1,'second_after_switch')]:
            chosen=age==k;row[name]={'accuracy':float(hit[chosen].mean()) if chosen.any() else None,'n':int(chosen.sum())}
        by_level[level]=row
    return {'accuracy':float(((scores>=.5)==d['targets']).mean()),'mse':float(np.mean((scores-d['targets'])**2)),'levels':by_level}


def objective(model,data,train=TRAIN,validation=VALIDATION):
    fitted=fit_heads(model,data,train);d=flatten(subset(data,validation),model.config.washout)
    losses=[];noises=[];complexities=[]
    for bit,level in enumerate('AB'):
        mask=d['levels']==bit;raw=fitted.banks[level].head.scores(d['features'][mask])[:,0];target=d['targets'][mask]
        losses.append(float(np.mean((raw-target)**2)))
        one,zero=raw[target==1],raw[target==0]
        noises.append(float(.5*(one.std()+zero.std())/(abs(one.mean()-zero.mean())+1e-6)))
        complexities.append(float(np.mean(fitted.banks[level].head.weights[:-1]**2)))
    energy=subset(data,range(train.start,validation.stop))['energy'][:,model.config.washout:]
    k=np.arange(energy.shape[1],dtype=float);k-=k.mean();logs=np.log(energy+1e-9)
    drift=float(np.median(np.abs((logs-logs.mean(axis=1,keepdims=True))@k/(k@k)))*len(k))
    mean_loss=float(np.mean(losses));noise=float(np.mean(noises));complexity=float(np.mean(complexities))
    j=mean_loss+.05*noise+.1*drift+.01*complexity
    if not np.isfinite([j,mean_loss,noise,drift,complexity]).all():raise FloatingPointError('Non-finite objective')
    return fitted,{'j':j,'l_val':mean_loss,'r_noise':noise,'r_drift':drift,'r_complexity':complexity,'finite':True,'validation':metrics(fitted,data,validation)}


def encode(model):
    return np.concatenate([model.configs[level].space.encode(model.configs[level].theta_init) for level in 'AB'])


def decode(base,vector):
    vector=np.asarray(vector);dimensions=[base.configs[l].space.dim for l in 'AB']
    if vector.shape!=(sum(dimensions),):raise ValueError('Paired theta vector dimension mismatch')
    payload=deepcopy(base.to_payload());start=0
    for level,dim in zip('AB',dimensions):
        payload['banks'][level]['config']['theta_init']=base.configs[level].space.decode(vector[start:start+dim]).to_payload()
        payload['banks'][level]['head']=None;start+=dim
    return Model.from_payload(payload)


def candidate(task):
    base_payload,vector=task
    try:
        model=decode(Model.from_payload(base_payload),vector)
        data=observe(model,range(TRAIN.start,VALIDATION.stop));_,result=objective(model,data)
        result['max_switching_energy']=float(np.abs(data['switching_energy']).max())
        return result
    except (FloatingPointError,np.linalg.LinAlgError) as error:
        return {'j':10.,'finite':False,'failure':type(error).__name__+': '+str(error)}


def new_state(base,seed=1):
    u=encode(base)
    return {'generation':-1,'mean':u.tolist(),'sigma':np.full(len(u),.5).tolist(),'best_u':u.tolist(),
            'best':None,'rng':np.random.default_rng(seed).bit_generator.state,'history':[],'evaluations':0}


def generation(state,evaluate_candidates,population=24,elite=6):
    rng=np.random.default_rng();rng.bit_generator.state=state['rng'];mean=np.array(state['mean']);sigma=np.array(state['sigma']);gen=state['generation']+1
    candidates=mean[None] if gen==0 else np.vstack([mean,mean+rng.standard_normal((population-1,len(mean)))*sigma])
    results=evaluate_candidates(candidates);js=np.array([r['j'] for r in results])
    if len(results)!=len(candidates) or not np.isfinite(js).all():raise ValueError('Invalid candidate results')
    top=int(js.argmin());best=state['best'];best_u=state['best_u']
    if best is None or results[top]['j']<best['j']:best=results[top];best_u=candidates[top].tolist()
    if gen:
        weights=np.log(elite+.5)-np.log(np.arange(1,elite+1));weights/=weights.sum()
        chosen=candidates[np.argsort(js,kind='stable')[:elite]];spread=np.sqrt(weights@(chosen-mean)**2)
        mean=weights@chosen;sigma=np.maximum(.7*spread+.3*sigma,.05)
    row={'generation':gen,'best':best,'failed':sum(not r['finite'] for r in results),'sigma_mean':float(sigma.mean()),'candidates':results}
    return {'generation':gen,'mean':mean.tolist(),'sigma':sigma.tolist(),'best_u':best_u,'best':best,
            'rng':rng.bit_generator.state,'history':[*state['history'],row],'evaluations':state['evaluations']+len(candidates)}


def passed(metric):
    return set(metric['levels'])=={'A','B'} and all(set(metric['levels'][level]['per_op'])==set(LEVELS[level]) and metric['levels'][level]['accuracy']>=.9 and all(v['accuracy']>=.85 and v['n']>0 for v in metric['levels'][level]['per_op'].values()) for level in 'AB')
