"""Independent fixed-level CEM with exactly one physics/readout per model."""
from copy import deepcopy
import numpy as np
from tele_neuron.endophalon.readout import RidgeReadout
from .model import Module
from .session import Session
from .isa import Instruction,OPS,random_stream,evaluate


def observe(module,ids,pulses=None):
    ids=tuple(ids);n=pulses or module.config.stream.pulses;s=Session(module,ids)
    programs=[random_stream(module.level,module.config.physics.seed,k,n) for k in ids];x=[];energy=[]
    for k in range(n):
        rows=s.step([Instruction(module.level,OPS[p.ops[k]],int(p.a[k]),int(p.b[k])) for p in programs]);x.append([r.features for r in rows]);energy.append([r.kinetic_before for r in rows])
    return {'streams':np.array(ids),'features':np.array(x).transpose(1,0,2),'energy':np.array(energy).T,'ops':np.stack([p.ops for p in programs]),'targets':np.stack([evaluate(p.ops,p.a,p.b)[0] for p in programs])}


def subset(data,ids):
    chosen=np.isin(data['streams'],list(ids));return {k:v[chosen] for k,v in data.items()}


def flatten(module,data):
    w=module.config.washout
    return data['features'][:,w:].reshape(-1,16),data['targets'][:,w:].reshape(-1,1),data['ops'][:,w:].ravel()


def metrics(module,data):
    x,y,ops=flatten(module,data);raw=module.head.scores(x)[:,0];hit=(raw>=.5)==y[:,0]
    return {'accuracy':float(hit.mean()),'mse':float(np.mean((raw-y[:,0])**2)),'per_op':{op:{'accuracy':float(hit[ops==OPS.index(op)].mean()),'n':int((ops==OPS.index(op)).sum())} for op in module.config.ops}}


def fit(module,data,ids):
    x,y,_=flatten(module,subset(data,ids));head=RidgeReadout(module.config.ridge_alpha).fit(x,y);return Module(module.raw,head,module.provenance)


def fitness(module,data,train,val):
    fitted=fit(module,data,train);x,y,_=flatten(module,subset(data,val));raw=fitted.head.scores(x)[:,0];target=y[:,0]
    one,zero=raw[target==1],raw[target==0];noise=float(.5*(one.std()+zero.std())/(abs(one.mean()-zero.mean())+1e-6));complexity=float(np.mean(fitted.head.weights[:-1]**2))
    e=data['energy'][:,module.config.washout:];k=np.arange(e.shape[1],dtype=float);k-=k.mean();logs=np.log(e+1e-9);drift=float(np.median(np.abs((logs-logs.mean(axis=1,keepdims=True))@k/(k@k)))*len(k))
    loss=float(np.mean((raw-target)**2));j=loss+.05*noise+.1*drift+.01*complexity
    if not np.isfinite([j,loss,noise,complexity,drift]).all():raise FloatingPointError('Non-finite fitness')
    return fitted,{'j':j,'l_val':loss,'r_noise':noise,'r_drift':drift,'r_complexity':complexity,'finite':True,'validation':metrics(fitted,subset(data,val))}


def decode(raw,u):
    seed=Module(raw);out=deepcopy(raw);out['theta_init']=seed.config.space.decode(np.asarray(u)).to_payload();return Module(out)


def candidate(task):
    raw,u=task;ids=raw['training_stream_ids'];train=ids['train'];val=ids['validation']
    try:
        m=decode(raw,u);data=observe(m,[*train,*val]);_,result=fitness(m,data,train,val);return result
    except (FloatingPointError,np.linalg.LinAlgError) as e:return {'j':10.,'finite':False,'failure':type(e).__name__+': '+str(e)}


def new_state(raw):
    m=Module(raw);u=m.config.space.encode(m.config.theta_init)
    return {'generation':-1,'mean':u.tolist(),'sigma':np.full(len(u),.5).tolist(),'best_u':u.tolist(),'best':None,'rng':np.random.default_rng(int(raw['outer']['seed'])).bit_generator.state,'history':[],'evaluations':0}


def advance(state,score,population=24,elite=6):
    rng=np.random.default_rng();rng.bit_generator.state=state['rng'];mean=np.array(state['mean']);sigma=np.array(state['sigma']);g=state['generation']+1
    u=mean[None] if g==0 else np.vstack([mean,mean+rng.standard_normal((population-1,len(mean)))*sigma]);results=score(u);js=np.array([r['j'] for r in results]);i=int(js.argmin())
    if len(results)!=len(u) or not np.isfinite(js).all():raise ValueError('Invalid evaluations')
    best=state['best'];best_u=state['best_u']
    if best is None or results[i]['j']<best['j']:best=results[i];best_u=u[i].tolist()
    if g:
        w=np.log(elite+.5)-np.log(np.arange(1,elite+1));w/=w.sum();top=u[np.argsort(js,kind='stable')[:elite]];spread=np.sqrt(w@(top-mean)**2);mean=w@top;sigma=np.maximum(.7*spread+.3*sigma,.05)
    return {'generation':g,'mean':mean.tolist(),'sigma':sigma.tolist(),'best_u':best_u,'best':best,'rng':rng.bit_generator.state,'history':[*state['history'],{'generation':g,'best':best,'failed':sum(not r['finite'] for r in results),'candidates':results}],'evaluations':state['evaluations']+len(u)}
