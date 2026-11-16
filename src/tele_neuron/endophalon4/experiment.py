"""Deterministic disjoint streams, exact sender trajectory cache, local readouts."""
from functools import lru_cache
from pathlib import Path
import hashlib,json,time
import numpy as np
from tele_neuron.endophalon3_software.model import Module
from tele_neuron.endophalon3_software.session import Session
from tele_neuron.endophalon3_software.isa import random_stream
from tele_neuron.endophalon.readout import RidgeReadout
from .runtime import zones,reference,coupling_force,tick

ROOT=Path(__file__).resolve().parents[3]
CONFIG=ROOT/'configs/endophalon4/experiment.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config():return json.loads(CONFIG.read_text())


@lru_cache(None)
def models(direction):
    a,b=direction.split('_to_')
    return tuple(Module.load(ROOT/f'models/endophalon3_software/{k}.json') for k in [a,b])


def provenance():
    paths=[CONFIG,*sorted((ROOT/'src/tele_neuron/endophalon4').glob('*.py')),
           *sorted((ROOT/'src/tele_neuron/endophalon3_software').glob('*.py')),
           *sorted((ROOT/'src/tele_neuron/endophalon2_pro').glob('*.py')),
           ROOT/'src/tele_neuron/endophalon2/engine.py',
           ROOT/'src/tele_neuron/endophalon/readout.py',
           ROOT/'scripts/train_endophalon4.py']
    return {'models':{k:sha(ROOT/f'models/endophalon3_software/{k}.json') for k in ['A','B','current']},
            'sources':{str(p.relative_to(ROOT)):sha(p) for p in paths},'dataset_and_config':sha(CONFIG)}


@lru_cache(None)
def data(direction, split):
    cfg=config();sp=cfg['directions'][direction]['splits'][split]
    sender,receiver=models(direction);ids=sp['ids'];n=cfg['events']
    source=[random_stream(sender.level,sp['seed'],k,n) for k in ids]
    dest=[random_stream(receiver.level,sp['seed']+10000,k+1000000,n) for k in ids]
    sch=np.stack([p.channels(sender.config.encoding) for p in source])
    rch=np.stack([p.channels(receiver.config.encoding) for p in dest])
    s=Session(sender,tuple(ids));warm=s.spec.warmup;steps=warm+n*16
    obs=np.empty((steps,len(ids),16));sf=np.empty((len(ids),n,16))
    for t in range(steps):
        obs[t]=zones(s)
        e,d=divmod(t-warm,16)
        s._tick(sch[:,e] if e>=0 and d<s.spec.pulse_steps else None)
        if e>=0 and d==11:sf[:,e]=zones(s)
    a=np.stack([p.a for p in source]);w=cfg['washout']
    targets=np.stack([a[:,w-l:n-l] for l in cfg['lags']],axis=-1)
    return {'ids':tuple(k+1000000 for k in ids),'obs':obs,'home':reference(s),
            'receiver_channels':rch,'targets':targets,'sender_features':sf,
            'sender_ops':np.stack([p.ops for p in source]),
            'receiver_ops':np.stack([p.ops for p in dest]),
            'sender_targets':np.stack([p.targets()[0] for p in source]),
            'receiver_targets':np.stack([p.targets()[0] for p in dest])}


def observe(direction,split,u,mode='coupled',events=None):
    d=data(direction,split);_,receiver=models(direction)
    s=Session(receiver,d['ids']);n=events or config()['events'];warm=s.spec.warmup
    obs=d['obs']
    if mode=='shuffled':
        obs=obs.copy();seed=config()['directions'][direction]['splits'][split]['seed']
        rng=np.random.default_rng([seed,0x53485546])
        for b in range(s.count):obs[:,b]=obs[rng.permutation(len(obs)),b]
    f=np.empty((s.count,n,16));max_pos=0.;max_vel=0.
    for t in range(warm+n*16):
        e,k=divmod(t-warm,16)
        force=coupling_force(s,obs[t],d['home'],u,mode not in ['uncoupled','zero'])
        channels=d['receiver_channels'][:,e] if e>=0 and k<s.spec.pulse_steps else None
        tick(s,channels,force)
        if e>=0 and k==11:f[:,e]=zones(s)
        max_pos=max(max_pos,float(np.max(np.abs(s.positions))))
        max_vel=max(max_vel,float(np.max(np.abs(s.velocities))))
    return f,{'finite':True,'max_abs_position':max_pos,'max_abs_velocity':max_vel,'ticks':s.clock}


def fit(direction,u,mode='coupled'):
    cfg=config();f,_=observe(direction,'train',u,mode);w=cfg['washout']
    y=data(direction,'train')['targets']
    return RidgeReadout(cfg['ridge']).fit(f[:,w:].reshape(-1,16),y.reshape(-1,4))


def memory(head,features,targets):
    w=config()['washout'];raw=head.scores(features[:,w:].reshape(-1,16)).reshape(targets.shape)
    if not np.isfinite(raw).all():raise FloatingPointError('Non-finite readout')
    hit=(raw>=.5)==targets
    return {'accuracy':float(hit.mean()),'mse':float(np.mean((raw-targets)**2)),
            'per_lag':{str(l):float(hit[...,i].mean()) for i,l in enumerate(config()['lags'])},
            'stream_accuracy':hit.mean(axis=(1,2)).tolist(),'n_bits':hit.size}


def candidate(task):
    direction,u=task;start=time.monotonic()
    try:
        head=fit(direction,u);f,stability=observe(direction,'validation',u)
        m=memory(head,f,data(direction,'validation')['targets'])
        return {'j':m['mse'],'finite':True,'validation':m,'stability':stability,
                'seconds':time.monotonic()-start}
    except (FloatingPointError,np.linalg.LinAlgError) as e:
        return {'j':1e6,'finite':False,'failure':str(e),'seconds':time.monotonic()-start}
