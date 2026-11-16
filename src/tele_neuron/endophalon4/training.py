"""Atomic checksummed generation checkpoints and independent resumable CEM."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
import hashlib,json,os,platform,sys,time
import numpy as np
from tele_neuron.endophalon3_software.training import advance
from .experiment import ROOT,config,provenance,candidate,fit,sha

OUT=ROOT/'models/endophalon4'


def encoded(value):return json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()


def atomic(path,value,envelope=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if envelope:value={'sha256':hashlib.sha256(encoded(value)).hexdigest(),'payload':value}
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def load(path):
    obj=json.loads(Path(path).read_text());p=obj['payload']
    if hashlib.sha256(encoded(p)).hexdigest()!=obj['sha256']:raise ValueError('Checkpoint checksum mismatch')
    return p


def environment():
    return {'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
            'cpu_count':os.cpu_count(),'affinity':len(os.sched_getaffinity(0)),
            'cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
            'memory_max':Path('/sys/fs/cgroup/memory.max').read_text().strip(),
            'workers':config()['workers'],'blas_threads':os.environ.get('OPENBLAS_NUM_THREADS')}


def initial(direction):
    return {'schema':'endophalon4-checkpoint/v1','direction':direction,'generation':-1,
            'evaluations':0,'mean':[0.]*21,'sigma':[.5]*21,'variance':[.25]*21,
            'best_u':[0.]*21,'best':None,'rng':np.random.default_rng(config()['directions'][direction]['search_seed']).bit_generator.state,
            'history':[],'segments':[],'provenance':provenance(),'environment':environment(),
            'elapsed_seconds':0.}


def verify(state,direction,strict=True):
    if state['direction']!=direction:raise ValueError('Direction mismatch')
    g=state['generation']
    if not -1<=g<=40 or state['evaluations']!=(0 if g<0 else 1+24*g):raise ValueError('Wrong generation/evaluation count')
    if [h['generation'] for h in state['history']]!=list(range(g+1)):raise ValueError('Generation gap/duplicate')
    for h in state['history']:
        if len(h['candidates'])!=(1 if h['generation']==0 else 24):raise ValueError('Candidate gap')
    for key in ['mean','sigma','best_u']:
        v=np.asarray(state[key]);
        if v.shape!=(21,) or not np.isfinite(v).all():raise ValueError('Invalid optimizer vector')
    if not np.array_equal(np.square(state['sigma']),state['variance']):raise ValueError('Invalid variance')
    r=np.random.default_rng();r.bit_generator.state=state['rng']
    if state['provenance']!=provenance():raise ValueError('Frozen sources/models/config changed')
    if strict and state['environment']!=environment():raise ValueError('Execution environment changed')
    return True


def latest(direction):
    paths=sorted((OUT/direction/'checkpoints').glob('generation_*.json'))
    if not paths:return initial(direction)
    # Never silently repeat a completed generation after corruption.
    for i,p in enumerate(paths):
        if p.name!=f'generation_{i:03}.json':raise ValueError('Checkpoint generation gap')
        s=load(p)
        if s['generation']!=i:raise ValueError('Checkpoint filename mismatch')
    state=load(paths[-1]);verify(state,direction)
    return state


def advance_record(state,score):
    start=time.monotonic();out=advance(deepcopy(state),score,24,6)
    for k in ['schema','direction','provenance','environment','segments']:out[k]=deepcopy(state[k])
    out['elapsed_seconds']=state['elapsed_seconds']+time.monotonic()-start
    out['variance']=np.square(out['sigma']).tolist()
    return out


def segment(direction,count=5):
    cfg=config();state=latest(direction)
    if state['generation']==40:
        finalize(direction,state)
        print(direction,'already complete; no evaluations repeated',flush=True);return
    before=state['generation'];target=min(40,max(0,before)+count)
    # Cache exact sender trajectories before forking; immutable pages are shared.
    from .experiment import data
    for split in ['train','validation']:data(direction,split)
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        def score(values):return list(pool.map(candidate,[(direction,v.tolist()) for v in values]))
        while state['generation']<target:
            state=advance_record(state,score)
            if not state['best']['finite']:raise FloatingPointError('No finite best candidate')
            g=state['generation']
            if g==target:
                state['segments'].append({'start_generation':max(1,before+1),'end_generation':g,
                    'before_generation':before,'before_evaluations':0 if before<0 else 1+24*before,
                    'after_evaluations':state['evaluations'],'seconds':time.monotonic()-start})
            verify(state,direction)
            path=OUT/direction/'checkpoints'/f'generation_{g:03}.json'
            if path.exists():raise FileExistsError('Refusing completed generation overwrite')
            atomic(path,state,True);verify(load(path),direction)
            print(direction,'generation',g,'evaluations',state['evaluations'],
                  'validation',round(state['best']['validation']['accuracy'],6),
                  'MSE',round(state['best']['j'],6),'failed',state['history'][-1]['failed'],flush=True)
    if state['generation']==40:finalize(direction,state)


def finalize(direction,state):
    path=OUT/direction/'model.json'
    if path.exists():
        obj=json.loads(path.read_text())
        if obj['parameters']!=state['best_u'] or obj['provenance']!=state['provenance']:raise ValueError('Final bundle mismatch')
        return
    head=fit(direction,state['best_u'])
    atomic(path,{'schema':'endophalon4-coupling/v1','direction':direction,
                'parameters':state['best_u'],'readout':head.to_payload(),
                'provenance':state['provenance'],'generations':40,'evaluations':961,
                'selection':'mean four-lag validation MSE only; train-only Ridge'})
    atomic(ROOT/f'docs/endophalon4/{direction}_history.json',
           {'history':state['history'],'segments':state['segments'],
            'environment':state['environment'],'elapsed_seconds':state['elapsed_seconds']})
