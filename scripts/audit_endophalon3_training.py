"""Audit saved Endophalon-3 CLI cards; never train or modify model/source files.

Replay the serialized readout, check disjoint whole-stream splits and fitness,
retain event-level predictions, and measure all regularizers on each split.
"""
from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from tele_neuron.endophalon.readout import RidgeReadout
from tele_neuron.endophalon2_pro.core import drift
from tele_neuron.endophalon3.config import load_config
from tele_neuron.endophalon3.core import collect, dataset, fitness
from tele_neuron.endophalon3.isa import OPS, evaluate, random_stream
from tele_neuron.endophalon3.theta import Theta3

RUN=ROOT/'docs/endophalon3_runs/20261031-training'


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def metrics(config,data,streams,readout):
    x,y,ops=dataset(config,data,streams)
    scores=readout.scores(x)[:,0];target=y[:,0].astype(int);pred=(scores>=0.5).astype(int)
    one,zero=scores[target==1],scores[target==0]
    noise=float(.5*(one.std()+zero.std())/(abs(one.mean()-zero.mean())+1e-6))
    complexity=float(np.mean(readout.weights[:-1]**2))
    d=drift(config,data,streams)
    mse=float(np.mean((scores-target)**2))
    confusion={}
    for i,name in enumerate(OPS):
        mask=ops==i
        if not mask.any():continue
        t,p=target[mask],pred[mask]
        confusion[name]={'n':int(mask.sum()),'accuracy':float((p==t).mean()),'tn':int(((t==0)&(p==0)).sum()),'fp':int(((t==0)&(p==1)).sum()),'fn':int(((t==1)&(p==0)).sum()),'tp':int(((t==1)&(p==1)).sum()),'majority_accuracy':float(max(t.mean(),1-t.mean()))}
    return {'accuracy':float((pred==target).mean()),'mse':mse,'r_noise':noise,'r_drift':d,'r_complexity':complexity,'j_split':mse+config.objective['lambda_noise']*noise+config.objective['lambda_drift']*d+config.objective['lambda_complexity']*complexity,'per_op':confusion}


def audit(path):
    card=json.loads(path.read_text());name=path.stem.removeprefix('20261031_')
    output=RUN/(name+'.audit.json')
    if output.exists():raise FileExistsError(output)
    meta=json.loads((RUN/(name+'.run.json')).read_text())
    cfg=replace(load_config(ROOT/card['config']),workers=4,regions=card['metrics']['regions'])
    assert (cfg.network.nodes,cfg.physics.substeps,cfg.zones)==(144,2,16)
    assert cfg.outer['generations']==40 and cfg.outer['population']==24
    assert card['metrics']['evaluations']==961 and len(card['metrics']['history'])==41
    train,val,test=cfg.stream_ranges();splits={'train':train,'validation':val,'test':test}
    assert not (set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
    assert cfg.washout>=2
    assert card['operations']==list(cfg.ops) and card['input_channels']==list(cfg.channels)
    history=card['metrics']['history']
    assert [h['generation'] for h in history]==list(range(41))
    assert all(a['best_j']>=b['best_j'] for a,b in zip(history,history[1:]))
    theta=Theta3.from_payload(card['theta']);[data]=collect(cfg,[theta],range(test.stop),nodes=True)
    assert data.finite, 'Selected theta has a non-finite integration flag'
    readout=RidgeReadout.from_payload(card['readout'])
    x,y,_=dataset(cfg,data,train);refit=RidgeReadout(cfg.ridge_alpha).fit(x,y)
    refit_errors={k:float(np.max(np.abs(getattr(readout,k)-getattr(refit,k)))) for k in ['mean','scale','weights']}
    bitwise_statistics_equal=bool(np.array_equal(readout.mean,refit.mean) and np.array_equal(readout.scale,refit.scale))
    print(name, 'readout refit max absolute errors', refit_errors, 'bitwise statistics equal', bitwise_statistics_equal, flush=True)
    assert np.allclose(readout.mean,refit.mean,rtol=1e-12,atol=1e-12) and np.allclose(readout.scale,refit.scale,rtol=1e-12,atol=1e-12), refit_errors
    assert np.allclose(readout.weights,refit.weights,rtol=1e-10,atol=1e-10)
    actual_fitness=fitness(cfg,data,train,val)
    for key in ['j','l_val','r_noise','r_drift','r_complexity','val_accuracy']:
        assert np.isclose(getattr(actual_fitness,key),card['metrics']['fitness_trained'][key],rtol=1e-10,atol=1e-10),key
    result={k:metrics(cfg,data,v,readout) for k,v in splits.items()}
    expected=card['metrics']['test']['trained']['single_readout']
    assert result['test']['accuracy']==expected['accuracy']
    assert {k:v['accuracy'] for k,v in result['test']['per_op'].items()}==expected['per_op']
    fingerprints={}
    for label,streams in splits.items():
        fingerprints[label]={str(s):sha(random_stream(cfg.level,cfg.physics.seed,s,cfg.stream.pulses).channels(cfg.encoding)) for s in streams}
    hash_sets=[set(v.values()) for v in fingerprints.values()]
    assert not(hash_sets[0]&hash_sets[1] or hash_sets[0]&hash_sets[2] or hash_sets[1]&hash_sets[2])
    numerical={}
    for label,streams in splits.items():
        indices=data.index(streams)
        numerical[label]={}
        for field in ['features','nodes','energy','wall']:
            a=getattr(data,field)[indices]
            numerical[label][field]={'all_finite':bool(np.isfinite(a).all()),'min':float(a.min()),'max':float(a.max()),'max_abs':float(np.abs(a).max())}
        assert all(v['all_finite'] for v in numerical[label].values())
    # Preserve scored test events, including histories for failure-pattern analysis.
    sx,sy,sops=dataset(cfg,data,test);raw=readout.scores(sx)[:,0]
    events=[];cursor=0
    for row in data.index(test):
        target,valid=evaluate(data.ops[row],data.a[row],data.b[row])
        for k in range(cfg.washout,cfg.stream.pulses):
            if not valid[k]:continue
            a,b,a1,a2=(int(data.a[row,k]),int(data.b[row,k]),int(data.a[row,k-1]),int(data.a[row,k-2]))
            events.append({'stream':data.streams[row],'event':k,'op':OPS[data.ops[row,k]],'a':a,'b':b,'a1':a1,'a2':a2,'target':int(target[k]),'score':float(raw[cursor]),'prediction':int(raw[cursor]>=.5),'correct':int((raw[cursor]>=.5)==target[k])});cursor+=1
    csvpath=RUN/(name+'.predictions.csv')
    with csvpath.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(events[0]));writer.writeheader();writer.writerows(events)
    patterns={}
    for op in cfg.ops:
        groups={}
        for e in events:
            if e['op']!=op:continue
            key=''.join(str(e[k]) for k in ['a','b','a1','a2'])
            group=groups.setdefault(key,{'n':0,'errors':0});group['n']+=1;group['errors']+=1-e['correct']
        patterns[op]=groups
    failed=sum(h['failed'] for h in card['metrics']['history'])
    controls=card['metrics']['test']['trained']
    payload={'card':str(path.relative_to(ROOT)),'replay':'PASS','bitwise_statistics_equal':bitwise_statistics_equal,'readout_refit_max_abs_error':refit_errors,'mean_scale_tolerance':{'rtol':1e-12,'atol':1e-12},'saved_readout_matches_train_refit':True,'fitness_matches_card':True,'invariants':{'nodes':144,'substeps':2,'zones':16,'generations':40,'population':24,'theta_evaluations':961,'physics_seed':cfg.physics.seed,'search_seed':meta['search_seed'],'theta_dimensions':cfg.space.dim,'slow_nodes':int(cfg.slow.sum())},'splits':{k:list(v) for k,v in splits.items()},'events_per_stream':cfg.stream.pulses,'washout':cfg.washout,'split_fingerprints':fingerprints,'split_metrics':result,'optimization_drift_scope':'combined train+validation, not validation alone','fitness':card['metrics']['fitness_trained'],'numerical':numerical,'failed_training_candidates':failed,'finite_flag':data.finite,'controls':{k:v for k,v in controls.items() if k not in ['drift','finite']},'history_baseline':card['metrics']['test']['history_baseline'],'pattern_errors_abaa1a2':patterns,'timing':meta,'limitations':['Search seed changes the outer optimizer only; physics and streams retain seed 1.','Numerical finite/drift measurements cover 64-event streams; no long-horizon guarantee.','Failed flags count feature/end-position divergence; no per-substep telemetry is available.','No test selection: theta is selected by train+validation fitness, before test replay.']}
    if card['metrics']['random_search'] is not None:
        r=card['metrics']['random_search'];[rd]=collect(cfg,[Theta3.from_payload(r['theta'])],range(test.stop))
        rx,ry,_=dataset(cfg,rd,train);rr=RidgeReadout(cfg.ridge_alpha).fit(rx,ry)
        rm={k:metrics(cfg,rd,v,rr) for k,v in splits.items()}
        assert rm['test']['accuracy']==r['test']['accuracy']
        assert np.isclose(fitness(cfg,rd,train,val).j,r['j'])
        payload['random_search']={'budget':961,'search_seed':meta['search_seed']+1000,'split_metrics':rm,'selected_j':r['j'],'candidate_failure_count':'not retained by the existing CLI'}
    output.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(name,'PASS',result['test']['accuracy'],flush=True)


def main():
    for path in sorted((ROOT/'models/endophalon3').glob('20261031_*.json')):
        name=path.stem.removeprefix('20261031_')
        meta=RUN/(name+'.run.json')
        if not meta.exists() or json.loads(meta.read_text()).get('status')!='completed':continue
        if not (RUN/(name+'.audit.json')).exists():audit(path)

if __name__=='__main__':main()
