"""Bounded, checkpointed AB training; no held-out data enters model selection."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from tele_neuron.endophalon3.config import load_config
from tele_neuron.endophalon3.core import collect, dataset, fitness, single_readout
from tele_neuron.endophalon3.isa import OPS, LEVELS
from tele_neuron.endophalon3.theta import Theta3
from tele_neuron.endophalon.readout import RidgeReadout


def atomic(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def grouped(cfg,data,streams,head):
    x,y,ops=dataset(cfg,data,streams); hit=(head.scores(x)[:,0]>=.5)==y[:,0]
    levels=data.levels[data.index(streams),cfg.washout:].reshape(-1)
    out={'accuracy':float(hit.mean()),'levels':{}}
    for level,bit in [('A',0),('B',1)]:
        mask=levels==bit
        out['levels'][level]={'accuracy':float(hit[mask].mean()) if mask.any() else 0.,'per_op':{op:float(hit[mask&(ops==OPS.index(op))].mean()) if (mask&(ops==OPS.index(op))).any() else 0. for op in LEVELS[level]}}
    return out


def passed(result):
    return all(v['accuracy']>=.9 and all(a>=.85 for a in v['per_op'].values()) for v in result['levels'].values())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True,choices=['ab_binary_s1','ab_binary_s2','ab_onehot_s1']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    root=Path.cwd();directory=root/'docs/endophalon3_joint_runs/20261101'/args.run;directory.mkdir(parents=True,exist_ok=True)
    cfg=replace(load_config(root/f'configs/endophalon3/{args.run}.json'),workers=args.workers)
    assert (cfg.level,cfg.network.nodes,cfg.physics.substeps,cfg.zones)==('AB',144,2,16)
    checkpoint=directory/'checkpoint.json';final=directory/'model.json'
    if final.exists():raise FileExistsError('Completed run must not be overwritten')
    fingerprint=hashlib.sha256((root/f'configs/endophalon3/{args.run}.json').read_bytes()).hexdigest()
    train,val,_=cfg.stream_ranges();space=cfg.space;pop=24;elite=6;budget=120;patience=40
    rng=np.random.default_rng(int(cfg.outer['seed']));weights=np.log(elite+.5)-np.log(np.arange(1,elite+1));weights/=weights.sum()
    if checkpoint.exists():
        state=json.loads(checkpoint.read_text());assert state['config_sha256']==fingerprint
        mean=np.array(state['mean']);sigma=np.array(state['sigma']);best_u=np.array(state['best_u']);best=state['best'];history=state['history'];last=state['last_improved'];rng.bit_generator.state=state['rng'];start=state['generation']+1
    else:
        mean=space.encode(cfg.theta_init);sigma=np.full(space.dim,.5);best_u=mean.copy();best=None;history=[];last=0;start=0
    reason='budget';started=time.monotonic()
    for gen in range(start,budget+1):
        candidates=mean[None] if gen==0 else np.vstack([mean,mean+rng.standard_normal((pop-1,space.dim))*sigma])
        data=collect(cfg,[space.decode(u) for u in candidates],range(val.stop));scores=[fitness(cfg,d,train,val) for d in data];js=np.array([f.j for f in scores]);top=int(js.argmin())
        if best is None or scores[top].j<best['j']-1e-12:
            best_u=candidates[top].copy();best=asdict(scores[top]);last=gen
        goal_candidate=None
        # Validation-only gates: test/confirmation never checked in this loop.
        for i,f in enumerate(scores):
            if f.val_accuracy>=.9 and f.finite:
                head=single_readout(cfg,data[i],train,val)['readout'];v=grouped(cfg,data[i],val,head)
                if passed(v) and (goal_candidate is None or f.j<scores[goal_candidate].j):goal_candidate=i
        if goal_candidate is not None:best_u=candidates[goal_candidate].copy();best=asdict(scores[goal_candidate]);reason='validation_gate'
        history.append({'generation':gen,'best':best,'failed':sum(not f.finite for f in scores),'sigma_mean':float(sigma.mean())})
        if gen:
            chosen=candidates[np.argsort(js)[:elite]];spread=np.sqrt(weights@(chosen-mean)**2);mean=weights@chosen;sigma=np.maximum(.7*spread+.3*sigma,.05)
        atomic(checkpoint,{'config_sha256':fingerprint,'generation':gen,'mean':mean.tolist(),'sigma':sigma.tolist(),'best_u':best_u.tolist(),'best':best,'last_improved':last,'rng':rng.bit_generator.state,'history':history})
        print(args.run,'gen',gen,'J',best['j'],'val',best['val_accuracy'],'per_op',best['val_per_op'],flush=True)
        if reason=='validation_gate':break
        if gen-last>=patience:reason='plateau';break
    theta=space.decode(best_u);[fitted]=collect(cfg,[theta],range(val.stop));head=single_readout(cfg,fitted,train,val)['readout']
    results={'train':grouped(cfg,fitted,train,head),'validation':grouped(cfg,fitted,val,head)}
    for name,streams in [('original_test',range(32,48)),('fresh_confirmation',range(20000,20064))]:
        [d]=collect(cfg,[theta],streams);assert d.finite;results[name]=grouped(cfg,d,streams,head)
    # One frozen head serves A/B, including level switches with shared history.
    atomic(final,{'family':'Teleneuron-Endophalon-3-AB','config':f'configs/endophalon3/{args.run}.json','theta':theta.to_payload(),'readout':head.to_payload(),'fitness':best,'metrics':results,'stop_reason':reason,'evaluations':1+24*(len(history)-1),'history':history,'fresh_gate_passed':passed(results['fresh_confirmation']),'elapsed_seconds_this_process':time.monotonic()-started})
    print('DONE',args.run,reason,json.dumps(results),flush=True)

if __name__=='__main__':main()
