"""Run the predeclared routing-loss experiment and verify saved A models."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tele_neuron.endophalon.readout import RidgeReadout
from tele_neuron.endophalon2_pro.core import drift
from tele_neuron.endophalon3.config import load_config
from tele_neuron.endophalon3.core import collect, dataset, fitness, operation_balanced_loss
from tele_neuron.endophalon3.theta import Theta3
from diagnose_endophalon3_routing import metrics

RUN=ROOT/'docs/endophalon3_routing_runs'
BASE='24ebc6bb4133634380d7122070465e8445de014e'
CONFIRM=range(10000,10016)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify_baseline():
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE,'--','models','configs','docs/endophalon3_runs','src/tele_neuron/endophalon3/engine.py','src/tele_neuron/endophalon3/isa.py','src/tele_neuron/endophalon3/theta.py','src/tele_neuron/endophalon/readout.py'],cwd=ROOT,text=True).splitlines()
    checked={}
    for p in paths:
        actual=(ROOT/p).read_bytes();expected=subprocess.check_output(['git','show',BASE+':'+p],cwd=ROOT)
        assert actual==expected,p
        checked[p]=hashlib.sha256(actual).hexdigest()
    save(RUN/'baseline_preservation.json',{'base':BASE,'all_unchanged':True,'sha256':checked})


def legacy_validation():
    output={}
    for seed in [1,2]:
        card=json.loads((ROOT/f'models/endophalon3/20261031_a_binary_s{seed}.json').read_text())
        cfg=replace(load_config(ROOT/card['config']),workers=1)
        train,val,_=cfg.stream_ranges();[data]=collect(cfg,[Theta3.from_payload(card['theta'])],range(val.stop))
        f=fitness(cfg,data,train,val);old=card['metrics']['fitness_trained']
        errors={key:float(abs(getattr(f,key)-old[key])) for key in ['j','l_val','r_noise','r_drift','r_complexity','val_accuracy']}
        assert max(errors.values())<1e-10,errors
        assert f.val_per_op==old['val_per_op']
        output[str(seed)]={'scalar_abs_errors':errors,'per_op_accuracy_exact':True}
    save(RUN/'legacy_validation.json',output)


def execute(seed):
    name=f'a_binary_s{seed}';meta=RUN/(name+'.run.json');log=RUN/(name+'.log')
    card=ROOT/f'models/endophalon3_routing/20261031_{name}.json'
    if meta.exists() and json.loads(meta.read_text()).get('status')=='completed' and card.exists():return
    assert not any(p.exists() for p in [meta,log,card]),'Do not overwrite an existing attempt'
    cfg=load_config(ROOT/'configs/endophalon3/level_a_routing.json')
    assert (cfg.network.nodes,cfg.physics.substeps,cfg.zones,cfg.outer['generations'],cfg.outer['population'])==(144,2,16,40,24)
    assert cfg.objective['worst_op_weight']==.5
    cmd=[sys.executable,'-u','-m','tele_neuron.endophalon3','--config','configs/endophalon3/level_a_routing.json','--seed',str(seed),'--workers','4','--output',str(card.relative_to(ROOT))]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(ROOT/'src'))
    record={'name':name,'argv':cmd,'base_sha':BASE,'checkout_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'start_utc':datetime.now(timezone.utc).isoformat(),'status':'running','search_seed':seed,'physics_dataset_seed':1,
            'nodes':144,'substeps':2,'zones':16,'generations':40,'population':24,'workers':4,
            'worst_op_weight':.5,'config_sha256':hashlib.sha256((ROOT/'configs/endophalon3/level_a_routing.json').read_bytes()).hexdigest(),
            'environment':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform()}}
    save(meta,record);start=time.monotonic()
    with log.open('x') as stream:
        stream.write('$ '+' '.join(cmd)+'\n');stream.flush()
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            rc=process.wait(timeout=3300)
            record.update(returncode=rc,status='completed' if rc==0 and card.exists() else 'failed')
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM);process.wait()
            record.update(returncode=process.returncode,status='timeout')
    record.update(seconds=time.monotonic()-start,end_utc=datetime.now(timezone.utc).isoformat())
    if card.exists():record['card']=str(card.relative_to(ROOT))
    save(meta,record);print(name,record['status'],record['seconds'],flush=True)


def score_split(cfg,data,split,readout):
    x,y,ops=dataset(cfg,data,split);raw=readout.scores(x)[:,0];target=y[:,0]
    result=metrics(raw,target,ops)
    balanced,mean,worst=operation_balanced_loss(raw,target,ops,.5)
    one,zero=raw[target==1],raw[target==0]
    noise=float(.5*(one.std()+zero.std())/(abs(one.mean()-zero.mean())+1e-6))
    d=drift(cfg,data,split);complexity=float(np.mean(readout.weights[:-1]**2))
    result.update(l_balanced=balanced,l_mean=mean,l_worst=worst,r_noise=noise,r_drift=d,r_complexity=complexity,
                  j_split=balanced+.05*noise+.1*d+.01*complexity)
    return result


def gate(metric):
    return metric['accuracy']>=.9 and all(v['accuracy']>=.85 for v in metric['per_op'].values())


def evaluate(seed):
    name=f'a_binary_s{seed}';path=RUN/(name+'.evaluation.json')
    if path.exists():return json.loads(path.read_text())
    meta=json.loads((RUN/(name+'.run.json')).read_text())
    if meta['status']!='completed':return {'name':name,'status':meta['status'],'pass_both':False}
    card=json.loads((ROOT/meta['card']).read_text());cfg=replace(load_config(ROOT/card['config']),workers=1)
    train,val,test=cfg.stream_ranges();theta=Theta3.from_payload(card['theta']);readout=RidgeReadout.from_payload(card['readout'])
    assert card['metrics']['evaluations']==961 and len(card['metrics']['history'])==41
    [data]=collect(cfg,[theta],range(test.stop),nodes=True);assert data.finite
    x,y,_=dataset(cfg,data,train);refit=RidgeReadout(cfg.ridge_alpha).fit(x,y)
    assert np.allclose(refit.weights,readout.weights,rtol=1e-10,atol=1e-10)
    f=fitness(cfg,data,train,val);expected=card['metrics']['fitness_trained']
    for key in ['j','l_val','l_val_mean','l_val_worst','r_noise','r_drift','r_complexity']:
        assert np.isclose(getattr(f,key),expected[key],rtol=1e-10,atol=1e-10),(key,getattr(f,key),expected[key])
    splits={label:score_split(cfg,data,split,readout) for label,split in [('train',train),('validation',val),('test',test)]}
    reported=card['metrics']['test']['trained']['single_readout']
    assert splits['test']['accuracy']==reported['accuracy']
    assert {k:v['accuracy'] for k,v in splits['test']['per_op'].items()}==reported['per_op']
    # First use of the predeclared new confirmation streams, after theta selection.
    assert not set(CONFIRM)&set(range(test.stop))
    [new]=collect(cfg,[theta],CONFIRM,nodes=True);assert new.finite
    confirmation=score_split(cfg,new,CONFIRM,readout)
    result={'name':name,'status':'audited','splits':splits,'confirmation':confirmation,'confirmation_streams':list(CONFIRM),
            'pass_test':gate(splits['test']),'pass_confirmation':gate(confirmation),
            'pass_both':gate(splits['test']) and gate(confirmation),'fitness_validation':asdict(f),
            'failed_candidates':sum(v['failed'] for v in card['metrics']['history']),
            'finite':True,'max_abs_displacement_test':float(np.abs(data.nodes).max()),
            'max_abs_displacement_confirmation':float(np.abs(new.nodes).max()),
            'controls':card['metrics']['test'],'timing':meta,
            'objective_note':'l_val is mean/worst mixture; drift in selected J uses train+validation union. j_split uses only that split.'}
    save(path,result);return result


def main():
    RUN.mkdir(parents=True,exist_ok=True);verify_baseline()
    if not (RUN/'legacy_validation.json').exists():legacy_validation()
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(execute,[1,2]))
    results=[evaluate(seed) for seed in [1,2]]
    passed=all(r.get('pass_both',False) for r in results)
    save(RUN/'gate.json',{'seeds':results,'b_allowed':passed,'criteria':'Both A seeds: >=90% overall and >=85% every op on original test and untouched confirmation.'})
    verify_baseline()
    print('A complete; B gate:',passed,flush=True)

if __name__=='__main__':main()
