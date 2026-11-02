"""Run the owner's fixed 120-generation paired-bank physics experiment."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tele_neuron.endophalon3_alpha.model import Model
from tele_neuron.endophalon3_alpha.train import TRAIN, VALIDATION, TEST, CONFIRMATION, observe, objective, metrics, encode, decode, candidate, new_state, generation, passed

ROOT=Path(__file__).resolve().parents[1]
START=ROOT/'models/endophalon3_alpha/20261102_seed1_binary.json'


def atomic(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


def run(out,workers):
    base=Model.load(START);start_sha=hashlib.sha256(START.read_bytes()).hexdigest()
    settings={'schema':'endophalon3-alpha-training/v1','start_sha256':start_sha,'generations':120,'population':24,'elite':6,'init_sigma':.5,'min_sigma':.05,'search_seed':1,'train':list(TRAIN),'validation':list(VALIDATION),'test':list(TEST),'confirmation':list(CONFIRMATION),'objective':'mean(A_MSE,B_MSE)+0.05 mean(bank_noise)+0.1 per_stream_kinetic_log_drift+0.01 mean(bank_weight_complexity)','source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),ROOT/'src/tele_neuron/endophalon3_alpha/train.py',ROOT/'src/tele_neuron/endophalon3_alpha/model.py',ROOT/'src/tele_neuron/endophalon3_alpha/session.py',ROOT/'src/tele_neuron/endophalon3/engine.py',ROOT/'src/tele_neuron/endophalon3/isa.py',ROOT/'src/tele_neuron/endophalon3/theta.py',ROOT/'src/tele_neuron/endophalon/readout.py']}}
    digest=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True,exist_ok=True);checkpoint=out/'checkpoint.json'
    if (out/'model.json').exists():raise FileExistsError('Completed run must not be overwritten')
    if checkpoint.exists():
        state=json.loads(checkpoint.read_text())
        if state.get('settings_digest')!=digest:raise ValueError('Checkpoint/settings mismatch; refusing a new search')
    else:
        if (out/'settings.json').exists():
            if json.loads((out/'settings.json').read_text())!=settings:raise ValueError('Settings already differ')
        atomic(out/'settings.json',settings);state=new_state(base)
    environment={'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'workers':workers}
    atomic(out/'environment.json',environment);started=time.monotonic()
    archive=out/'checkpoints';archive.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        def evaluate(candidates):return list(pool.map(candidate,[(base.to_payload(),u.tolist()) for u in candidates]))
        while state['generation']<120:
            state=generation(state,evaluate);state['settings_digest']=digest
            if not state['best']['finite']:raise FloatingPointError('Initial candidate invalid; no useful search baseline')
            atomic(checkpoint,state)
            slim={k:v for k,v in state.items() if k!='history'}
            atomic(archive/f"generation_{state['generation']:03}.json",slim)
            validation=state['best']['validation']
            print('generation',state['generation'],'evaluations',state['evaluations'],'J',state['best']['j'],'validation',validation['accuracy'],'A',validation['levels']['A']['accuracy'],'B',validation['levels']['B']['accuracy'],'failed',state['history'][-1]['failed'],flush=True)
    assert state['evaluations']==2881 and len(state['history'])==121
    selected=decode(base,state['best_u']);fitting=observe(selected,range(TRAIN.start,VALIDATION.stop));fitted,fitness=objective(selected,fitting)
    for key in ('j','l_val','r_noise','r_drift','r_complexity'):
        assert np.isclose(fitness[key],state['best'][key],rtol=1e-10,atol=1e-10),key
    fitted.provenance={**base.provenance,'training':settings,'selection':'minimum train/validation objective; held-out sets excluded'}
    report={'status':'completed','settings':settings,'environment':environment,'evaluations':2881,'generations':120,'fitness':fitness,'initial_fitness':state['history'][0]['best'],'failed_candidates':sum(h['failed'] for h in state['history']),'elapsed_seconds_this_process':time.monotonic()-started,'metrics':{'train':metrics(fitted,fitting,TRAIN),'validation':metrics(fitted,fitting,VALIDATION)},'baseline':{}}
    for name,ids in [('test',TEST),('confirmation',CONFIRMATION)]:
        data=observe(fitted,ids);baseline=observe(base,ids);reset=observe(fitted,ids,reset=True)
        report['metrics'][name]=metrics(fitted,data,ids);report['baseline'][name]=metrics(base,baseline,ids)
        report.setdefault('reset_control',{})[name]=metrics(fitted,reset,ids)
        report.setdefault('maximum_abs_switching_energy',{})[name]=float(np.abs(data['switching_energy']).max())
        np.savez_compressed(out/f'{name}_observations.npz',**data)
    report['confirmation_gate_passed']=passed(report['metrics']['confirmation'])
    atomic(out/'evaluation.json',report);fitted.save(out/'model.json')
    # Reopen the serialized artifact and require exact held-out accuracy replay.
    saved=Model.load(out/'model.json')
    with np.load(out/'confirmation_observations.npz') as loaded:
        assert metrics(saved,dict(loaded),CONFIRMATION)==report['metrics']['confirmation']
    print('COMPLETE; confirmation gate:',report['confirmation_gate_passed'],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',required=True);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    if args.workers<1:ap.error('workers must be positive')
    run(Path(args.output_dir),args.workers)

if __name__=='__main__':main()
