"""Read-only audit of the fixed-budget paired-bank training output."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tele_neuron.endophalon3_alpha.model import Model
from tele_neuron.endophalon3_alpha.train import TRAIN,VALIDATION,TEST,CONFIRMATION,observe,objective,metrics,decode

ROOT=Path(__file__).resolve().parents[1]


def audit(out):
    report=json.loads((out/'evaluation.json').read_text());state=json.loads((out/'checkpoint.json').read_text());settings=json.loads((out/'settings.json').read_text())
    assert report['settings']==settings
    assert report['status']=='completed' and state['generation']==120
    assert state['evaluations']==report['evaluations']==2881
    assert [h['generation'] for h in state['history']]==list(range(121))
    assert all(a['best']['j']>=b['best']['j'] for a,b in zip(state['history'],state['history'][1:]))
    assert [len(h['candidates']) for h in state['history']]==[1]+[24]*120
    assert sum(h['failed'] for h in state['history'])==report['failed_candidates']
    assert len(list((out/'checkpoints').glob('generation_*.json')))==121
    for name,sha in settings['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    start=ROOT/'models/endophalon3_alpha/20261102_seed1_binary.json'
    assert hashlib.sha256(start.read_bytes()).hexdigest()==settings['start_sha256']
    base=Model.load(start);saved=Model.load(out/'model.json');selected=decode(base,state['best_u'])
    for level in 'AB':assert selected.configs[level].theta_init.to_payload()==saved.configs[level].theta_init.to_payload()
    trainval=observe(saved,range(TRAIN.start,VALIDATION.stop));refit,f=objective(saved,trainval)
    for level in 'AB':
        for field in ('mean','scale','weights'):np.testing.assert_allclose(getattr(saved.banks[level].head,field),getattr(refit.banks[level].head,field),rtol=1e-10,atol=1e-10)
    for key in ('j','l_val','r_noise','r_drift','r_complexity'):assert np.isclose(f[key],report['fitness'][key],rtol=1e-10,atol=1e-10),key
    assert metrics(saved,trainval,TRAIN)==report['metrics']['train']
    assert metrics(saved,trainval,VALIDATION)==report['metrics']['validation']
    for name,ids in [('test',TEST),('confirmation',CONFIRMATION)]:
        with np.load(out/f'{name}_observations.npz',allow_pickle=False) as loaded:data=dict(loaded)
        assert metrics(saved,data,ids)==report['metrics'][name]
        chosen=range(ids.start,ids.start+3);fresh=observe(saved,chosen)
        for key in data:np.testing.assert_array_equal(fresh[key],data[key][:3])
    print('PASS: 120 generations, 2881 pairs, 121 checkpoints, unchanged sources/start, separate head refits, fitness, full held-out metrics and exact six-stream physical replay')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args();audit(args.directory)
