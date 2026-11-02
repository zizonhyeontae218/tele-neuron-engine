"""Validate both native independent runs, before any old-version deletion."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tele_neuron.endophalon3_software.training import Module,observe,fitness,metrics,subset,decode,fit,new_state
ROOT=Path(__file__).resolve().parents[1]


def audit(out):
    report=json.loads((out/'evaluation.json').read_text());state=json.loads((out/'checkpoint.json').read_text());settings=json.loads((out/'settings.json').read_text());raw=settings['config'];ids=raw['training_stream_ids']
    assert state['settings_digest']==hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    assert report['status']=='completed' and state['generation']==120 and state['evaluations']==report['evaluations']==2881
    assert [h['generation'] for h in state['history']]==list(range(121)) and len(list((out/'checkpoints').glob('generation_*.json')))==121
    assert [len(h['candidates']) for h in state['history']]==[1]+[24]*120
    assert all(a['best']['j']>=b['best']['j'] for a,b in zip(state['history'],state['history'][1:]))
    assert sum(h['failed'] for h in state['history'])==report['failed_candidates']
    for name,sha in settings['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    model=Module.load(out/'model.json');selected=decode(raw,state['best_u']);assert selected.raw==model.raw and model.provenance['training_settings']==settings
    for generation in range(121):
        saved=json.loads((out/'checkpoints'/f'generation_{generation:03}.json').read_text())
        assert saved['generation']==generation and saved['evaluations']==1+24*generation and saved['settings_digest']==state['settings_digest']
        assert saved['best']==state['history'][generation]['best']
    data=observe(model,[*ids['train'],*ids['validation']]);refit,f=fitness(model,data,ids['train'],ids['validation'])
    for k in ['mean','scale','weights']:np.testing.assert_allclose(getattr(model.head,k),getattr(refit.head,k),rtol=1e-10,atol=1e-10)
    for k in ['j','l_val','r_noise','r_drift','r_complexity']:assert np.isclose(f[k],report['fitness'][k],rtol=1e-10,atol=1e-10)
    for name in ['train','validation']:assert metrics(model,subset(data,ids[name]))==report['metrics'][name]
    baseline=decode(raw,new_state(raw)['best_u']);baseline=fit(baseline,observe(baseline,ids['train']),ids['train'])
    for name in ['test','confirmation']:
        with np.load(out/f'{name}_observations.npz') as loaded:saved=dict(loaded)
        assert metrics(model,saved)==report['metrics'][name]
        replay=observe(model,ids[name][:3])
        for key in saved:np.testing.assert_array_equal(replay[key],saved[key][:3])
        assert metrics(baseline,observe(baseline,ids[name]))==report['baseline'][name]
    expected_target=.9 if model.level=='A' else .8
    assert report['user_target']=={'overall':expected_target,'passed':report['metrics']['confirmation']['accuracy']>=expected_target}
    assert report['accuracy_gate']==(report['metrics']['confirmation']['accuracy']>=.9 and all(v['accuracy']>=.85 for v in report['metrics']['confirmation']['per_op'].values()))
    print('PASS',model.level,'120 generations / 2881 candidates; source/head/fitness and six-stream exact physical replay',flush=True)
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);a=ap.parse_args();audit(a.directory)
