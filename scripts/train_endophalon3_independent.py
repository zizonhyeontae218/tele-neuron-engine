"""One independent 120-generation run; separate command used for A and B."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,hashlib,json,os,platform,subprocess,sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tele_neuron.endophalon3_software.training import Module,observe,subset,fit,fitness,metrics,decode,candidate,new_state,advance
ROOT=Path(__file__).resolve().parents[1]


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


def run(level,out,workers):
    config=ROOT/f'configs/endophalon3_software/{level}.json';raw=json.loads(config.read_text());ids=raw['training_stream_ids'];m=Module(raw)
    assert (raw['outer']['generations'],raw['outer']['population'],raw['outer']['elite'])==(120,24,6)
    assert len({v for split in ids.values() for v in split})==sum(len(v) for v in ids.values()),'Overlapping splits'
    base=ROOT/'src/tele_neuron';retired={'endophalon3','endophalon3_alpha','endophalon3t'}
    sources=[config,*sorted(p for p in base.rglob('*.py') if p.relative_to(base).parts[0] not in retired),Path(__file__).resolve()]
    settings={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'level':level,'generations':120,'population':24,'elite':6,'search_seed':raw['outer']['seed'],'streams':ids,'config':raw,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    digest=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest();out.mkdir(parents=True,exist_ok=True);checkpoint=out/'checkpoint.json'
    if (out/'model.json').exists():raise FileExistsError('Completed run must not repeat')
    if checkpoint.exists():
        state=json.loads(checkpoint.read_text());assert state['settings_digest']==digest,'Checkpoint/source mismatch'
    else:
        if (out/'settings.json').exists():assert json.loads((out/'settings.json').read_text())==settings
        atomic(out/'settings.json',settings);state=new_state(raw)
    atomic(out/'environment.json',{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'workers':workers});started=time.monotonic();archive=out/'checkpoints';archive.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        def score(u):return list(pool.map(candidate,[(raw,v.tolist()) for v in u]))
        while state['generation']<120:
            state=advance(state,score);state['settings_digest']=digest
            if not state['best']['finite']:raise FloatingPointError('No valid initial model')
            atomic(checkpoint,state);atomic(archive/f"generation_{state['generation']:03}.json",{k:v for k,v in state.items() if k!='history'})
            print(level,'generation',state['generation'],'evaluations',state['evaluations'],'J',state['best']['j'],'validation',state['best']['validation']['accuracy'],'failed',state['history'][-1]['failed'],flush=True)
    selected=decode(raw,state['best_u']);data=observe(selected,[*ids['train'],*ids['validation']]);fitted,f=fitness(selected,data,ids['train'],ids['validation'])
    assert state['evaluations']==2881 and len(state['history'])==121
    for key in ['j','l_val','r_noise','r_drift','r_complexity']:assert np.isclose(f[key],state['best'][key],rtol=1e-10,atol=1e-10)
    baseline=decode(raw,new_state(raw)['best_u']);initial_data=observe(baseline,ids['train']);baseline=fit(baseline,initial_data,ids['train'])
    fitted.provenance={'training_settings':settings,'selection':'validation objective only; no held-out selection'}
    report={'status':'completed','level':level,'generations':120,'evaluations':2881,'fitness':f,'initial_fitness':state['history'][0]['best'],'failed_candidates':sum(h['failed'] for h in state['history']),'elapsed_seconds':time.monotonic()-started,'metrics':{'train':metrics(fitted,subset(data,ids['train'])),'validation':metrics(fitted,subset(data,ids['validation']))},'baseline':{}}
    for name in ['test','confirmation']:
        observed=observe(fitted,ids[name]);base_data=observe(baseline,ids[name]);report['metrics'][name]=metrics(fitted,observed);report['baseline'][name]=metrics(baseline,base_data);np.savez_compressed(out/f'{name}_observations.npz',**observed)
    report['user_target']={'overall':.9 if level=='A' else .8,'passed':report['metrics']['confirmation']['accuracy']>=(.9 if level=='A' else .8)}
    report['accuracy_gate']=report['metrics']['confirmation']['accuracy']>=.9 and all(v['accuracy']>=.85 for v in report['metrics']['confirmation']['per_op'].values())
    atomic(out/'evaluation.json',report);fitted.save(out/'model.json');saved=Module.load(out/'model.json')
    with np.load(out/'confirmation_observations.npz') as loaded:assert metrics(saved,dict(loaded))==report['metrics']['confirmation']
    print('COMPLETE',level,'confirmation',report['metrics']['confirmation']['accuracy'],'gate',report['accuracy_gate'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--level',choices=['A','B'],required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();run(a.level,a.output_dir,a.workers)
