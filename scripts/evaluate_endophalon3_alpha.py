"""Bounded MVP: fixed bank physics, train-only A/B head refits and reset control."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tele_neuron.endophalon.readout import RidgeReadout
from tele_neuron.endophalon3.isa import OPS, Instruction, random_stream, evaluate
from tele_neuron.endophalon3_alpha.model import Model, Bank
from tele_neuron.endophalon3_alpha.session import Session

ROOT=Path(__file__).resolve().parents[1]
SPLITS={'train':range(30000,30024),'validation':range(30024,30032),'test':range(30032,30064),'confirmation':range(30064,30128)}


def record(model,ids,reset=False):
    session=Session(model,tuple(ids));x0,v0=session.positions.copy(),session.velocities.copy()
    inputs=[random_stream('AB',model.config.physics.seed,s,model.config.stream.pulses) for s in ids]
    features=[];jumps=[];age=np.full(len(ids),1000);ages=[]
    for k in range(model.config.stream.pulses):
        current=np.array([s.levels[k] for s in inputs])
        changed=current!=np.array([s.levels[k-1] for s in inputs]) if k else np.zeros(len(ids),bool)
        if reset:
            session.positions[changed]=x0[changed];session.velocities[changed]=v0[changed]
        age=np.where(changed,0,age+1);ages.append(age.copy())
        rows=session.step([Instruction('AB'[s.levels[k]],OPS[s.ops[k]],int(s.a[k]),int(s.b[k])) for s in inputs])
        features.append([r.features for r in rows]);jumps.append([r.switching_energy for r in rows])
    result={'features':np.array(features).transpose(1,0,2),'levels':np.stack([s.levels for s in inputs]),'ops':np.stack([s.ops for s in inputs]),'targets':np.stack([evaluate(s.ops,s.a,s.b)[0] for s in inputs]),'age':np.array(ages).T,'switching_energy':np.array(jumps).T,'streams':np.array(ids)}
    assert np.isfinite(result['features']).all()
    return result


def flattened(data,washout=4):
    return {k:(v[:,washout:].reshape(-1,v.shape[-1]) if k=='features' else v[:,washout:].ravel()) for k,v in data.items() if k!='streams'}


def score(model,data,single=None):
    d=flattened(data,model.config.washout);scores=np.empty(len(d['targets']))
    for bit,level in enumerate('AB'):
        mask=d['levels']==bit
        scores[mask]=(single or model.banks[level].head).scores(d['features'][mask])[:,0]
    hits=(scores>=.5)==d['targets'];out={'overall':float(hits.mean()),'levels':{}}
    for bit,level in enumerate('AB'):
        mask=d['levels']==bit
        operations={OPS[i]:{'accuracy':float(hits[mask&(d['ops']==i)].mean()),'n':int((mask&(d['ops']==i)).sum())} for i in sorted(set(d['ops'][mask]))}
        row={'accuracy':float(hits[mask].mean()),'n':int(mask.sum()),'per_op':operations}
        for age in (0,1):
            chosen=mask&(d['age']==age);row['first_after_switch' if age==0 else 'second_after_switch']={'accuracy':float(hits[chosen].mean()) if chosen.any() else None,'n':int(chosen.sum())}
        out['levels'][level]=row
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',required=True);args=ap.parse_args()
    out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Never overwrite an experiment')
    out.mkdir(parents=True,exist_ok=True)
    baseline=Model.from_cards(ROOT/'models/endophalon3/20261031_a_binary_s1.json',ROOT/'models/endophalon3/20261031_b_binary_s1.json',ROOT)
    started=time.monotonic();all_data=record(baseline,range(30000,30128))
    split_data={name:{k:(v[np.isin(all_data['streams'],list(ids))]) for k,v in all_data.items()} for name,ids in SPLITS.items()}
    train=flattened(split_data['train']);banks={}
    for bit,level in enumerate('AB'):
        mask=train['levels']==bit;cfg=baseline.configs[level]
        banks[level]=Bank(baseline.banks[level].raw_config,RidgeReadout(cfg.ridge_alpha).fit(train['features'][mask],train['targets'][mask,None]))
    fitted=Model(banks,{**baseline.provenance,'method':'fixed independent physics; separate A/B train-only Ridge refit','train_streams':list(SPLITS['train'])})
    single=RidgeReadout(baseline.config.ridge_alpha).fit(train['features'],train['targets'][:,None])
    metrics={name:{'original_heads':score(baseline,data),'two_fitted_heads':score(fitted,data),'pooled_head_control':score(fitted,data,single)} for name,data in split_data.items()}
    print('Head fits complete; evaluating reset-at-switch control',flush=True)
    reset_data=record(fitted,range(30032,30128),reset=True)
    for name in ('test','confirmation'):
        mask=np.isin(reset_data['streams'],list(SPLITS[name]));part={k:v[mask] for k,v in reset_data.items()};metrics[name]['same_heads_reset_at_switch']=score(fitted,part)
    card=out/'model.json';fitted.save(card)
    restored=Model.load(card)
    for name,data in split_data.items():assert score(restored,data)==metrics[name]['two_fitted_heads']
    np.savez_compressed(out/'observations.npz',**all_data)
    np.savez_compressed(out/'reset_observations.npz',**reset_data)
    atomic_roundtrip=Session.restore(fitted,Session(fitted).snapshot());assert atomic_roundtrip.events==0
    payload={'status':'completed','physics_training':'none','head_training':'one train-only Ridge fit for each of A/B','streams':{n:list(ids) for n,ids in SPLITS.items()},'seed':1,'events':64,'washout':4,'active_zones_per_event':16,'stored_zone_banks':2,'metrics':metrics,'elapsed_seconds':time.monotonic()-started,'switching_energy':{'maximum_abs':float(np.abs(all_data['switching_energy']).max()),'note':'Total kinetic + spring + anchor energy jump from parameter changes at fixed positions/velocities; not conservation error'},'reset_control':'At profile changes reset only positions/velocities to original jitter states; preserve clock and RNG, no extra warmup. Evaluate with the same fitted heads.','limitations':['One physical/data seed and one predeclared binary bank pair; no seed or encoding ranking.','Observed 64-event streams; no infinite-horizon claim.','Head fit and physics-state continuity do not guarantee useful cross-bank memory.','External level dispatcher selects A/B; opcode routing remains inside each bank.']}
    (out/'evaluation.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(files,indent=2)+'\n')
    for name in ('test','confirmation'):
        print(name,json.dumps(metrics[name]),flush=True)

if __name__=='__main__':main()
