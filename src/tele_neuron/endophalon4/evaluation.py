"""One-shot held-out comparisons, frozen operations and stream-level uncertainty."""
from pathlib import Path
import json,time
import numpy as np
from tele_neuron.endophalon.readout import RidgeReadout
from tele_neuron.endophalon3_software.session import Session
from tele_neuron.endophalon3_software.isa import Instruction,OPS
from .experiment import ROOT,config,models,data,observe,fit,memory,sha
from .runtime import Coupled
from .training import OUT,latest,verify,atomic


def operations(direction,split,features):
    d=data(direction,split);w=config()['washout'];out={}
    for name,mod,x in zip(['sender','receiver'],models(direction),[d['sender_features'],features]):
        targets=d[name+'_targets'][:,w:];ops=d[name+'_ops'][:,w:]
        hit=(mod.head.scores(x[:,w:].reshape(-1,16))[:,0].reshape(targets.shape)>=.5)==targets
        out[name]={'level':mod.level,'accuracy':float(hit.mean()),'n':hit.size,
                   'per_op':{op:{'accuracy':float(hit[ops==OPS.index(op)].mean()),
                                 'n':int((ops==OPS.index(op)).sum())} for op in mod.config.ops}}
    out['overall_accuracy']=float(np.mean([out[k]['accuracy'] for k in ['sender','receiver']]))
    return out


def bootstrap(coupled,baseline):
    delta=np.asarray(coupled)-np.asarray(baseline);rng=np.random.default_rng(441104)
    draws=delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(axis=1)
    return {'unit':'paired independent stream clusters','draws':2000,
            'difference':float(delta.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist()}


def software_reference(direction,split):
    """Actual saved-head predictions passed through explicit lag registers."""
    d=data(direction,split);sender,receiver=models(direction);cfg=config();sp=cfg['directions'][direction]['splits'][split]
    # Use the same source a bits, but an explicit XOR(a,0) transmission program.
    from tele_neuron.endophalon3_software.isa import random_stream
    bits=np.stack([random_stream(sender.level,sp['seed'],k,cfg['events']).a for k in sp['ids']])
    source=Session(sender,tuple(sp['ids']))
    dest=[Session(receiver,d['ids']) for _ in cfg['lags']];register=[];hits=[]
    for e in range(cfg['events']):
        result=source.step([Instruction(sender.level,'xor',int(b),0) for b in bits[:,e]])
        register.append(np.array([r.output for r in result]))
        columns=[]
        for lag,s in zip(cfg['lags'],dest):
            prediction=register[e-lag] if e>=lag else np.zeros(len(bits),dtype=int)
            out=s.step([Instruction(receiver.level,'xor',int(v),0) for v in prediction])
            columns.append(np.array([r.output for r in out]))
        if e>=cfg['washout']:
            hits.append(np.stack(columns,axis=-1)==d['targets'][:,e-cfg['washout']])
    hit=np.stack(hits,axis=1)
    return {'accuracy':float(hit.mean()),'per_lag':{str(l):float(hit[...,i].mean()) for i,l in enumerate(cfg['lags'])},
            'n_bits':hit.size,'protocol':'saved sender XOR(a,0) predicted-bit registers -> saved receiver XOR(register,0); synchronous independent clocks; separate probe program'}


def evaluate_split(direction,split,bundle):
    u=bundle['parameters'];head=RidgeReadout.from_payload(bundle['readout']);d=data(direction,split)
    all_metrics={};features={};stability={}
    for mode in ['coupled','uncoupled','zero','shuffled']:
        h=head if mode=='coupled' else fit(direction,u,mode)
        f,st=observe(direction,split,u,mode)
        features[mode]=f;stability[mode]=st
        all_metrics[mode]=memory(h,f,d['targets'])
    if not np.array_equal(features['uncoupled'],features['zero']):raise AssertionError('Zero/off mismatch')
    all_metrics['zero_selected_head']=memory(head,features['zero'],d['targets'])
    op={mode:operations(direction,split,features[mode]) for mode in ['uncoupled','coupled']}
    replay=RidgeReadout.from_payload(json.loads((OUT/direction/'model.json').read_text())['readout'])
    # Fresh full physical execution, not just re-scoring cached receiving features.
    f2,st2=observe(direction,split,u)
    if not np.array_equal(f2,features['coupled']):raise AssertionError('Physical replay differs')
    if memory(replay,f2,d['targets'])!=all_metrics['coupled']:raise AssertionError('Saved-head replay differs')
    # Inspectable evidence contains local features and labels, never used by runtime.
    path=ROOT/f'docs/endophalon4/{direction}_{split}_observations.npz'
    np.savez_compressed(path,**{k:v for k,v in features.items()},targets=d['targets'],
                        streams=np.asarray(d['ids']),sender_features=d['sender_features'],
                        sender_targets=d['sender_targets'],receiver_targets=d['receiver_targets'],
                        sender_ops=d['sender_ops'],receiver_ops=d['receiver_ops'])
    return {'memory':all_metrics,'operations':op,'stability':stability,
            'improvement_uncertainty':bootstrap(all_metrics['coupled']['stream_accuracy'],all_metrics['uncoupled']['stream_accuracy']),
            'software_register':software_reference(direction,split),'physical_replay_bitwise':True,
            'observations_sha256':sha(path)}


def long_horizon(direction,u,events=1024):
    cfg=config();sender,receiver=models(direction)
    ids=(78000 if direction=='A_to_B' else 79000,)
    live=Coupled(sender,receiver,ids,(ids[0]+1000000,),u)
    from tele_neuron.endophalon3_software.isa import random_stream
    a=random_stream(sender.level,99001,ids[0],events).channels(sender.config.encoding)
    b=random_stream(receiver.level,99002,ids[0]+1000000,events).channels(receiver.config.encoding)
    max_p=0.;max_v=0.
    for t in range(32+events*16):
        e,k=divmod(t-32,16)
        live.step(a[e:e+1] if e>=0 and k<sender.config.stream.pulse_steps else None,
                  b[e:e+1] if e>=0 and k<receiver.config.stream.pulse_steps else None)
        max_p=max(max_p,float(abs(live.receiver.positions).max()))
        max_v=max(max_v,float(abs(live.receiver.velocities).max()))
    return {'finite':True,'events':events,'ticks_each':live.sender.clock,'max_abs_receiver_position':max_p,'max_abs_receiver_velocity':max_v}


def run():
    report_path=ROOT/'docs/endophalon4/evaluation.json'
    marker=ROOT/'docs/endophalon4/confirmation.started.json'
    if report_path.exists():raise FileExistsError('Evaluation already complete; inspect saved evidence without rerunning confirmation')
    if marker.exists():raise FileExistsError('Confirmation already consumed or interrupted; never silently repeat')
    directions=list(config()['directions']);bundles={}
    for direction in directions:
        state=latest(direction);verify(state,direction)
        if state['generation']!=40 or state['evaluations']!=961:raise ValueError('Both searches must finish before held-out evaluation')
        bundle=json.loads((OUT/direction/'model.json').read_text())
        if bundle['parameters']!=state['best_u']:raise ValueError('Selection mismatch')
        bundles[direction]=bundle
    result={'schema':'endophalon4-evaluation/v1','directions':{},'confirmation_policy':'single invocation after both searches; immutable evidence for audits'}
    for d in directions:
        result['directions'][d]={'test':evaluate_split(d,'test',bundles[d]),
                                 'long_horizon':long_horizon(d,bundles[d]['parameters'])}
        atomic(ROOT/f'docs/endophalon4/{d}_test.json',result['directions'][d])
        print(d,'test complete',flush=True)
    with marker.open('x') as f:
        json.dump({'started_unix':time.time(),'model_sha256':{d:sha(OUT/d/'model.json') for d in directions}},f,indent=2)
        f.flush()
        import os
        os.fsync(f.fileno())
    for d in directions:
        r=evaluate_split(d,'confirmation',bundles[d]);result['directions'][d]['confirmation']=r
        atomic(ROOT/f'docs/endophalon4/{d}_confirmation.json',r)
        print(d,'confirmation accuracy',r['memory']['coupled']['accuracy'],flush=True)
    for d,r in result['directions'].items():
        c=r['confirmation'];m=c['memory'];op=c['operations']
        drop=op['uncoupled']['overall_accuracy']-op['coupled']['overall_accuracy']
        r['gates']={'memory_accuracy_75':m['coupled']['accuracy']>=.75,
            'improvement_10pp':m['coupled']['accuracy']-m['uncoupled']['accuracy']>=.10,
            'operation_drop_le_3pp':drop<=.03,'operation_drop':drop,
            'receiver_operation_drop':op['uncoupled']['receiver']['accuracy']-op['coupled']['receiver']['accuracy'],
            'all_lags_75':all(v>=.75 for v in m['coupled']['per_lag'].values()),
            'finite':all(s['finite'] for s in c['stability'].values()) and r['long_horizon']['finite'],
            'reproducible':c['physical_replay_bitwise']}
    result['primary_success']=all(all(r['gates'][k] for k in ['memory_accuracy_75','improvement_10pp','operation_drop_le_3pp','finite','reproducible']) for r in result['directions'].values())
    atomic(report_path,result)
    return result
