"""No-training replay of imported native cards plus predeclared fresh local streams."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tele_neuron.endophalon3.isa import Instruction,OPS,random_stream,evaluate
from tele_neuron.endophalon3_software import SoftwareModel,Router

ROOT=Path(__file__).resolve().parents[1]
CARDS={'A':'models/endophalon3/20261031_a_binary_s2.json','B':'models/endophalon3/20261031_b_binary_s1.json'}


def summarize(level,inputs,predictions,washout):
    targets=np.stack([evaluate(s.ops,s.a,s.b)[0] for s in inputs])[:,washout:]
    ops=np.stack([s.ops for s in inputs])[:,washout:];hit=predictions[:,washout:]==targets
    return {'accuracy':float(hit.mean()),'n':int(hit.size),'per_op':{OPS[i]:float(hit[ops==i].mean()) for i in sorted(set(ops.ravel()))}}


def replay(model,streams,levels):
    ids=tuple(streams);router=Router(model,ids);events=model.parameters.config.stream.pulses
    inputs={level:[random_stream(level,model.parameters.config.physics.seed,s,events) for s in ids] for level in levels}
    predictions={level:[] for level in levels};features={level:[] for level in levels}
    for k in range(events):
        for level in levels:
            rows=router.dispatch(level,[Instruction(level,OPS[p.ops[k]],int(p.a[k]),int(p.b[k])) for p in inputs[level]])
            predictions[level].append([r.output for r in rows]);features[level].append([r.features for r in rows])
    stats={level:summarize(level,inputs[level],np.array(predictions[level]).T,model.parameters.configs[level].washout) for level in levels}
    arrays={level:np.array(features[level]).transpose(1,0,2) for level in levels}
    return stats,arrays


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',required=True);args=ap.parse_args();out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Existing results must not be overwritten')
    out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    model=SoftwareModel.from_cards(ROOT/CARDS['A'],ROOT/CARDS['B'],ROOT)
    report={'training':'none','architecture':'two paused-independent physical sessions with explicit software output registers','selected_cards':CARDS,'original_test_replay':{},'fresh_local_streams':list(range(50000,50032))}
    arrays={}
    for level,path in CARDS.items():
        card=json.loads((ROOT/path).read_text());cfg=model.parameters.configs[level];ids=cfg.stream_ranges()[2]
        stats,x=replay(model,ids,[level]);expected=card['metrics']['test']['trained']['single_readout']
        assert stats[level]['accuracy']==expected['accuracy'] and stats[level]['per_op']==expected['per_op']
        report['original_test_replay'][level]={'streams':list(ids),'exact_overall_and_per_op':True,**stats[level]}
        arrays['native_'+level]=x[level]
        print(level,'native test exact replay',stats[level]['accuracy'],flush=True)
    report['fresh_local'],fresh=replay(model,range(50000,50032),['A','B']);arrays.update({'fresh_'+level:x for level,x in fresh.items()})
    # Each fresh step alternates A then B, while every local input stream is preserved.
    report['fresh_schedule']='A[k],B[k] alternating; paused inactive model; labels use each model local history'
    report['fresh_joint_input_history_task']=False
    report['limitations']=['Deployment cards chosen from already viewed historical test results.','One stored binary pair and physical/data seed 1; no model ranking claim.','Independent local-stream accuracy is not accuracy on the old shared-global-history task.','Register wiring preserves model errors; no end-to-end chain accuracy is established by marginal scores.','Session inactivity freezes local time; real-time idle dynamics are not simulated.']
    model.save(out/'model.json');np.savez_compressed(out/'observations.npz',**arrays)
    report['elapsed_seconds']=time.monotonic()-start
    (out/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Fresh independent local tasks:',report['fresh_local'],flush=True)

if __name__=='__main__':main()
