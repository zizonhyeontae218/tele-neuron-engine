"""Read-only saved-bundle checks, native/fresh accuracy replay and original-card identity."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from evaluate_endophalon3_software import SoftwareModel,ROOT,CARDS,summarize
from tele_neuron.endophalon3.isa import random_stream


def audit(out):
    manifest=json.loads((out/'manifest.json').read_text())
    for name,sha in manifest.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==sha,name
    report=json.loads((out/'evaluation.json').read_text());model=SoftwareModel.load(out/'model.json')
    with np.load(out/'observations.npz',allow_pickle=False) as observed:
        for level,path in CARDS.items():
            card=json.loads((ROOT/path).read_text());bank=model.parameters.banks[level];cfg=model.parameters.configs[level]
            assert bank.raw_config['theta_init']==card['theta']
            assert bank.head.to_payload()==card['readout']
            for prefix,ids,expected in [('native_',report['original_test_replay'][level]['streams'],report['original_test_replay'][level]),('fresh_',report['fresh_local_streams'],report['fresh_local'][level])]:
                x=observed[prefix+level];pred=(bank.head.scores(x.reshape(-1,16))[:,0]>=.5).reshape(x.shape[:2])
                inputs=[random_stream(level,cfg.physics.seed,s,cfg.stream.pulses) for s in ids]
                actual=summarize(level,inputs,pred,cfg.washout)
                assert actual['accuracy']==expected['accuracy'] and actual['per_op']==expected['per_op']
    print('PASS: saved bundle/observations hashes, exact imported theta/readouts, native and fresh local per-op accuracies')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args();audit(args.directory)
