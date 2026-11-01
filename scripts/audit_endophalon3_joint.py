"""Replay frozen joint AB cards without modifying published evidence or retraining."""
from dataclasses import replace
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import hashlib
import numpy as np
from tele_neuron.endophalon3.config import load_config
from tele_neuron.endophalon3.core import collect, dataset, fitness
from tele_neuron.endophalon3.theta import Theta3
from tele_neuron.endophalon.readout import RidgeReadout
from train_endophalon3_joint import grouped, passed


def audit(root, names=None):
    base=root/'docs/endophalon3_joint_runs/20261101'
    for name,sha in json.loads((base/'artifact_manifest.json').read_text())['sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
    for path in sorted((root/'models/endophalon3_joint').glob('*.json')):
        if names and path.stem not in names: continue
        card=json.loads(path.read_text());cfg=replace(load_config(root/card['config']),workers=1)
        theta=Theta3.from_payload(card['theta']);head=RidgeReadout.from_payload(card['readout'])
        train,val,test=cfg.stream_ranges();[data]=collect(cfg,[theta],range(test.stop));assert data.finite
        x,y,_=dataset(cfg,data,train);refit=RidgeReadout(cfg.ridge_alpha).fit(x,y)
        for field in ['mean','scale','weights']:
            assert np.allclose(getattr(head,field),getattr(refit,field),rtol=1e-10,atol=1e-10),field
        f=fitness(cfg,data,train,val)
        for key in ['j','l_val','r_noise','r_drift','r_complexity','val_accuracy']:
            assert np.isclose(getattr(f,key),card['fitness'][key],rtol=1e-10,atol=1e-10),key
        for label,streams in [('train',train),('validation',val),('original_test',test)]:
            assert grouped(cfg,data,streams,head)==card['metrics'][label],label
        confirm=range(20000,20064);assert not set(confirm)&set(range(test.stop))
        [fresh]=collect(cfg,[theta],confirm);assert fresh.finite
        assert grouped(cfg,fresh,confirm,head)==card['metrics']['fresh_confirmation']
        assert passed(card['metrics']['fresh_confirmation'])==card['fresh_gate_passed']
        checkpoint=json.loads((base/path.stem/'checkpoint.json').read_text())
        assert checkpoint['generation']==120 and len(card['history'])==121 and card['evaluations']==2881
        # encode clips probabilities at 1e-4; it is not an exact inverse at bounds.
        assert cfg.space.decode(np.array(checkpoint['best_u'])).to_payload()==card['theta']
        print(path.stem,'PASS: exact per-level/per-operation replay; refit, fitness, finite, hashes, checkpoint',flush=True)

if __name__=='__main__':audit(Path.cwd(),set(sys.argv[1:]))
