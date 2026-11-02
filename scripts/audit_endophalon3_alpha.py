"""Read-only model/evidence audit, including fresh physical replay of three streams."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from evaluate_endophalon3_alpha import Model, Bank, RidgeReadout, record, flattened, score, SPLITS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory');args=ap.parse_args();root=Path(args.directory)
    for name,sha in json.loads((root/'manifest.json').read_text()).items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
    model=Model.load(root/'model.json');evaluation=json.loads((root/'evaluation.json').read_text())
    with np.load(root/'observations.npz',allow_pickle=False) as loaded:data=dict(loaded)
    positions={int(v):i for i,v in enumerate(data['streams'])};assert len(positions)==len(data['streams'])
    split_sets=[set(v) for v in evaluation['streams'].values()]
    assert all(not a&b for i,a in enumerate(split_sets) for b in split_sets[i+1:])
    train=flattened({k:v[np.isin(data['streams'],evaluation['streams']['train'])] for k,v in data.items()})
    for bit,level in enumerate('AB'):
        mask=train['levels']==bit;head=RidgeReadout(model.configs[level].ridge_alpha).fit(train['features'][mask],train['targets'][mask,None])
        for field in ('mean','scale','weights'):
            np.testing.assert_allclose(getattr(head,field),getattr(model.banks[level].head,field),rtol=1e-12,atol=1e-12)
    for name,ids in evaluation['streams'].items():
        subset={k:v[np.isin(data['streams'],ids)] for k,v in data.items()}
        assert score(model,subset)==evaluation['metrics'][name]['two_fitted_heads'],name
    ids=evaluation['streams']['confirmation'][:3];replayed=record(model,ids)
    for key in replayed:
        np.testing.assert_array_equal(replayed[key],data[key][[positions[s] for s in ids]])
    print('PASS: manifest, disjoint splits, train-only head refits, all split metrics and three-stream exact physical replay')

if __name__=='__main__':main()
