"""Audit saved evidence without consuming confirmation physics again."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,hashlib,tarfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tele_neuron.endophalon4.training import latest,verify,load,OUT,advance_record,atomic
from tele_neuron.endophalon4.experiment import ROOT,config,memory,sha,fit
from tele_neuron.endophalon4.evaluation import bootstrap
from tele_neuron.endophalon.readout import RidgeReadout


def main():
    cfg=config();seen_ids=set();seen_seeds=set();result={}
    preservation=json.loads((ROOT/'docs/endophalon4/preservation_baseline.json').read_text())
    for name,digest in preservation['files'].items():
        assert sha(ROOT/name)==digest,('Protected historical file changed',name)
    manifest=json.loads((ROOT/'docs/endophalon4/checkpoint_manifest.json').read_text())
    for direction,archive in manifest['directions'].items():
        assert sha(ROOT/archive['archive'])==archive['sha256']
        assert sha(OUT/direction/'model.json')==archive['model_sha256']
        with tarfile.open(ROOT/archive['archive'],'r:gz') as tar:
            assert sorted(tar.getnames())==sorted(archive['members'])
            for entry in tar:
                assert hashlib.sha256(tar.extractfile(entry).read()).hexdigest()==archive['members'][entry.name]['file_sha256']
    marker=json.loads((ROOT/'docs/endophalon4/confirmation.started.json').read_text())
    for direction,digest in marker['model_sha256'].items():
        assert sha(OUT/direction/'model.json')==digest,'Model changed after confirmation began'
    evaluation=json.loads((ROOT/'docs/endophalon4/evaluation.json').read_text())
    for d,settings in cfg['directions'].items():
        for split,spec in settings['splits'].items():
            assert not seen_ids.intersection(spec['ids']);seen_ids.update(spec['ids'])
            assert spec['seed'] not in seen_seeds;seen_seeds.add(spec['seed'])
            assert not set(spec['ids']).intersection(set(range(60000,60120))|set(range(61000,61128)))
        state=latest(d);verify(state,d);assert state['generation']==40 and state['evaluations']==961
        assert [(s['start_generation'],s['end_generation']) for s in state['segments']]==[(i,i+4) for i in range(1,41,5)]
        # Real candidate-result history replay: no extra physical candidates/search budget.
        old=load(OUT/d/'checkpoints/generation_000.json')
        gen1=load(OUT/d/'checkpoints/generation_001.json')
        captured=[]
        def score(u):captured.extend(u.tolist());return gen1['history'][1]['candidates']
        replay=advance_record(old,score)
        for key in ['mean','sigma','variance','best_u','best','rng','evaluations','history']:
            assert replay[key]==gen1[key],key
        # Reconstruct all candidate coordinates and count identities from RNG states.
        for g in range(1,41):
            prev=load(OUT/d/'checkpoints'/f'generation_{g-1:03}.json')
            nxt=load(OUT/d/'checkpoints'/f'generation_{g:03}.json')
            def recorded_score(u):return nxt['history'][g]['candidates']
            pred=advance_record(prev,recorded_score)
            for k in ['mean','sigma','best_u','rng','evaluations','best']:
                assert pred[k]==nxt[k],(g,k)
        bundle=json.loads((OUT/d/'model.json').read_text());assert bundle['parameters']==state['best_u']
        head=RidgeReadout.from_payload(bundle['readout'])
        train_only=fit(d,bundle['parameters'])
        for key in ['mean','scale','weights']:
            np.testing.assert_array_equal(getattr(head,key),getattr(train_only,key))
        for split in ['test','confirmation']:
            p=ROOT/f'docs/endophalon4/{d}_{split}_observations.npz'
            r=evaluation['directions'][d][split]
            assert sha(p)==r['observations_sha256']
            with np.load(p) as obs:
                assert np.array_equal(obs['uncoupled'],obs['zero'])
                for key in obs.files:assert np.isfinite(obs[key]).all()
                assert memory(head,obs['coupled'],obs['targets'])==r['memory']['coupled']
                ci=bootstrap(r['memory']['coupled']['stream_accuracy'],r['memory']['uncoupled']['stream_accuracy'])
                assert ci==r['improvement_uncertainty']
        result[d]={'generations':40,'evaluations':961,'segments':8,'all_checkpoint_transitions_replayed':True,
                   'saved_model_evidence_reproduced':True,'train_only_head_refit_bitwise':True,'protected_historical_files':len(preservation['files']),'checkpoint_archive_verified':True,'models_unchanged':state['provenance']['models'],
                   'failed_candidates':sum(h['failed'] for h in state['history'])}
    atomic(ROOT/'docs/endophalon4/audit.json',{'passed':True,'directions':result,'total_evaluations':1922})
    print(json.dumps(result,indent=2));print('PASS: budgets, transitions, split isolation, models and saved evidence')


if __name__=='__main__':main()
