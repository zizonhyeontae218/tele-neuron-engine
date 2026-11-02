"""Training must preserve bank structure, train-only fits and deterministic resume."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
from tele_neuron.endophalon3_alpha.model import Model
from tele_neuron.endophalon3_alpha import train as t

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def base():
    return Model.load(ROOT/'models/endophalon3_alpha/20261102_seed1_binary.json')


def test_checkpoint_roundtrip_preserves_next_cem_candidates(base):
    def evaluate(vectors):return [{'j':float(np.mean(v*v)),'finite':True} for v in vectors]
    state=t.generation(t.new_state(base),evaluate,population=4,elite=2)
    state=t.generation(state,evaluate,population=4,elite=2)
    saved=json.loads(json.dumps(state))
    assert t.generation(state,evaluate,4,2)==t.generation(saved,evaluate,4,2)
    assert state['evaluations']==5
    assert len(state['mean'])==154 and len(state['sigma'])==154


def test_validation_targets_never_enter_head_fit(base):
    rng=np.random.default_rng(9);streams=np.arange(4);pulses=64
    levels=rng.integers(0,2,(4,pulses));ops=np.where(levels==0,rng.integers(0,4,(4,pulses)),rng.integers(0,8,(4,pulses)))
    data={'features':rng.normal(size=(4,pulses,16)),'targets':rng.integers(0,2,(4,pulses)),'levels':levels,'ops':ops,'age':np.ones((4,pulses),int),'energy':rng.uniform(.01,.02,(4,pulses)),'streams':streams}
    changed=deepcopy(data);changed['targets'][2:]=1-changed['targets'][2:]
    a,fa=t.objective(base,data,range(0,2),range(2,4));b,fb=t.objective(base,changed,range(0,2),range(2,4))
    for level in 'AB':
        for field in ('mean','scale','weights'):np.testing.assert_array_equal(getattr(a.banks[level].head,field),getattr(b.banks[level].head,field))
    assert fa['j']!=fb['j']


def test_pair_decode_keeps_separate_banks_and_geometry(base):
    u=t.encode(base);u[:77]+=.01;u[77:]-=.01
    model=t.decode(base,u)
    assert model.configs['A'].level=='A' and model.configs['B'].level=='B'
    assert model.banks['A'].head is None and model.banks['B'].head is None
    assert model.config.network.nodes==144 and model.config.physics.substeps==2
    assert model.configs['A'].theta_init!=model.configs['B'].theta_init
    with pytest.raises(ValueError):t.decode(base,u[:-1])


def test_divergent_candidate_is_counted_as_failure(base,monkeypatch):
    def fail(*args,**kwargs):raise FloatingPointError('test divergence')
    monkeypatch.setattr(t,'observe',fail)
    result=t.candidate((base.to_payload(),t.encode(base)))
    assert not result['finite'] and result['j']==10. and 'test divergence' in result['failure']


def test_training_observations_match_production_session(base):
    from tele_neuron.endophalon3_alpha.session import Session
    from tele_neuron.endophalon3.isa import Instruction,OPS,random_stream
    ids=(0,1);data=t.observe(base,ids,pulses=8);session=Session(base,ids)
    programs=[random_stream('AB',1,s,8) for s in ids]
    for k in range(8):
        result=session.step([Instruction('AB'[p.levels[k]],OPS[p.ops[k]],int(p.a[k]),int(p.b[k])) for p in programs])
        np.testing.assert_array_equal(data['features'][:,k],np.array([r.features for r in result]))
        np.testing.assert_array_equal(data['energy'][:,k],np.array([r.kinetic_before for r in result]))


def test_gate_requires_every_supported_operation_and_both_levels():
    result={'levels':{l:{'accuracy':.95,'per_op':{op:{'accuracy':.9,'n':10} for op in t.LEVELS[l]}} for l in 'AB'}}
    assert t.passed(result)
    result['levels']['B']['per_op']['recall2']['accuracy']=.84
    assert not t.passed(result)


def test_gate_rejects_missing_operation():
    result={'levels':{l:{'accuracy':1.,'per_op':{op:{'accuracy':1.,'n':10} for op in t.LEVELS[l]}} for l in 'AB'}}
    del result['levels']['B']['per_op']['parity3w']
    assert not t.passed(result)
