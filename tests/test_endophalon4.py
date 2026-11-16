"""Critical physical boundaries and serialization/resumption invariants."""
from copy import deepcopy
import json
import numpy as np
import pytest
from tele_neuron.endophalon3_software.session import Session
from tele_neuron.endophalon4.runtime import tick,zones,reference,coupling_force,Coupled
from tele_neuron.endophalon4.experiment import models
from tele_neuron.endophalon4.training import initial,advance_record,atomic,load


def test_zero_force_is_exact_original_physics_and_resumes():
    mod=models('A_to_B')[1]
    a=Session(mod,(1072000,));b=Session(mod,(1072000,))
    for t in range(9):
        ch=np.array([[1,0,1,0,1,0]]) if t<4 else None
        a._tick(ch);tick(b,ch,np.zeros_like(b.positions))
    assert np.array_equal(a.positions,b.positions)
    assert np.array_equal(a.velocities,b.velocities)
    # Raw tick snapshots use an explicit physics state rather than event clock schema.
    state={'positions':b.positions.copy(),'velocities':b.velocities.copy(),
           'clock':b.clock,'rng':[deepcopy(r.bit_generator.state) for r in b.rngs]}
    c=Session(mod,(1072000,));c.positions=state['positions'];c.velocities=state['velocities'];c.clock=state['clock']
    for r,s in zip(c.rngs,state['rng']):r.bit_generator.state=json.loads(json.dumps(s))
    for _ in range(7):
        tick(b,None,np.zeros_like(b.positions));tick(c,None,np.zeros_like(c.positions))
    assert np.array_equal(b.positions,c.positions)
    assert np.array_equal(b.velocities,c.velocities)


def test_live_and_cached_sender_match_and_force_bound():
    sender,receiver=models('A_to_B');u=np.linspace(-1,1,21)
    live=Coupled(sender,receiver,(72000,),(1072000,),u)
    source=Session(sender,(72000,));dest=Session(receiver,(1072000,));home=reference(source)
    assert not np.shares_memory(live.sender.positions,live.receiver.positions)
    for t in range(12):
        ch=np.array([[0,0,0,0,t%2,1]])
        before=zones(source).copy();source._tick(ch)
        force=coupling_force(dest,before,home,u)
        assert np.linalg.norm(force,axis=-1).max()<=.08
        tick(dest,ch,force);live.step(ch,ch)
        assert np.array_equal(live.sender.positions,source.positions)
        assert np.array_equal(live.receiver.positions,dest.positions)
        assert live.sender.clock==live.receiver.clock
    # Runtime coupling interface deliberately has no labels/targets/readout argument.
    import inspect
    assert list(inspect.signature(coupling_force).parameters)==['s','observation','home','u','enabled']


def test_optimizer_resume_atomic_checksum_and_rng(tmp_path):
    state=initial('A_to_B')
    def score(u):return [{'j':float(v@v),'finite':True,'validation':{'accuracy':.5}} for v in u]
    g0=advance_record(state,score);g1=advance_record(g0,score);continuous=advance_record(g1,score)
    path=tmp_path/'checkpoint.json';atomic(path,g1,True);resumed=advance_record(load(path),score)
    for k in ['mean','sigma','variance','best_u','best','rng','history','evaluations']:
        assert resumed[k]==continuous[k]
    payload=json.loads(path.read_text());payload['payload']['evaluations']=999
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='checksum'):load(path)
