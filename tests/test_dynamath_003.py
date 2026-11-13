import copy, json
import pytest
import torch
from dynamath.korean_data import CONTROLS
from dynamath.v002 import Config
from dynamath.v003 import Model, Session, BASE, mutation, validate, interpret


def model():
    torch.manual_seed(13)
    return Model(Config(CONTROLS+'abc',memories=1,latent=4,rank=3)).eval()

def test_ast_roundtrip_identity_and_causal_effect():
    ast=json.loads(json.dumps(mutation()));validate(ast)
    x=torch.tensor([-0.8,0.2,0.9])
    assert torch.equal(interpret(ast,x,torch.tensor(0.)),interpret(BASE,x,torch.tensor(0.)))
    assert not torch.equal(interpret(ast,x,torch.tensor(.3)),interpret(BASE,x,torch.tensor(.3)))
    assert interpret(ast,x,torch.tensor(100.)).abs().max() <= 1
    with pytest.raises(ValueError): validate({'op':'eval','arg':{'op':'input'}})

def test_inherited_parent_and_edited_law():
    m=model(); s,mem=m.zero();s=m.inject(s,[5]);edited=copy.deepcopy(m);edited.asts[0]=mutation()
    assert torch.equal(m.step(s,mem,0)[0],edited.step(s,mem,0)[0])
    with torch.no_grad(): edited.alpha[0]=.7
    assert not torch.equal(m.step(s,mem,0)[0],edited.step(s,mem,0)[0])
    assert torch.equal(m.step(s,mem,1)[0],edited.step(s,mem,1)[0])

def test_session_clock_queue_and_checkpoint(tmp_path):
    m=model(); session=Session(m,threshold=-1);session.enqueue([5,6])
    for _ in range(12):session.tick()
    assert [r['law'] for r in session.log]==[1,2,3,4,5,6,7,8,9,1,2,3]
    assert all(r['admitted'] is None for r in session.log if r['tick']%3)
    assert all(t%3 in (1,2) for t,_ in session.events)
    snap=session.snapshot();before=session.s.clone();session.poll();assert torch.equal(before,session.s)
    path=tmp_path/'checkpoint.pt';m.save(path,snapshot=snap)
    restored,p=Model.load(path);other=Session(restored);other.restore(p['snapshot'])
    for _ in range(12):session.tick();other.tick()
    assert session.snapshot()==other.snapshot()

def test_scalar_persistence_and_finite():
    m=model();s,mem=m.zero();mem.fill_(1.2)
    # Explicit scalar read feeds state and runtime scalar remains bounded.
    altered=mem.clone().zero_()
    assert not torch.equal(m.step(s,mem,0)[0],m.step(s,altered,0)[0])
    for t in range(90):s,mem,_=m.step(s,mem,t)
    assert torch.isfinite(s).all() and torch.isfinite(mem).all()
    assert mem.abs().max()<=torch.e
