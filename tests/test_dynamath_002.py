import copy
import math
import torch
import pytest
from dynamath.v002 import Model,Config,Session
from dynamath.v002.__main__ import block,generate

torch.set_num_threads(2)
def model(memories=9,cross=False,frozen=False):
    torch.manual_seed(2)
    return Model(Config('\ue000\ue001\ue002\ue003\ue004ab',memories,cross,frozen,latent=4,rank=3))

def test_clock_queue_and_single_state():
    m=model().eval();s=Session(m,threshold=99);s.enqueue([5,6,5]);initial_s,initial_m=m.zero()
    expected=initial_s;mem=initial_m
    for t in range(20):
        if t in (0,3,6):expected=m.inject(expected,[[5,6,5][t//3]])
        expected,mem,_=m.step(expected,mem,t);s.tick()
        assert torch.equal(s.s,expected)
        assert s.last_poll==(t if t%3 else t-1)
    assert [r['law'] for r in s.log]==[i%9+1 for i in range(20)]
    assert [r['tick'] for r in s.log if r['admitted']]==[0,3,6]
    assert not s.queue
    assert m.inject(initial_s,[[5,5]])[0,5]==2

@pytest.mark.parametrize('count,cross',[(9,False),(3,False),(3,True),(1,False)])
def test_scalar_ownership_and_hard_retention(count,cross):
    m=model(count,cross).eval();s,mem=m.zero()
    assert mem.shape==(1,count)
    assert all(v in (0.,1.,float(torch.tensor(math.e))) for v in mem[0].tolist())
    assert not any('memory' in name for name,_ in m.named_parameters())
    with torch.no_grad():m.gb.fill_(-100)
    for t in range(19):
        s,new,g=m.step(s,mem,t);assert torch.equal(mem,new);mem=new
    with torch.no_grad():m.gb.fill_(100);m.wb.fill_(.4)
    for t in range(9):
        old=mem.clone();s,mem,g=m.step(s,mem,t)
        others=[j for j in range(count) if j!=m.owner(t)]
        assert torch.equal(old[:,others],mem[:,others])
        assert mem.abs().max()<=math.e
    s0,mem0=m.zero();altered=mem0.clone();altered[:,m.owner(0)]=2
    assert not torch.equal(m.step(s0,mem0,0)[0],m.step(s0,altered,0)[0])

def test_cross_access_mask():
    for cross in (False,True):
        m=model(3,cross).eval();s,mem=m.zero();changed=mem.clone();changed[:,2]+=1
        same=torch.equal(m.step(s,mem,0)[0],m.step(s,changed,0)[0])
        assert same != cross

def test_poll_real_double_event_fifo_and_snapshot():
    m=model().eval();session=Session(m)
    # Fixed positive output scores persist; no mode checks; two actual events.
    with torch.no_grad():
        for p in m.parameters():p.zero_()
        m.c[:,len(m.config.vocabulary)+5]=5
    session.tick();assert session.events==[]
    session.tick();assert len(session.events)==1
    before=session.snapshot();assert session.poll() is None;assert session.snapshot()==before
    session.tick();assert len(session.events)==2;assert len(session.queue)==2
    saved=session.snapshot();restored=Session(m);restored.restore(saved)
    for _ in range(12):
        assert session.tick()==restored.tick()
        assert session.snapshot()==restored.snapshot()
    assert any(r['admitted'] and r['admitted'][1]=='feedback' for r in session.log)

def test_tie_silence_threshold_eos():
    session=Session(model().eval());w=len(session.model.config.vocabulary)
    session.t=2;session.s.zero_();session.s[0,w+2]=.5
    assert session.poll()==2
    session.t=3;session.s.zero_();session.s[0,w+3]=1
    assert session.poll() is None
    session.t=5;session.s.zero_();session.s[0,w]=.5;session.s[0,w+1]=.5
    assert session.poll()==0
    session.t=6;session.s.zero_();assert session.poll() is None

def test_exact_training_restart(tmp_path):
    torch.manual_seed(9);m=model();opt=torch.optim.Adam(m.parameters(),lr=.002)
    s,mem=m.zero();t=0
    def update(m,opt,s,mem,t):
        token=torch.randint(5,7,(1,));s,mem,out,_=block(m,s,mem,t,token)
        loss=sum(torch.nn.functional.cross_entropy(o,token) for o in out)
        opt.zero_grad();loss.backward();opt.step()
        return s.detach(),mem.detach(),t+3
    s,mem,t=update(m,opt,s,mem,t)
    path=tmp_path/'restart.pt';m.save(path,optimizer=opt.state_dict(),rng=torch.get_rng_state(),s=s,m=mem,t=t,data='fixture-v1')
    a=update(m,opt,s,mem,t)
    restored,p=Model.load(path);restored.train();opt2=torch.optim.Adam(restored.parameters(),lr=.002);opt2.load_state_dict(p['optimizer'])
    torch.set_rng_state(p['rng']);b=update(restored,opt2,p['s'],p['m'],p['t'])
    assert p['data']=='fixture-v1'
    assert all(torch.equal(x,y) for x,y in zip(a[:2],b[:2]))
    assert all(torch.equal(x,y) for x,y in zip(m.parameters(),restored.parameters()))

def test_no_dataset_generation(tmp_path,monkeypatch):
    m=model().eval();monkeypatch.chdir(tmp_path)
    result=generate(m,'ab',30)
    assert result['ticks']<=30

def test_public_training_resume_equivalence():
    from dynamath.v002.__main__ import train_language
    cfg=dict(lr=.002,steps=4,tbptt_slots=3,seconds_cap_per_run=60)
    x=torch.tensor([[5],[6],[5],[6]]);y=torch.tensor([[6],[5],[6],[5]])
    whole=model();part=model()
    _,a=train_language(whole,x,y,cfg)
    _,r=train_language(part,x,y,{**cfg,'steps':2})
    _,b=train_language(part,x,y,cfg,r)
    assert all(torch.equal(x,y) for x,y in zip(whole.parameters(),part.parameters()))
    assert torch.equal(a['s'],b['s']) and torch.equal(a['m'],b['m']) and a['t']==b['t']
    with pytest.raises(ValueError,match='identity'):
        train_language(part,x+1,y,cfg,r)

def test_completed_primary_restart_is_noop():
    from dynamath.v002.__main__ import train_language
    cfg=dict(lr=.002,steps=2,tbptt_slots=3,seconds_cap_per_run=60,seed=43)
    x=torch.tensor([[5],[6]]);y=torch.tensor([[6],[5]])
    m=model();_,r=train_language(m,x,y,cfg)
    r.pop('python_rng');r.pop('completed_steps')
    before=copy.deepcopy(m.state_dict());info,new=train_language(m,x,y,cfg,r)
    assert info['steps']==2 and new['t']==r['t']
    assert all(torch.equal(value,m.state_dict()[key]) for key,value in before.items())
