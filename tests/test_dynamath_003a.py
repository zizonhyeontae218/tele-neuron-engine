import copy,json,torch,pytest
from dynamath.v002 import Config,Session
from dynamath.v003a import Model,BASE
from dynamath.v003a.ast import validate,canonical,dumps,interpret
from dynamath.v003a.formula_forge import catalog,manifest,binary,leaf,unary
from dynamath.v003a.mutation import mutate
from dynamath.v003a.english_data import pairs
from dynamath.v003a.training import streams,train
from dynamath.korean_data import Tokenizer,CONTROLS

def model(): return Model(Config(CONTROLS+'abc ',memories=1,latent=8,rank=3))

def test_roundtrip_and_commutativity():
    a=binary('add',leaf('z'),leaf('S')); b=binary('add',leaf('S'),leaf('z'))
    assert dumps(canonical(a))==dumps(canonical(b))
    assert json.loads(dumps(a))==a
    rows=catalog(200);assert manifest(rows,43)['unique']==200

def test_invalid_and_bounded():
    with pytest.raises(ValueError): validate({'op':'exec','arg':leaf('z')})
    with pytest.raises(ValueError): validate(binary('sub',leaf('z'),leaf('z')))
    z=torch.tensor([[-1e10,0.,1e10]]);m=torch.tensor([[2.]])
    for row in catalog(300):
        out=interpret(row['ast'],z,z,m,torch.tensor(1.));assert out.shape==z.shape and torch.isfinite(out).all() and out.abs().max()<=1

def test_structural_change_and_identity():
    ast,kind=mutate(BASE,'add_state');assert kind=='identity' and ast!=BASE
    z=torch.randn(2,5);m=torch.randn(2,1)
    assert torch.equal(interpret(BASE,z,z,m,torch.tensor(0.)),interpret(ast,z,z,m,torch.tensor(0.)))

def test_cycle_injection_checks_persistence_and_checkpoint(tmp_path):
    m=model().eval();s=Session(m);s.enqueue([5,6]);before=s.s.clone()
    for i in range(12):
        s.tick();assert s.log[-1]['law']==i%9+1
        assert (s.log[-1]['admitted'] is not None)==(i in (0,3)) or s.log[-1]['admitted'] is not None and s.log[-1]['admitted'][1]=='feedback'
        if i%3==0: assert s.log[-1]['event'] is None
    assert not torch.equal(before,s.s)
    snap=s.snapshot();q=Session(m);q.restore(snap)
    for _ in range(5): assert s.tick()==q.tick()
    assert torch.equal(s.s,q.s)
    p=tmp_path/'model.pt';m.save(p);loaded,_=Model.load(p);assert torch.equal(m.v,loaded.v)
    assert Session(loaded).tick() is None

def test_session_reset_stream_and_equal_budget():
    ds=[{'turns':['a','b','b','a']} for _ in range(8)]
    assert len(pairs(ds[0]))==2
    data=streams(ds,Tokenizer(CONTROLS+'ab'),2)
    assert data[2][0].all() and data[2].sum()==8
    cfg={'lr':.0002,'steps':2,'seconds_cap_per_run':100,'tbptt_slots':2}
    base=model();a=copy.deepcopy(base);b=copy.deepcopy(base);b.asts[0]=binary('residual',leaf('z'),leaf('S'))
    x=streams(ds,Tokenizer(CONTROLS+'abc '),2)
    sa,ra=train(a,x,cfg);sb,rb=train(b,x,cfg)
    assert sa['steps']==sb['steps']==2 and sa['lane_transitions']==sb['lane_transitions']
    cfg['steps']=3;sc,rc=train(a,x,cfg,ra);assert sc['steps']==3

def test_racing_resumes_and_rejects_unequal_budgets():
    from dynamath.v003a.search import race
    cfg={'lr':.0002,'steps':1,'seconds_cap_per_run':100,'tbptt_slots':1}
    ds=[{'turns':['a','b']} for _ in range(4)]
    data=streams(ds,Tokenizer(CONTROLS+'abc '),2)
    def fit(m,n,r): return train(m,data,dict(cfg,steps=n),r)
    winners,hist=race(model(),[(0,binary('residual',leaf('z'),leaf('S')))],[(2,1),(1,2)],fit,lambda m:{'value':float(m.alpha[0].detach())},lambda v,n: n)
    assert len(winners)==1 and [h['updates'] for h in hist]==[1,2]
    with pytest.raises(RuntimeError):
        race(model(),[],[(1,1)],lambda m,n,r:({'steps':0},None),lambda m:{},lambda v,n:0)

def test_public_split_dialogue_leakage_is_removed():
    from dynamath.v003a.english_data import remove_cross_split_dialogues
    def d(i,turns):return {'id':i,'turns':turns}
    raw={'train':[d('a',['x','y']),d('b',['a','b'])], 'validation':[d('c',['a','b'])], 'test':[d('d',['x','y'])]}
    clean,removed=remove_cross_split_dialogues(raw)
    assert not clean['train'] and len(removed['train'])==2 and clean['test'][0]['id']=='d'

def test_unedited_law_is_exact_002_model1():
    from dynamath.v002 import Model as OldModel
    cfg=Config(CONTROLS+'abc ',memories=1,latent=8,rank=3)
    old=OldModel(cfg).eval();new=Model(cfg).eval();new.load_state_dict(old.state_dict(),strict=False)
    s,m=old.zero(2);s=torch.randn_like(s);m=torch.randn_like(m)
    for t in range(9):
        a=old.step(s,m,t);b=new.step(s,m,t)
        assert all(torch.equal(x,y) for x,y in zip(a,b))
