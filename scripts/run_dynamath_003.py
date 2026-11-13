"""Run the frozen 003 CPU feasibility experiment; inference needs no corpus."""
import copy, json, time, platform, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import torch
from dynamath.korean_data import load, fetch
from dynamath.v002.__main__ import streams, train_language, evaluate, generate
from dynamath.v003 import Model, BASE, mutation, expression

ROOT=Path('docs/dynamath/experiment-003'); MODELS=Path('models/dynamath/003')
def dump(path, obj): Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def main():
    cfg=json.loads(Path('configs/dynamath/experiment-003.json').read_text())
    torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed'])
    data=Path('data/dynamath/001a/ChatbotData.csv')
    if not data.exists(): data.write_bytes(fetch('ChatbotData.csv'))
    splits,tok=load(data.parent)
    train=splits['train'][:cfg['train_pairs']];val=splits['validation'][:cfg['validation_pairs']];test=splits['test'][:cfg['test_pairs']]
    dump(ROOT/'evaluation-plan.json',{'config':cfg,'train_ids':[r['id'] for r in train], 'validation_ids':[r['id'] for r in val], 'test_ids':[r['id'] for r in test], 'generation_ids':[r['id'] for r in test[:cfg['generation_pairs']]]})
    x,y=streams(train,tok,cfg['batch']); origin=Model.from_002('models/dynamath/002a/model1-step18000.pt')
    results={}; models={}; restarts={}
    for law in [None]+cfg['candidate_laws']:
        name='control' if law is None else f'law{law+1}'
        model=copy.deepcopy(origin)
        if law is not None: model.asts[law]=mutation()
        start=time.monotonic();training,restart=train_language(model,x,y,cfg)
        score=evaluate(model,val,tok)
        model.save(MODELS/(name+'-search.pt'),restart=restart,training=training)
        results[name]={'law':law,'validation':score,'training':training,'elapsed_seconds':time.monotonic()-start,'alpha':model.alpha.detach().tolist()}
        models[name]=model;restarts[name]=restart
        print(name,score['response_CE'],flush=True)
    best=min(results,key=lambda n:results[n]['validation']['response_CE'])
    # Continue selected arm and a matched no-edit parent, including the no-edit winner.
    final={}; continuation=dict(cfg,steps=cfg['steps']+cfg['confirm_steps'],seconds_cap_per_run=120)
    for name in dict.fromkeys(['control',best]):
        model=models[name];training,restart=train_language(model,x,y,continuation,restarts[name])
        model.save(MODELS/(name+'-final.pt'),restart=restart,training=training)
        final[name]={'training':training,'test':evaluate(model,test,tok),
                     'generations':[generate(model,r['Q'],cfg['generation_ticks']) for r in test[:cfg['generation_pairs']]]}
    baseline={}
    for label,path in [('002a','models/dynamath/002a/model1-step18000.pt'),('002b','models/dynamath/002b/model1-step72000.pt')]:
        from dynamath.v002 import Model as OldModel
        model,_=OldModel.load(path)
        baseline[label]={'test':evaluate(model,test,tok),'generations':[generate(model,r['Q'],cfg['generation_ticks']) for r in test[:cfg['generation_pairs']]]}
    dump(ROOT/'results-003.json',{'environment':{'torch':torch.__version__,'platform':platform.platform()},'selection':best,'candidates':results,'final':final,'baselines':baseline,
        'before_ast':BASE,'after_ast':mutation(),'before_expression':expression(BASE),'after_expression':expression(mutation()),'parameters':sum(p.numel() for p in origin.parameters())})
if __name__=='__main__': main()
