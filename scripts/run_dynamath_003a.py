"""Bounded paired search. Validation selects; test is read only after selection."""
import argparse,copy,json,time,random
from pathlib import Path
import torch
from dynamath.korean_data import load as load_ko, fetch
from dynamath.v002.__main__ import streams as ko_streams,train_language, evaluate as ko_eval,generate
from dynamath.v003a import Model,BASE
from dynamath.v003a.ast import expression
from dynamath.v003a.formula_forge import catalog,binary,leaf,unary
from dynamath.v003a.mutation import mutate
from dynamath.v003a.english_data import load as load_en,audit,pairs
from dynamath.v003a.training import streams,train,evaluate

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--language',choices=['english','korean'],required=True);ap.add_argument('--steps',type=int,default=20);args=ap.parse_args()
    cfg=json.loads(Path('configs/dynamath/experiment-003a.json').read_text());torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
    cfg.update(steps=args.steps,seconds_cap_per_run=300)
    if args.language=='korean':
        p=Path('data/dynamath/001a/ChatbotData.csv')
        if not p.exists():p.write_bytes(fetch('ChatbotData.csv'))
        splits,tok=load_ko(p.parent); tr=splits['train'][:512];val=splits['validation'][:32];test=splits['test'][:32]
        data=ko_streams(tr,tok,cfg['batch']); origin=Model.from_002('models/dynamath/002a/model1-step18000.pt')
        fit=lambda m:train_language(m,*data,cfg)
        score=lambda m,rows:ko_eval(m,rows,tok)
        prompt=lambda rows:[r['Q'] for r in rows[:3]]
    else:
        splits=load_en();tok,_=audit(splits);tr=splits['train'][:cfg['train_dialogues']];val=splits['validation'][:cfg['validation_dialogues']];test=splits['test'][:cfg['test_dialogues']]
        data=streams(tr,tok,cfg['batch']);origin,_=Model.load('models/dynamath/003a/english/control.pt')
        fit=lambda m:train(m,data,cfg)
        score=lambda m,rows:evaluate(m,streams(rows,tok,cfg['batch']))
        prompt=lambda rows:[pairs(d)[0]['Q'] for d in rows[:3]]
    # Include identity-preserving insertion and two distinct nonidentity catalog structures.
    lib=catalog(); candidates=[None,binary('residual',leaf('z'),unary('sin',leaf('S'))),lib[2]['ast'],lib[8]['ast']]
    results={};models={};restarts={}
    for i,ast in enumerate(candidates):
        model=copy.deepcopy(origin)
        if ast is not None: model.asts[0]=ast
        start=time.monotonic();initial=score(model,val);stats,restart=fit(model); elapsed=time.monotonic()-start
        validation=score(model,val);name='control' if ast is None else f'candidate{i}'
        path=Path(f'models/dynamath/003a/{args.language}/{name}-search.pt');path.parent.mkdir(parents=True,exist_ok=True)
        model.save(path,restart=restart,training=stats,config_run=cfg)
        results[name]={'initialization':'same' if i==0 else 'identity' if i==1 else 'nonidentity','ast':ast,'initial_validation':initial,'validation':validation,'training':stats,'elapsed_with_initial_validation':elapsed,'parameters':sum(p.numel() for p in model.parameters())}
        models[name]=model;restarts[name]=restart; print(args.language,name,validation['response_CE'],flush=True)
    # Mutate the validation leader, retaining it and a budget-matched control.
    leader=min(results,key=lambda n:results[n]['validation']['response_CE'])
    next_cfg=dict(cfg,steps=2*args.steps)
    evolved=copy.deepcopy(models[leader]); evolved.asts[0],kind=mutate(evolved.asts[0],'add_state',seed=cfg['seed'])
    if float(evolved.alpha[0].detach()) != 0.0: kind='nonidentity: inherited alpha is nonzero'
    second={}
    arms={'control':(copy.deepcopy(models['control']),copy.deepcopy(restarts['control'])),
          'leader':(copy.deepcopy(models[leader]),copy.deepcopy(restarts[leader])),
          'offspring':(evolved,copy.deepcopy(restarts[leader]))}
    for name,(m,restart) in arms.items():
        if args.language=='korean': stats,res=train_language(m,*data,next_cfg,restart)
        else: stats,res=train(m,data,next_cfg,restart)
        metrics=score(m,val);second[name]={'validation':metrics,'training':stats,'parent':leader,'initialization':kind if name=='offspring' else 'continued'}
        m.save(Path(f'models/dynamath/003a/{args.language}/{name}-final.pt'),restart=res,training=stats,config_run=next_cfg)
        arms[name]=(m,res);print(args.language,'stage2',name,metrics['response_CE'],flush=True)
    chosen=min(second,key=lambda n:second[n]['validation']['response_CE'])
    final={}
    for name in dict.fromkeys(['control',chosen]):
        m=arms[name][0]; restored=copy.deepcopy(m);restored.asts[0]=BASE
        final[name]={'test':score(m,test),'restored_validation':score(restored,val),'generations':[generate(m,q,cfg['generation_ticks']) for q in prompt(test)],'expressions':[expression(a) for a in m.asts]}
    output={'language':args.language,'config':cfg,'selection_rule':'pilot validation CE only; not final multidimensional selection','stage1_winner':leader,'winner':chosen,'candidates':results,'stage2':second,'final':final,'limitation':'pilot; no statistical claim; no full multi-metric search'}
    Path(f'docs/dynamath/experiment-003a/{args.language}-search.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
