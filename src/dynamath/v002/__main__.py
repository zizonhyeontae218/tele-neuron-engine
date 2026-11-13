"""Reproducible bounded 002 experiments and dataset-independent inference."""
import argparse, json, time, random, math, platform, resource
from pathlib import Path
import torch
from torch.nn import functional as F
from . import Model, Config, Session
from ..korean_data import Tokenizer, load, fetch, sha

ROOT=Path('docs/dynamath/experiment-002')
MODELS=Path('models/dynamath/002')
def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def variant(name,vocab,cfg):
    return Config(vocab,memories=3 if 'model3' in name else 1 if name=='model1' else 9,
                  cross='cross' in name,frozen='frozen' in name,rank=cfg['rank'],latent=cfg['latent'])
def streams(rows,tok,batch):
    lanes=[]
    for lane in range(batch):
        x=[];y=[]
        for row in rows[lane::batch]:
            a,b=tok.pairs(row);x+=a;y+=b
        lanes.append((x,y))
    n=min(len(x) for x,y in lanes)
    return torch.tensor([x[:n] for x,y in lanes]).T,torch.tensor([y[:n] for x,y in lanes]).T

def block(model,s,m,t,x):
    s=model.inject(s,x); outputs=[]; writes=[]
    for p in range(3):
        s,m,g=model.step(s,m,t+p);writes.append(g)
        if p: outputs.append(model.logits(s))
    return s,m,outputs,writes

def train_language(model,x,y,cfg,resume=None):
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr']);s,m=model.zero(x.shape[1]);t=0;pos=0
    data_hash=sha(json.dumps([x.tolist(),y.tolist()],separators=(',',':')).encode())
    first_step=0
    if resume is not None:
        if resume['data_sha256']!=data_hash: raise ValueError('restart data identity mismatch')
        opt.load_state_dict(resume['optimizer']);torch.set_rng_state(resume['rng'])
        random.setstate(resume.get('python_rng',random.Random(cfg.get('seed',43)).getstate()))
        s,m,t,pos=resume['s'],resume['m'],resume['t'],resume['pos']
        first_step=resume.get('completed_steps',t//(3*cfg['tbptt_slots']))
    start=time.monotonic();history=[];model.train();step=first_step-1
    for step in range(first_step,cfg['steps']):
        opt.zero_grad();loss=0
        for j in range(cfg['tbptt_slots']):
            s,m,out,g=block(model,s,m,t,x[pos]);t+=3
            # Equal non-silence/SILENCE slot weights; no target-dependent runtime mask.
            loss=loss+(F.cross_entropy(out[0],y[pos])+F.cross_entropy(out[1],torch.full_like(y[pos],3)))/2
            pos=(pos+1)%len(x)
        loss=loss/cfg['tbptt_slots'];loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step();s=s.detach();m=m.detach()
        if step%50==0: history.append(dict(step=step,loss=float(loss.detach()),seconds=time.monotonic()-start,ticks=t))
        if time.monotonic()-start>=cfg['seconds_cap_per_run']: break
    model.eval()
    return dict(steps=step+1,seconds=time.monotonic()-start,ticks=t,slots=t//3,
                stream_slots=len(x),stream_passes=(t//3)/len(x),history=history),dict(
                optimizer=opt.state_dict(),rng=torch.get_rng_state(),python_rng=random.getstate(),
                completed_steps=step+1,s=s,m=m,t=t,pos=pos,data_sha256=data_hash)

@torch.no_grad()
def evaluate(model,rows,tok):
    x,y=streams(rows,tok,8);s,m=model.zero(8);ce=0.;n=0;correct=0;false=0;questions=0;omitted=0;writes=[0]*9
    for pos in range(len(x)):
        s,m,out,gs=block(model,s,m,pos*3,x[pos])
        for p in (0,1):
            target=y[pos] if p==0 else torch.full_like(y[pos],3)
            active=target!=3
            ce+=float(F.cross_entropy(out[p],target,reduction='none')[active].sum());n+=int(active.sum())
            pred=out[p].argmax(1);emit=(out[p].max(1).values/model.config.gain>=.1)&(pred!=3)
            correct+=int(((pred==target)&active).sum());omitted+=int((active&~emit).sum())
            questions+=int((target==3).sum());false+=int(((target==3)&emit).sum())
        for p,g in enumerate(gs): writes[(pos*3+p)%9]+=int(g.sum())
    return dict(response_CE=ce/n,response_perplexity=math.exp(ce/n),response_accuracy=correct/n,
                response_omission_rate=omitted/n,silence_false_emission_rate=false/questions,
                response_tokens=n,law_gate_counts=writes,final_m=m.tolist())

def generate(model,question,max_ticks=360):
    tok=Tokenizer(model.config.vocabulary);session=Session(model);answer=[];qevents=[];eos=False
    # Enqueue question as one external burst. Feedback joins its tail; BOT is already
    # in FIFO, so no external priority or suppression of question-time emissions.
    session.enqueue([0]+tok.encode(question)+[1])
    responding=False
    for _ in range(max_ticks):
        event=session.tick();row=session.log[-1]
        if row['admitted']==(1,'external'): responding=True
        if event is not None:
            if responding:
                answer.append(event)
                if event==2: eos=True;break
            else:qevents.append(event)
    text=tok.decode(answer)
    return dict(question=question,text=text,token_ids=answer,question_events=qevents,eos=eos,
                response_omission=not answer,unexpected_controls=sum(x in (0,1,4) for x in answer),
                uncomposed_jamo=sum('\u1100'<=c<='\u11ff' for c in text),replacement_characters=text.count('�'),
                ticks=session.t,peak_queue=max(r['queue_depth'] for r in session.log),
                final_queue=len(session.queue),write_counts=[sum(r['changed'] for r in session.log if r['law']==i+1) for i in range(9)],
                snapshot=session.snapshot())

# Separate learned binary recall: fixed input addresses 0/1 encode bit, address 2
# queries; fixed output addresses 0/1 answer. No target-dependent scalar assignment.
def recall_trial(model,bits,delay,reset='none'):
    s,m=model.zero(len(bits));s=model.inject(s,bits);s,m,g=model.step(s,m,0)
    if reset in ('state','both'):s=torch.zeros_like(s)
    if reset in ('memory','both'):m=model.zero(len(bits))[1]
    t=1
    for _ in range(delay):s,m,g=model.step(s,m,t);t+=1
    # Wait for lawful query admission slot; exact requested idle count is logged.
    while t%3:s,m,g=model.step(s,m,t);t+=1
    s=model.inject(s,torch.full_like(bits,2))
    for p in range(3):s,m,g=model.step(s,m,t+p)
    return model.logits(s)[:,:2],m,t+3

def recall(name,cfg):
    torch.manual_seed(cfg['seed']+100)
    model=Model(variant(name,'01234',cfg));opt=torch.optim.Adam(model.parameters(),lr=.003)
    start=time.monotonic();history=[]
    for step in range(cfg['recall_steps']):
        model.train();bits=torch.arange(12)%2;delay=cfg['recall_delays'][step%3]
        logits,m,t=recall_trial(model,bits,delay);loss=F.cross_entropy(logits,bits)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%100==0:history.append(dict(step=step,loss=float(loss.detach())))
    model.eval();results={};gen=torch.Generator().manual_seed(cfg['recall_test_seed']);bits=torch.randint(2,(128,),generator=gen)
    with torch.no_grad():
        for delay in cfg['recall_delays']:
            results[str(delay)]={}
            for arm in ('none','state','memory','both'):
                logits,m,t=recall_trial(model,bits,delay,arm)
                results[str(delay)][arm]=dict(accuracy=float((logits.argmax(1)==bits).float().mean()),
                    m_bit0=m[bits==0].mean(0).tolist(),m_bit1=m[bits==1].mean(0).tolist(),ticks=t)
    model.save(MODELS/(name+'-recall.pt'))
    return dict(seconds=time.monotonic()-start,history=history,results=results,
                limitation='held-out seeded bit sequences, only two bit classes; no natural-language memory claim')

def run(resume_path=None):
    cfg=json.loads(Path('configs/dynamath/experiment-002.json').read_text());torch.set_num_threads(cfg['threads'])
    data=Path('data/dynamath/001a/ChatbotData.csv')
    if not data.exists():data.write_bytes(fetch('ChatbotData.csv'))
    splits,tok=load(data.parent);MODELS.mkdir(parents=True,exist_ok=True)
    frozen=dict(config=cfg,test_ids=[r['id'] for r in splits['test'][:cfg['eval_pairs']]],
                generation_ids=[r['id'] for r in splits['test'][:cfg['generation_pairs']]],
                source_sha256=sha(data.read_bytes()),hardware=platform.platform(),torch=torch.__version__)
    dump(ROOT/'evaluation-plan.json',frozen)
    x,y=streams(splits['train'],tok,cfg['batch']);all_results={}
    for name in cfg['variants']:
        torch.manual_seed(cfg['seed']);random.seed(cfg['seed']);model=Model(variant(name,tok.vocabulary,cfg))
        resume=None
        if resume_path:
            saved,p=Model.load(resume_path)
            if saved.config==model.config: model=saved;resume=p['restart']
            else: continue
        training,restart=train_language(model,x,y,cfg,resume)
        model.save(MODELS/(name+'.pt'),training=training,restart=restart)
        result=dict(parameters=sum(p.numel() for p in model.parameters()),training=training,
                    teacher_forced=evaluate(model,splits['test'][:cfg['eval_pairs']],tok),
                    generations=[generate(model,r['Q'],cfg['generation_ticks']) for r in splits['test'][:cfg['generation_pairs']]])
        result['recall']=recall(name,cfg);result['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        dump(ROOT/(name+'.json'),result);all_results[name]=result
        print(name,training['steps'],result['teacher_forced']['response_CE'],flush=True)
    dump(ROOT/'results.json',all_results)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['train','evaluate','generate','chat','tick'])
    ap.add_argument('--checkpoint',default='models/dynamath/002/model9.pt');ap.add_argument('--question',default='안녕하세요')
    ap.add_argument('--ticks',type=int,default=360);ap.add_argument('--snapshot');ap.add_argument('--restore');ap.add_argument('--resume')
    args=ap.parse_args();torch.set_num_threads(2)
    if args.command=='train':run(args.resume);return
    model,p=Model.load(args.checkpoint)
    if args.command=='evaluate':
        splits,tok=load('data/dynamath/001a');print(json.dumps(evaluate(model,splits['test'][:96],tok),ensure_ascii=False));return
    if args.command=='generate':print(json.dumps(generate(model,args.question,args.ticks),ensure_ascii=False));return
    session=Session(model);tok=Tokenizer(model.config.vocabulary)
    if args.restore:session.restore(json.loads(Path(args.restore).read_text()))
    if args.command=='tick':
        for _ in range(args.ticks):session.tick()
    else:
        while True:
            try:q=input('> ')
            except EOFError:break
            session.enqueue([0]+tok.encode(q)+[1]);out=[]
            for _ in range(args.ticks):
                event=session.tick()
                if event is not None:out.append(event)
                if event==2:break
            print(tok.decode(out))
    if args.snapshot:dump(args.snapshot,session.snapshot())
    else:print(json.dumps(session.snapshot(),ensure_ascii=False))
if __name__=='__main__':main()
