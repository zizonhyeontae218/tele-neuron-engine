"""Session-reset-aware lane streams and restartable training."""
import hashlib,json,random,time,math
import torch
from torch.nn import functional as F
from dynamath.v002.__main__ import block
from .english_data import pairs

def streams(dialogues,tok,batch):
    lanes=[]
    for lane in range(batch):
        xs=[];ys=[];rs=[]
        for d in dialogues[lane::batch]:
            first=True
            for p in pairs(d):
                x,y=tok.pairs(p);xs+=x;ys+=y;rs += [first]+[False]*(len(x)-1);first=False
        lanes.append((xs,ys,rs))
    n=min(len(a) for a,b,c in lanes)
    if not n: raise ValueError('empty lane')
    return tuple(torch.tensor([lane[j][:n] for lane in lanes]).T for j in range(3))

def train(model,data,cfg,resume=None):
    x,y,reset=data; digest=hashlib.sha256(b''.join(t.numpy().tobytes() for t in data)).hexdigest()
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr']);s,m=model.zero(x.shape[1]);t=pos=step=0
    if resume:
        if resume['data_sha256']!=digest: raise ValueError('training data mismatch')
        opt.load_state_dict(resume['optimizer']);torch.set_rng_state(resume['rng']);random.setstate(resume['python_rng'])
        s,m,t,pos,step=(resume[k] for k in ('s','m','t','pos','completed_steps'))
    start=time.monotonic();hist=[]; model.train()
    while step<cfg['steps'] and time.monotonic()-start<cfg['seconds_cap_per_run']:
        opt.zero_grad();loss=0
        for _ in range(cfg['tbptt_slots']):
            mask=reset[pos][:,None];zs,zm=model.zero(len(s));s=torch.where(mask,zs,s);m=torch.where(mask,zm,m)
            s,m,out,_=block(model,s,m,t,x[pos]);t+=3
            loss+=(F.cross_entropy(out[0],y[pos])+F.cross_entropy(out[1],torch.full_like(y[pos],3)))/2
            pos=(pos+1)%len(x)
        loss/=cfg['tbptt_slots'];loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();s=s.detach();m=m.detach();step+=1
        if step==1 or step%100==0:hist.append({'step':step,'loss':float(loss.detach())})
    model.eval(); elapsed=time.monotonic()-start
    restart={'optimizer':opt.state_dict(),'rng':torch.get_rng_state(),'python_rng':random.getstate(),'s':s,'m':m,'t':t,'pos':pos,'completed_steps':step,'data_sha256':digest}
    return {'steps':step,'ticks_per_lane':t,'lane_transitions':t*x.shape[1],'seconds':elapsed,'history':hist},restart

@torch.no_grad()
def evaluate(model,data):
    x,y,reset=data;s,m=model.zero(x.shape[1]);ce=n=correct=omitted=false=silent=0
    for pos in range(len(x)):
        mask=reset[pos][:,None];zs,zm=model.zero(len(s));s=torch.where(mask,zs,s);m=torch.where(mask,zm,m)
        s,m,out,_=block(model,s,m,pos*3,x[pos])
        for p in (0,1):
            target=y[pos] if p==0 else torch.full_like(y[pos],3);active=target!=3
            logits=out[p];pred=logits.argmax(1);emit=(logits.max(1).values/model.config.gain>=.1)&(pred!=3)
            ce+=float(F.cross_entropy(logits,target,reduction='none')[active].sum());n+=int(active.sum());correct+=int(((pred==target)&active).sum());omitted+=int((active&~emit).sum());silent+=int((~active).sum());false+=int(((~active)&emit).sum())
    return {'response_CE':ce/n,'response_perplexity':math.exp(ce/n),'response_accuracy':correct/n,'response_omission_rate':omitted/n,'silence_false_emission_rate':false/silent,'response_tokens':n}
