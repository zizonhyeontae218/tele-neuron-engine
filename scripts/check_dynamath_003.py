"""Predeclared supplemental checks: validation ablation and synthetic recurrence/recall."""
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from torch.nn import functional as F
from dynamath.korean_data import load
from dynamath.v002 import Config
from dynamath.v002.__main__ import evaluate, recall_trial
from dynamath.v003 import Model,BASE,mutation,interpret

def main():
    torch.set_num_threads(2);torch.manual_seed(71);out={}
    model,_=Model.load('models/dynamath/003/law1-final.pt');parent=copy.deepcopy(model);parent.asts[0]=copy.deepcopy(BASE)
    splits,tok=load('data/dynamath/001a');val=splits['validation'][:32]
    out['ablation']={'selected':evaluate(model,val,tok),'restore_parent_ast':evaluate(parent,val,tok),'selected_alpha':float(model.alpha[0].detach())}
    s,m=model.zero();s=model.inject(s,[5]);a=model.step(s,m,0)[0];b=parent.step(s,m,0)[0]
    out['ablation']['one_tick_max_abs_difference']=float((a-b).abs().max().detach())
    # Known bounded recurrence: x_(t+1)=clip(tanh(x_t)+0.3*sin(x_t)).
    # Both arms fit a coefficient; parent coefficient is deliberately unused, as in language control.
    x=torch.linspace(-1,1,101);target=(x.tanh()+.3*x.sin()).clamp(-1,1)
    cases={}
    for name,ast in [('parent',BASE),('mutation',mutation())]:
        alpha=torch.nn.Parameter(torch.tensor(0.));opt=torch.optim.Adam([alpha],lr=.05)
        for _ in range(100):
            loss=((interpret(ast,x[::2],alpha)-target[::2])**2).mean()+alpha*0
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            mse=((interpret(ast,x[1::2],alpha)-target[1::2])**2).mean()
            state=torch.tensor([.4]);truth=state.clone()
            for _ in range(9):
                state=interpret(ast,state,alpha);truth=(truth.tanh()+.3*truth.sin()).clamp(-1,1)
        cases[name]={'alpha':float(alpha.detach()),'heldout_one_step_mse':float(mse),'nine_step_abs_error':float((state-truth).abs())}
    out['synthetic_recurrence']={'target':'clip(tanh(x)+0.3*sin(x))','training_points':51,'heldout_interleaved_points':50,'updates_per_arm':100,'results':cases}
    # Independent tiny fixed bit-recall task, same initial weights and updates.
    torch.manual_seed(72);base=Model(Config('01234',memories=1,latent=8,rank=4));recall={}
    for name,edit in [('control',False),('law1',True)]:
        net=copy.deepcopy(base)
        if edit:net.asts[0]=mutation()
        opt=torch.optim.Adam(net.parameters(),lr=.003);start=time.monotonic()
        for step in range(80):
            bits=torch.arange(12)%2;logits,_,_=recall_trial(net,bits,[0,5,20][step%3]);loss=F.cross_entropy(logits,bits)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
        net.eval();g=torch.Generator().manual_seed(2026);bits=torch.randint(2,(128,),generator=g);metrics={}
        with torch.no_grad():
            for delay in [0,5,20]:
                metrics[str(delay)]={}
                for reset in ['none','state','memory','both']:
                    logits,_,_=recall_trial(net,bits,delay,reset);metrics[str(delay)][reset]=float((logits.argmax(1)==bits).float().mean())
        recall[name]={'accuracy':metrics,'seconds':time.monotonic()-start,'alpha':float(net.alpha[0].detach())}
    out['recall']={'training_steps':80,'seed':72,'test_seed':2026,'results':recall}
    Path('docs/dynamath/experiment-003/supplemental.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
