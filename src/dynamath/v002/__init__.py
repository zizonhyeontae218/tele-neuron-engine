"""Nine-law single-state dynamics with one, three or nine writable scalars."""
from dataclasses import dataclass, asdict
from collections import deque
import math
import torch
from torch import nn

@dataclass
class Config:
    vocabulary: str
    memories: int = 9
    cross: bool = False
    frozen: bool = False
    latent: int = 64
    rank: int = 12
    gain: float = 6.

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.memories not in (1,3,9) or (config.cross and config.memories != 3):
            raise ValueError('invalid scalar topology')
        self.config = config
        n, r = 2*len(config.vocabulary)+config.latent, config.rank
        self.n = n
        self.v = nn.Parameter(torch.randn(9,n,r)/n**.5)
        self.u = nn.Parameter(torch.randn(9,r,n)*.12)
        self.b = nn.Parameter(torch.zeros(9,r))
        self.c = nn.Parameter(torch.zeros(9,n))
        self.d = nn.Parameter(torch.full((9,n),.55))
        self.leak = nn.Parameter(torch.linspace(-1,1,n).repeat(9,1))
        self.q = nn.Parameter(torch.randn(9,config.memories,r)*.05)
        self.r = nn.Parameter(torch.randn(9,config.memories,n)*.05)
        self.writer = nn.Parameter(torch.randn(9,n)*.02)
        self.gate = nn.Parameter(torch.randn(9,n)*.02)
        self.wb = nn.Parameter(torch.zeros(9))
        self.gb = nn.Parameter(torch.full((9,),-1.))
    def owner(self,i):
        return i if self.config.memories == 9 else i//3 if self.config.memories == 3 else 0
    def zero(self,batch=1):
        constants = [0.,1.,math.e]
        m = self.v.new_tensor([constants[i%3] for i in range(self.config.memories)])
        return self.v.new_zeros(batch,self.n), m.repeat(batch,1)
    def inject(self,s,tokens):
        if tokens is None: return s
        tokens = torch.as_tensor(tokens,device=s.device,dtype=torch.long)
        if tokens.ndim == 1: tokens=tokens[:,None]
        if tokens.shape[0] != len(s) or (tokens<0).any() or (tokens>=len(self.config.vocabulary)).any():
            raise ValueError('invalid input address')
        return s+torch.zeros_like(s).scatter_add(1,tokens,torch.ones_like(tokens,dtype=s.dtype))
    def step(self,s,m,t):
        i=t%9; owner=self.owner(i)
        mask = m.new_zeros(self.config.memories)
        if self.config.cross: mask[:]=1
        else: mask[owner]=1
        read=m*mask
        h=(s@self.v[i]+self.b[i]+read@self.q[i]).tanh()
        candidate=(1.5*self.d[i].tanh()*s+h@self.u[i]+self.c[i]+read@self.r[i]).tanh()
        a=.05+.9*self.leak[i].sigmoid()
        ns=(1-a)*s+a*candidate
        # Read old m, update S, then hard retain-or-replace. STE only in training.
        probability=(ns@self.gate[i]+self.gb[i]).sigmoid()
        hard=(probability>=.5).to(s.dtype)
        gate=hard+(probability-probability.detach()) if self.training else hard
        value=math.e*(ns@self.writer[i]+self.wb[i]).tanh()  # [-e,e]
        new= m[:,owner] if self.config.frozen else (1-gate)*m[:,owner]+gate*value
        nm=m.clone(); nm[:,owner]=new
        return ns,nm,hard
    def logits(self,s):
        w=len(self.config.vocabulary)
        return self.config.gain*s[:,w:2*w]
    def save(self,path,**extra):
        torch.save(dict(schema='dynamath-002/v1',config=asdict(self.config),weights=self.state_dict(),**extra),path)
    @classmethod
    def load(cls,path):
        p=torch.load(path,map_location='cpu',weights_only=False)
        if p['schema']!='dynamath-002/v1': raise ValueError('schema')
        model=cls(Config(**p['config'])); model.load_state_dict(p['weights'])
        return model.eval(),p

class Session:
    def __init__(self,model,threshold=.1):
        self.model=model; self.s,self.m=model.zero(); self.t=0
        self.queue=deque(); self.last_poll=-1; self.threshold=threshold
        self.log=[]; self.events=[]; self.last_event=None
    def enqueue(self,tokens,source='external'):
        self.queue.extend((int(x),source) for x in tokens)
    @torch.no_grad()
    def tick(self):
        t=self.t; admitted=None
        if t%3==0 and self.queue:
            admitted=self.queue.popleft(); self.s=self.model.inject(self.s,[admitted[0]])
        old=self.m.clone()
        self.s,self.m,g=self.model.step(self.s,self.m,t); self.t+=1
        self.last_event=None
        if t%3 in (1,2): self.poll()
        row=dict(tick=t,law=t%9+1,block=t%9//3,admitted=admitted,
                 event=self.last_event,score=float(self.model.logits(self.s).max()/self.model.config.gain),
                 queue_depth=len(self.queue),m=self.m[0].tolist(),gate=float(g[0]),
                 changed=bool((old!=self.m).any()))
        self.log.append(row)
        return self.last_event
    def poll(self):
        t=self.t-1
        if t<0 or t%3==0 or self.last_poll==t: return None
        self.last_poll=t
        scores=self.model.logits(self.s)[0]/self.model.config.gain
        token=int(scores.argmax())  # first address wins ties
        if float(scores[token])<self.threshold or token==3: return None
        self.last_event=token; self.events.append((t,token))
        self.enqueue([token],'feedback')
        return token
    def snapshot(self):
        return dict(s=self.s.tolist(),m=self.m.tolist(),t=self.t,queue=list(self.queue),
                    last_poll=self.last_poll,threshold=self.threshold,events=self.events)
    def restore(self,p):
        self.s=self.s.new_tensor(p['s']); self.m=self.m.new_tensor(p['m']); self.t=p['t']
        self.queue=deque(tuple(x) for x in p['queue']); self.last_poll=p['last_poll']
        self.threshold=p['threshold']; self.events=[tuple(x) for x in p['events']]
