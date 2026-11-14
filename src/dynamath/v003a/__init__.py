"""003a: safe formula graphs on the unchanged nine-law Model1 runtime."""
import copy, math
from dataclasses import asdict
import torch
from torch import nn
from dynamath.v002 import Model as Base, Config, Session
from .ast import interpret, validate

# Leaf z is tanh(z); baseline is the leaf, preserving 002 exactly.
BASE={'op':'z'}
class Model(Base):
    def __init__(self,config,asts=None):
        super().__init__(config)
        if config.memories!=1 or config.cross or config.frozen: raise ValueError('Model1 required')
        self.asts=copy.deepcopy(asts if asts is not None else [BASE]*9)
        if len(self.asts)!=9: raise ValueError('nine laws required')
        for ast in self.asts: validate(ast)
        self.alpha=nn.Parameter(torch.zeros(9))
    @classmethod
    def from_002(cls,path):
        model,_=Base.load(path); obj=cls(model.config); obj.load_state_dict(model.state_dict(),strict=False); return obj
    def step(self,s,m,t):
        i=t%9; h=(s@self.v[i]+self.b[i]+m@self.q[i]).tanh()
        z=1.5*self.d[i].tanh()*s+h@self.u[i]+self.c[i]+m@self.r[i]
        candidate=interpret(self.asts[i],z,s,m,self.alpha[i])
        a=.05+.9*self.leak[i].sigmoid(); ns=(1-a)*s+a*candidate
        probability=(ns@self.gate[i]+self.gb[i]).sigmoid(); hard=(probability>=.5).to(s.dtype)
        gate=hard+(probability-probability.detach()) if self.training else hard
        value=math.e*(ns@self.writer[i]+self.wb[i]).tanh()
        return ns,((1-gate)*m[:,0]+gate*value)[:,None],hard
    def save(self,path,**extra):
        torch.save(dict(schema='dynamath-003a/v1',config=asdict(self.config),asts=self.asts,weights=self.state_dict(),**extra),path)
    @classmethod
    def load(cls,path):
        p=torch.load(path,map_location='cpu',weights_only=False)
        if p['schema']!='dynamath-003a/v1': raise ValueError('schema')
        obj=cls(Config(**p['config']),p['asts']);obj.load_state_dict(p['weights']);return obj.eval(),p
