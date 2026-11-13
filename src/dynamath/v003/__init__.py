"""Serializable typed expression search on the 002 nine-law runtime."""
import copy
import math
from dataclasses import asdict
import torch
from torch import nn
from dynamath.v002 import Model as Base, Config, Session

BASE = {'op': 'tanh', 'arg': {'op': 'input'}}

def mutation():
    # At alpha=0 this preserves the parent function exactly.
    return {'op': 'clip', 'arg': {'op': 'add', 'left': copy.deepcopy(BASE),
            'right': {'op': 'scale', 'arg': {'op': 'sin', 'arg': {'op': 'input'}}}}}

def validate(node, depth=0):
    if not isinstance(node, dict) or depth > 6: raise ValueError('invalid expression')
    op = node.get('op')
    keys = {'input': {'op'}, 'tanh': {'op','arg'}, 'sin': {'op','arg'},
            'clip': {'op','arg'}, 'scale': {'op','arg'}, 'add': {'op','left','right'}}
    if op not in keys or set(node) != keys[op]: raise ValueError('invalid operator or fields')
    for key in ('arg','left','right'):
        if key in node: validate(node[key], depth+1)

def interpret(node, x, alpha):
    op=node['op']
    if op=='input': return x
    if op=='add': return interpret(node['left'],x,alpha)+interpret(node['right'],x,alpha)
    a=interpret(node['arg'],x,alpha)
    if op=='tanh': return a.tanh()
    if op=='sin': return a.sin()
    if op=='clip': return a.clamp(-1,1)
    if op=='scale': return alpha.tanh()*a
    raise ValueError('unknown operator')

def expression(node):
    op=node['op']
    if op=='input': return 'z'
    if op=='add': return '('+expression(node['left'])+' + '+expression(node['right'])+')'
    if op=='scale': return 'tanh(α)·'+expression(node['arg'])
    return op+'('+expression(node['arg'])+')'

class Model(Base):
    def __init__(self,config,asts=None):
        super().__init__(config)
        if config.memories != 1 or config.cross or config.frozen: raise ValueError('003 requires Model1')
        self.asts=copy.deepcopy(asts if asts is not None else [BASE]*9)
        if len(self.asts)!=9: raise ValueError('nine laws required')
        for ast in self.asts: validate(ast)
        self.alpha=nn.Parameter(torch.zeros(9))
    @classmethod
    def from_002(cls,path):
        base,p=Base.load(path); model=cls(base.config)
        model.load_state_dict(base.state_dict(),strict=False)
        return model
    def step(self,s,m,t):
        i=t%9; read=m
        h=(s@self.v[i]+self.b[i]+read@self.q[i]).tanh()
        z=1.5*self.d[i].tanh()*s+h@self.u[i]+self.c[i]+read@self.r[i]
        candidate=interpret(self.asts[i],z,self.alpha[i])
        a=.05+.9*self.leak[i].sigmoid(); ns=(1-a)*s+a*candidate
        probability=(ns@self.gate[i]+self.gb[i]).sigmoid()
        hard=(probability>=.5).to(s.dtype)
        gate=hard+(probability-probability.detach()) if self.training else hard
        value=math.e*(ns@self.writer[i]+self.wb[i]).tanh()
        nm=((1-gate)*m[:,0]+gate*value)[:,None]
        return ns,nm,hard
    def save(self,path,**extra):
        torch.save(dict(schema='dynamath-003/v1',config=asdict(self.config),asts=self.asts,
                        weights=self.state_dict(),**extra),path)
    @classmethod
    def load(cls,path):
        p=torch.load(path,map_location='cpu',weights_only=False)
        if p['schema']!='dynamath-003/v1': raise ValueError('schema')
        model=cls(Config(**p['config']),p['asts']);model.load_state_dict(p['weights'])
        return model.eval(),p
