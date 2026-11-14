"""Bounded, shape preserving expression language; no evaluation of source text."""
import json, math
import torch

LEAVES={'z','S','m'}
UNARY={'tanh','sin','softsign','atan','abs','sigmoid','clip','poly','scale'}
BINARY={'add','sub','mul','residual','gate'}
BASE={'op':'z'}

def canonical(n):
    op=n['op']
    if op in LEAVES: return {'op':op}
    if op in UNARY: return {'op':op,'arg':canonical(n['arg'])}
    if op in BINARY:
        a,b=canonical(n['left']),canonical(n['right'])
        if op in ('add','mul') and dumps(a)>dumps(b): a,b=b,a
        return {'op':op,'left':a,'right':b}
    raise ValueError('unknown operation')

def dumps(n): return json.dumps(n,sort_keys=True,separators=(',',':'))

def validate(n):
    def visit(x,d):
        if not isinstance(x,dict) or d>7: raise ValueError('depth or node')
        op=x.get('op')
        keys={'op'} if op in LEAVES else {'op','arg'} if op in UNARY else {'op','left','right'} if op in BINARY else None
        if keys is None or set(x)!=keys: raise ValueError('operator or fields')
        if op in LEAVES: return 1
        if op in UNARY: return 1+visit(x['arg'],d+1)
        if dumps(x['left'])==dumps(x['right']) and op in ('sub','gate'): raise ValueError('degenerate')
        return 1+visit(x['left'],d+1)+visit(x['right'],d+1)
    count=visit(n,1)
    if count>24: raise ValueError('node budget')
    if '"z"' not in dumps(n): raise ValueError('must depend on z')
    return count

def interpret(n,z,s,m,alpha):
    op=n['op']
    if op=='z': return z.tanh()
    if op=='S': return s.tanh()
    if op=='m': return m.tanh().expand_as(z)
    if op in UNARY:
        a=interpret(n['arg'],z,s,m,alpha)
        if op=='tanh': return a.tanh()
        if op=='sin': return a.sin()
        if op=='softsign': return a/(1+a.abs())
        if op=='atan': return a.atan()*(2/math.pi)
        if op=='abs': return a.abs()
        if op=='sigmoid': return 2*a.sigmoid()-1
        if op=='clip': return a.clamp(-1,1)
        if op=='poly': return .5*(a+a*a*a)
        if op=='scale': return alpha.tanh()*a
    a=interpret(n['left'],z,s,m,alpha);b=interpret(n['right'],z,s,m,alpha)
    if op=='add': return (a+b)*.5
    if op=='sub': return (a-b)*.5
    if op=='mul': return a*b
    if op=='residual': return (a+alpha.tanh()*b).clamp(-1,1)
    if op=='gate': return torch.sigmoid(b)*a+(1-torch.sigmoid(b))*b
    raise ValueError('operator')

def expression(n):
    op=n['op']
    if op=='z': return 'tanh(z)'
    if op=='S': return 'tanh(S)'
    if op=='m': return 'tanh(m)'
    if op in UNARY:
        a=expression(n['arg']); return {'scale':f'tanh(α)·({a})','poly':f'({a}+({a})³)/2','sigmoid':f'2σ({a})−1','clip':f'clip({a},−1,1)'}.get(op,f'{op}({a})')
    a,b=expression(n['left']),expression(n['right'])
    return {'add':f'({a}+{b})/2','sub':f'({a}−{b})/2','mul':f'({a})·({b})','residual':f'clip({a}+tanh(α)·({b}),−1,1)','gate':f'σ({b})·({a})+(1−σ({b}))·({b})'}[op]
