"""Budgeted successive halving over discrete graphs with continuous training callbacks.

The caller owns data separation, checkpoint serialization and evaluation. Budgets are
cumulative updates, so each survivor resumes its own optimizer and recurrent state.
"""
import copy
from .ast import validate

def race(origin, candidates, stages, fit, evaluate, rank):
    """Return all stage evidence and survivor objects; includes an unedited control.

    fit(model, cumulative_steps, restart) returns (stats, restart). evaluate sees
    validation only. rank receives validation and node count (smaller is better).
    Each initial arm is copied from the same origin; no test data is accepted here.
    """
    if not stages or any(n<1 or steps<1 for n,steps in stages): raise ValueError('invalid stages')
    if any(stages[i][0]>stages[i-1][0] or stages[i][1]<=stages[i-1][1] for i in range(1,len(stages))): raise ValueError('stages must shrink and budgets increase')
    entries=[{'name':'control','model':copy.deepcopy(origin),'restart':None}]
    for i,(law,ast) in enumerate(candidates):
        if not 0<=law<9:raise ValueError('law')
        validate(ast);m=copy.deepcopy(origin);m.asts[law]=copy.deepcopy(ast)
        entries.append({'name':f'candidate{i+1}','model':m,'restart':None})
    history=[]
    for stage,(keep,steps) in enumerate(stages):
        if keep>len(entries):raise ValueError('not enough candidates')
        rows=[]
        for entry in entries:
            stats,restart=fit(entry['model'],steps,entry['restart']);entry['restart']=restart
            if stats['steps']!=steps:raise RuntimeError('budget interrupted; save restart rather than compare unequal arms')
            metrics=evaluate(entry['model']);nodes=sum(validate(a) for a in entry['model'].asts)
            rows.append((rank(metrics,nodes),entry,{'name':entry['name'],'metrics':metrics,'nodes':nodes,'training':stats}))
        rows.sort(key=lambda x:(x[0],x[1]['name']));history.append({'stage':stage,'updates':steps,'results':[row[2] for row in rows]})
        entries=[row[1] for row in rows[:keep]]
    return entries,history
