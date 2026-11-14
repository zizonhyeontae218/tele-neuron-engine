"""Recreate the catalog and numerically probe each AST on finite extreme inputs."""
import collections,json,time,torch
from pathlib import Path
from dynamath.v003a.formula_forge import catalog,manifest
from dynamath.v003a.ast import interpret,validate

def main():
    torch.set_num_threads(1);start=time.monotonic();rows=catalog();counts=collections.Counter();bad=[]
    z=torch.tensor([[-1.e10,-10.,-1.,-.1,0.,.2,1.,10.,1.e10]])
    s=torch.flip(z,[1]);m=torch.tensor([[2.7182818]]);alpha=torch.tensor(.7)
    for i,row in enumerate(rows):
        try:
            count=validate(row['ast']);out=interpret(row['ast'],z,s,m,alpha)
            if out.shape!=z.shape or not torch.isfinite(out).all() or out.abs().max()>1.000001:bad.append(i)
            if (out.max()-out.min()).abs()<1.e-7:counts['constant_on_probe']+=1
            counts['max_nodes']=max(counts['max_nodes'],count)
        except Exception: bad.append(i)
    report=manifest(rows,43);report['probe']={'cases':len(rows),'bad_indices':bad,'input':'z spans -1e10..1e10; S reversed; m=e; alpha=.7','constant_on_probe':counts['constant_on_probe'],'max_nodes':counts['max_nodes'],'seconds':time.monotonic()-start,'scope':'finite sample only; bounded operators guarantee range for finite inputs; algebraic duplicates beyond simple checks not exhaustively detected'}
    Path('docs/dynamath/experiment-003a/formula-library.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if bad:raise SystemExit(1)
if __name__=='__main__':main()
