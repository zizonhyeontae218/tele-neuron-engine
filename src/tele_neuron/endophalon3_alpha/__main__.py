"""Build a two-bank bundle or execute commands with an optional session checkpoint."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from .model import Model
from .session import Session


def main():
    ap=argparse.ArgumentParser(description='Endophalon 3: separate A/B modules, shared physical memory')
    source=ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--config',help='Bank card manifest, used with --build')
    source.add_argument('--model',help='Saved Endophalon 3 bundle')
    ap.add_argument('--build',help='New model bundle path; never overwrites')
    ap.add_argument('--program',help='Text program: A/B OP a b, level A/B, or OP a b')
    ap.add_argument('--checkpoint-in');ap.add_argument('--checkpoint-out')
    ap.add_argument('--output',help='New JSON result file; otherwise stdout')
    args=ap.parse_args()
    if args.config:
        if not args.build or any([args.program,args.checkpoint_in,args.checkpoint_out,args.output]):ap.error('--config requires --build only')
        path=Path(args.config);raw=json.loads(path.read_text());root=path.parent/raw.get('root','.')
        Model.from_cards(root/raw['A'],root/raw['B'],root).save(args.build)
        print('Saved Endophalon 3 bundle:',args.build);return
    if args.build or not args.program:ap.error('--model requires --program')
    # Reject conflicting output paths before any simulation.
    outputs=[Path(p) for p in [args.output,args.checkpoint_out] if p]
    if len({p.resolve() for p in outputs})!=len(outputs):ap.error('Output and checkpoint paths must differ')
    for p in outputs:
        if p.exists():raise FileExistsError(p)
    model=Model.load(args.model)
    session=Session.restore(model,json.loads(Path(args.checkpoint_in).read_text())) if args.checkpoint_in else Session(model)
    lines=[line.strip() for line in Path(args.program).read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
    results=[]
    for line in lines:
        result=session.execute(line)
        if result is not None:
            row=asdict(result);row['features']=row['features'].tolist();results.append(row)
    payload={'family':'Teleneuron-Endophalon-3','events':session.events,'clock':session.clock,'results':results}
    text=json.dumps(payload,indent=2,allow_nan=False)+'\n'
    if args.output:
        with Path(args.output).open('x') as f:f.write(text)
    else:print(text,end='')
    if args.checkpoint_out:
        with Path(args.checkpoint_out).open('x') as f:json.dump(session.snapshot(),f,indent=2,allow_nan=False);f.write('\n')

if __name__=='__main__':main()
