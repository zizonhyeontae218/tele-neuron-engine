"""Build or run independent A/B models with explicit software value connections."""
import argparse
import json
from pathlib import Path
from .router import SoftwareModel,Router


def main():
    ap=argparse.ArgumentParser(description='Endophalon 3: independent A/B, software router/registers')
    source=ap.add_mutually_exclusive_group(required=True);source.add_argument('--config');source.add_argument('--model')
    ap.add_argument('--build');ap.add_argument('--program');ap.add_argument('--output');ap.add_argument('--checkpoint-in');ap.add_argument('--checkpoint-out')
    args=ap.parse_args()
    if args.config:
        if not args.build or any([args.program,args.output,args.checkpoint_in,args.checkpoint_out]):ap.error('--config requires --build only')
        config=Path(args.config);raw=json.loads(config.read_text());root=config.parent/raw.get('root','.')
        SoftwareModel.from_cards(root/raw['A'],root/raw['B'],root).save(args.build);print('Saved software-linked model:',args.build);return
    if args.build or not args.program:ap.error('--model requires --program')
    paths=[Path(p) for p in [args.output,args.checkpoint_out] if p]
    if len({p.resolve() for p in paths})!=len(paths):ap.error('Output/checkpoint paths must differ')
    for p in paths:
        if p.exists():raise FileExistsError(p)
    model=SoftwareModel.load(args.model)
    router=Router.restore(model,json.loads(Path(args.checkpoint_in).read_text())) if args.checkpoint_in else Router(model)
    results=[]
    for line in Path(args.program).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):continue
        result=router.execute(line)
        if result is not None:results.append(result)
    payload={'family':'Teleneuron-Endophalon-3','mode':'independent models/software wiring','events':router.events,
             'local_events':{k:v.events for k,v in router.sessions.items()},'local_clocks':{k:v.clock for k,v in router.sessions.items()},'registers':router.registers,'results':results}
    text=json.dumps(payload,indent=2,allow_nan=False)+'\n'
    if args.output:
        with Path(args.output).open('x') as f:f.write(text)
    else:print(text,end='')
    if args.checkpoint_out:
        with Path(args.checkpoint_out).open('x') as f:json.dump(router.snapshot(),f,indent=2,allow_nan=False);f.write('\n')

if __name__=='__main__':main()
