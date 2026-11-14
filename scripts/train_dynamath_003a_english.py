"""Train/resume an independent English Model1. Run from repository root."""
import argparse,json,random
from pathlib import Path
import torch
from dynamath.v002 import Config
from dynamath.v002.__main__ import generate
from dynamath.v003a import Model
from dynamath.v003a.english_data import load,audit,pairs
from dynamath.v003a.training import streams,train,evaluate

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=5000);ap.add_argument('--seconds',type=int,default=1200);ap.add_argument('--resume');args=ap.parse_args()
    cfg=json.loads(Path('configs/dynamath/experiment-003a.json').read_text());cfg.update(steps=args.steps,seconds_cap_per_run=args.seconds)
    torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed']);random.seed(cfg['seed'])
    splits=load();tok,report=audit(splits)
    data=streams(splits['train'][:cfg['train_dialogues']],tok,cfg['batch'])
    if args.resume: model,p=Model.load(args.resume);restart=p['restart']
    else: model=Model(Config(tok.vocabulary,memories=1,latent=64,rank=12));restart=None
    stats,restart=train(model,data,cfg,restart)
    target=Path('models/dynamath/003a/english/control.pt');target.parent.mkdir(parents=True,exist_ok=True)
    model.save(target,restart=restart,training=stats,config_run=cfg,source_sha256=report['archive_sha256'],data_path='work/dailydialog/data.zip')
    val=evaluate(model,streams(splits['validation'][:cfg['validation_dialogues']],tok,cfg['batch']))
    generation=[generate(model,pairs(d)[0]['Q'],cfg['generation_ticks']) for d in splits['validation'][:3]]
    out={'data':report,'training':stats,'validation':val,'validation_generation':generation,'parameters':sum(p.numel() for p in model.parameters()),'config':cfg}
    Path('docs/dynamath/experiment-003a/english-results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'training':stats,'validation':val},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
