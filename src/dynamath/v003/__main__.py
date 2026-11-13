"""Dataset-independent 003 checkpoint generation and snapshot replay."""
import argparse,json
from pathlib import Path
import torch
from dynamath.v003 import Model, Session
from dynamath.korean_data import Tokenizer

def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--question',default='안녕하세요');p.add_argument('--ticks',type=int,default=180);p.add_argument('--restore');p.add_argument('--snapshot');a=p.parse_args()
    torch.set_num_threads(2);model,_=Model.load(a.checkpoint);session=Session(model)
    if a.restore: session.restore(json.loads(Path(a.restore).read_text()))
    else: session.enqueue([0]+Tokenizer(model.config.vocabulary).encode(a.question)+[1])
    for _ in range(a.ticks): session.tick()
    result={'events':session.events,'text':Tokenizer(model.config.vocabulary).decode([e[1] for e in session.events]),'snapshot':session.snapshot()}
    if a.snapshot: Path(a.snapshot).write_text(json.dumps(result['snapshot'],ensure_ascii=False))
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
