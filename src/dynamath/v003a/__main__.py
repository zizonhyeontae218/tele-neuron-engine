"""Dataset-independent generation from a 003a checkpoint."""
import argparse,json,torch
from dynamath.v003a import Model
from dynamath.v002.__main__ import generate

def main():
    ap=argparse.ArgumentParser();ap.add_argument('checkpoint');ap.add_argument('--question',required=True);ap.add_argument('--ticks',type=int,default=180);args=ap.parse_args()
    torch.set_num_threads(1);model,_=Model.load(args.checkpoint)
    print(json.dumps(generate(model,args.question,args.ticks),ensure_ascii=False))
if __name__=='__main__':main()
