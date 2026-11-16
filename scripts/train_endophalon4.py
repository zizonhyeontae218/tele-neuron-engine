"""Run one local five-generation segment per independent coupling direction."""
import os
# Must precede NumPy imports, including when invoked from a clean shell.
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tele_neuron.endophalon4.training import latest,segment,verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--segment-generations',type=int,default=5)
    p.add_argument('--resume',choices=['auto'],default='auto')
    p.add_argument('--direction',choices=['A_to_B','B_to_A','both'],default='both')
    p.add_argument('--status',action='store_true');p.add_argument('--verify-checkpoint',action='store_true')
    a=p.parse_args()
    if not 1<=a.segment_generations<=5:p.error('Use 1–5 generations per segment')
    for d in (['A_to_B','B_to_A'] if a.direction=='both' else [a.direction]):
        if a.status or a.verify_checkpoint:
            s=latest(d);verify(s,d)
            print(json.dumps({'direction':d,'generation':s['generation'],'evaluations':s['evaluations'],
                'checkpoint_valid':True,'segments':s['segments'],'elapsed_seconds':s['elapsed_seconds']}),flush=True)
        else:segment(d,a.segment_generations)


if __name__=='__main__':main()
