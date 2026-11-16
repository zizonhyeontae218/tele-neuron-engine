"""Consume fixed held-out streams once after both budgets complete."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tele_neuron.endophalon4.evaluation import run
if __name__=='__main__':
    print('Primary success:',run()['primary_success'])
