"""Run the fixed Endophalon-3 experiment matrix through the existing model CLI.

No physics, splits, population, or generations are changed. Completed outputs
are never overwritten. --stage selects a bounded phase (each child <= 55 min).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'docs/endophalon3_runs/20261031-training'
CARDS = ROOT / 'models/endophalon3'
STAGES = {
    'a_binary': [('a_binary_s1', 'level_a', 1, False, True), ('a_binary_s2', 'level_a', 2, False, False)],
    'a_ablation': [('a_no_regions_s1', 'level_a', 1, True, False), ('a_no_regions_s2', 'level_a', 2, True, False)],
    'a_onehot': [('a_onehot_s1', 'level_a_onehot', 1, False, False)],
    'b_binary': [('b_binary_s1', 'level_b', 1, False, False), ('b_binary_s2', 'level_b', 2, False, False)],
    'b_onehot': [('b_onehot_s1', 'level_b_onehot', 1, False, False), ('b_onehot_s2', 'level_b_onehot', 2, False, False)],
}


def execute(item):
    name, config, seed, ablate, random = item
    stem = '20261031_' + name
    card = CARDS / (stem + '.json')
    log, meta = RUN / (name + '.log'), RUN / (name + '.run.json')
    if any(p.exists() for p in (card, log, meta)):
        if meta.exists() and json.loads(meta.read_text()).get('status') == 'completed' and card.exists():
            print('PRESERVED', name, flush=True)
            return True
        raise FileExistsError(f'Incomplete/existing outputs for {name}; inspect rather than overwrite')
    cmd = [sys.executable, '-u', '-m', 'tele_neuron.endophalon3', '--config', f'configs/endophalon3/{config}.json', '--seed', str(seed), '--workers', '4', '--output', str(card.relative_to(ROOT))]
    if ablate: cmd.append('--no-regions')
    if random: cmd.append('--random-baseline')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=str(ROOT/'src'))
    record = {'name':name, 'argv':cmd, 'start_utc':datetime.now(timezone.utc).isoformat(), 'base_sha':subprocess.check_output(['git','merge-base','HEAD','origin/master'],cwd=ROOT,text=True).strip(), 'checkout_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), 'search_seed':seed, 'physics_and_dataset_seed':1, 'generations':40, 'population':24, 'workers':4, 'chunk_size':96, 'status':'running'}
    meta.write_text(json.dumps(record,indent=2)+'\n')
    start = time.monotonic()
    with log.open('x') as f:
        f.write('$ '+' '.join(cmd)+'\n');f.flush()
        process = subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            rc = process.wait(timeout=3300)
            record.update(returncode=rc,status='completed' if rc==0 and card.exists() else 'failed')
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(process.pid,signal.SIGTERM)
            process.wait()
            record.update(returncode=process.returncode,status='timeout',failure='Child exceeded 55 minutes; raw log preserved; no model claimed')
    record.update(seconds=time.monotonic()-start,end_utc=datetime.now(timezone.utc).isoformat())
    if card.exists():record['card']=str(card.relative_to(ROOT))
    meta.write_text(json.dumps(record,indent=2)+'\n')
    print(record['status'],name,round(record['seconds'],2),flush=True)
    return record['status']=='completed'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=STAGES,required=True)
    args=parser.parse_args()
    if not (RUN/'phase0_complete.json').exists():raise RuntimeError('Phase 0 must pass first')
    RUN.mkdir(parents=True,exist_ok=True);CARDS.mkdir(parents=True,exist_ok=True)
    # Two independent outer search seeds share fixed streams, as the CLI specifies.
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(execute,STAGES[args.stage]))
    if not all(results):raise SystemExit(1)


if __name__=='__main__':main()
