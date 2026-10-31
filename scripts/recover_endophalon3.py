"""Recover only the two infrastructure-interrupted experiments; scientific code is unchanged."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'docs/endophalon3_runs/20261031-training'
BASE='1425d63fafee8a3ab5127b59a51324600fed00e3'

def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def train(seed):
    from tele_neuron.endophalon3.config import load_config
    config=load_config(ROOT/'configs/endophalon3/level_b_onehot.json')
    assert (config.network.nodes,config.physics.substeps,config.zones)==(144,2,16)
    assert config.outer['generations']==40 and config.outer['population']==24
    name=f'b_onehot_s{seed}';card=ROOT/f'models/endophalon3/20261031_{name}.json'
    log,meta=RUN/(name+'.log'),RUN/(name+'.run.json')
    assert not any(p.exists() for p in (card,log,meta)), 'Never overwrite published outputs'
    cmd=[sys.executable,'-u','-m','tele_neuron.endophalon3','--config','configs/endophalon3/level_b_onehot.json','--seed',str(seed),'--workers','4','--output',str(card.relative_to(ROOT))]
    record={'name':name,'argv':cmd,'start_utc':datetime.now(timezone.utc).isoformat(),'base_sha':BASE,
        'checkout_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'search_seed':seed,'physics_and_dataset_seed':1,'generations':40,'population':24,'workers':4,'chunk_size':96,
        'status':'running','attempt':'github-actions-recovery-from-original-theta-init',
        'prior_attempt':'infrastructure lost; at least 25 generations observed; no final theta/readout recovered',
        'environment':{'python':sys.version,'platform':platform.platform(),'cpu_count':os.cpu_count(),'packages':{k:md.version(k) for k in ['numpy','matplotlib','pytest']}},
        'workflow_url':os.environ.get('GITHUB_SERVER_URL','https://github.com')+'/'+os.environ.get('GITHUB_REPOSITORY','')+'/actions/runs/'+os.environ.get('GITHUB_RUN_ID','')}
    write(meta,record);start=time.monotonic()
    with log.open('x') as f:
        f.write('$ '+' '.join(cmd)+'\n');f.flush()
        process=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=process.wait(timeout=3000)
            record.update(returncode=code,status='completed' if code==0 and card.exists() else 'failed')
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM);process.wait()
            record.update(returncode=process.returncode,status='timeout',failure='50-minute child limit')
    record.update(seconds=time.monotonic()-start,end_utc=datetime.now(timezone.utc).isoformat())
    if card.exists():record['card']=str(card.relative_to(ROOT))
    write(meta,record)
    print(record['status'],name,record['seconds'],flush=True)
    if record['status']!='completed':raise SystemExit(1)

def finalize():
    with (RUN/'recovery_audit.log').open('x') as log:
        audit=subprocess.run([sys.executable,'-u','scripts/audit_endophalon3_training.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    subprocess.run([sys.executable,'scripts/summarize_endophalon3_training.py'],cwd=ROOT,check=True)
    summary=json.loads((RUN/'summary.json').read_text());rows=summary['completed'];missing=summary['missing']
    checks={}
    allowed={'GOAL.md','README.md','README.ko.md','docs/endophalon3.md','docs/model_commands.md'}
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],cwd=ROOT,text=True).splitlines()
    for path in paths:
        if path in allowed:continue
        actual=(ROOT/path).read_bytes();expected=subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT)
        checks[path]={'sha256':hashlib.sha256(actual).hexdigest(),'unchanged':actual==expected}
    assert all(v['unchanged'] for v in checks.values()), 'Scientific source/config/original evidence drift'
    write(RUN/'preservation.json',{'base_sha':BASE,'files':checks,'all_unchanged':True,'allowed_updated_documents':sorted(allowed)})
    artifacts={}
    for row in rows:
        name=row['run'];meta=json.loads((RUN/(name+'.run.json')).read_text())
        result=json.loads((RUN/(name+'.audit.json')).read_text())
        assert meta['status']=='completed' and meta['returncode']==0 and result['replay']=='PASS'
        assert (result['invariants']['nodes'],result['invariants']['substeps'],result['invariants']['zones'])==(144,2,16)
        files=[ROOT/meta['card']]+[RUN/(name+suffix) for suffix in ['.log','.run.json','.audit.json','.predictions.csv']]
        artifacts[name]={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
    write(RUN/'artifact_manifest.json',artifacts)
    write(RUN/'recovery_verification.json',{'audit_returncode':audit.returncode,'completed_models':len(rows),'incomplete_models':missing,
        'scientific_source_and_original_evidence_preserved':True,
        'filetree_update_lint':'Not run: executor-owned Filetree skill became unavailable after the original executor was lost. Original generated index/ledger preserved.',
        'notes':['Recovered runs start from original theta_init and original optimizer seeds, not from an intermediate theta.','Interrupted cloud attempts consumed at least 601 theta evaluations each; full logs/timings were not recovered.','Successful model search budgets remain 961 theta evaluations each; hardware differs for the recovered pair.']})
    update_docs(summary)
    report=ROOT/'docs/endophalon3_training_report.ko.md'
    extra=['','## 실행 환경 중단과 복구','',
        '원래 Work 실행기가 연결 해제되어 B One-hot Seed 1·2의 최종 모델을 회수하지 못했다. 마지막 직접 확인은 양쪽 25/40세대다. 부분 실행을 성공으로 계산하지 않았다.',
        '각 중단 실행은 최소 601 theta 평가를 사용했다(초기 1 + 25×24). 정확한 최종 진행률·전체 원본 로그·wall time은 회수하지 못했으며, 관측한 tail 발췌만 interrupted_runs.json에 보존했다.',
        '복구 실행은 GitHub Actions에서 기존 theta_init과 같은 탐색 시드 1·2로 처음부터 수행했다. 중간 theta를 넘기거나 시드를 바꾸지 않았고, 각 완료 실행은 기존 CLI의 40×24 예산을 유지했다. 복구 환경·실제 argv·시간은 각 .run.json에 기록했다.',
        '원래 실행기와 복구 runner의 하드웨어가 다르므로 마지막 두 실행의 wall time을 모델 구조 차이 하나로 해석할 수 없다. 폐기된 부분 실행 비용은 완료 모델 학습 비용과 별개로 존재한다.',
        'Filetree 스킬의 updater/linter는 실행기 손실 뒤 사용할 수 없어 마지막 재생성·lint를 수행하지 못했다. 기존 생성 인덱스와 ledger는 보존했으며, 검증된 신규/기존 증거의 실제 SHA-256은 artifact_manifest.json과 preservation.json에 기록했다.',
        '복구 audit return code: '+str(audit.returncode)+'. 완료 '+str(len(rows))+'/9; 미완료: '+(', '.join(missing) if missing else '없음')+'.','']
    report.write_text(report.read_text()+'\n'.join(extra))
    print('Consolidated',len(rows),'audited models; missing',missing,flush=True)

def update_docs(summary):
    rows=summary['completed'];primary=[r for r in rows if r['level']=='A' and r['slow_region']]
    passed=sum(bool(r['a_accuracy_gate']) for r in primary);pct=lambda x:f'{100*x:.2f}%'
    path=ROOT/'docs/endophalon3.md';text=path.read_text()
    text=text.replace('(base, untrained)','(training experiments)',1).replace('Single zone readout','Single readout (16 zones)')
    old='**Status:** the base is built and tested. **Nothing has been trained.** Start\nwith [endophalon3_handoff.md](endophalon3_handoff.md).'
    text=text.replace(old,f'**Status (2026-10-31):** {len(rows)}/9 planned models trained and audited; A accuracy gates {passed}/{len(primary)}.\nSee the [research report](endophalon3_training_report.ko.md). The original\n[handoff](endophalon3_handoff.md) remains the fixed experiment protocol.')
    ops=['xor','recall1','recall2','xor_mem','and','or','parity3w','and_mem']
    lines=['','## Fixed-budget results — 2026-10-31','',
        'Original scientific source/configs/models/logs are preserved. All reported training runs use 144 nodes, 2 substeps, 16 zones and 40 generations ×24 population (961 theta evaluations). A and B train separately. Search seeds vary; physics and dataset seed remain 1.','',
        '| Run | Overall | '+' | '.join(ops)+' |','| '+' | '.join(['---']*10)+' |']
    for row in rows:lines.append('| '+row['run']+' | '+pct(row['test_accuracy'])+' | '+' | '.join(pct(row['per_op'][op]) if op in row['per_op'] else '—' for op in ops)+' |')
    lines+=['',f'A accuracy gates (>=90% overall and >=85% each op): {passed}/{len(primary)} non-ablated runs.','',
        'Slow Region recall2 effect: seed 1 +5.13 pp, seed 2 -6.84 pp; mean -0.85 pp, conditional paired-stream 95% CI [-3.66, +2.01] pp. Consistent improvement is not demonstrated. This interval conditions on two search seeds.','',
        'A binary seed-1 random search: 73.75%, versus trained 78.85% and initial 71.46%, each search using 961 theta evaluations.','',
        '85 pytest tests and 9 CLI/config smoke combinations passed; four untrained probes and four original ProMax native-task cards replay exactly. Saved E3 readouts, per-op accuracy, split fingerprints, regularizers and finite values are audited.','',
        'Actual objective scope: validation MSE/noise, train-fitted weight complexity, combined train+validation kinetic-energy drift. Numerical measurements cover 64-event streams, not indefinite stability.','',
        'New cards: models/endophalon3/20261031_*.json. Raw logs, run metadata, predictions, checksums and audits: docs/endophalon3_runs/20261031-training/.','',
        'The original executor was lost during B one-hot training. The incomplete attempts are recorded separately; recovered attempts restart from the original warm start and seeds. See the research report for hardware/timing and Filetree limitations.','']
    path.write_text(text+'\n'.join(lines))
    for file,heading,old,new in [
        ('README.md','## Teleneuron-Endophalon-3 (base, untrained)','The base and tests are in place. **No model has been trained yet.**\nStart at [docs/endophalon3_handoff.md](docs/endophalon3_handoff.md).',
         f'{len(rows)} fixed-budget models have been trained and audited; A accuracy gates {passed}/{len(primary)}. All seeds and negative results are reported.\n[Results](docs/endophalon3.md) · [research report](docs/endophalon3_training_report.ko.md).'),
        ('README.ko.md','## Teleneuron-Endophalon-3 (기반만 구성, 학습 전)','기반 코드와 테스트는 준비돼 있지만 **아직 학습한 모델은 없습니다.** [docs/endophalon3_handoff.md](docs/endophalon3_handoff.md)부터 보세요.',
         f'기존 설계와 40×24 예산으로 모델 {len(rows)}개를 학습·검증했습니다. 기본 A 모델의 정확도 조건 통과는 {passed}/{len(primary)}개입니다. 모든 시드·부정적 결과·실행 환경 중단 및 복구를 기록했습니다.\n[결과](docs/endophalon3.md) · [연구 보고서](docs/endophalon3_training_report.ko.md).')]:
        path=ROOT/file;text=path.read_text();assert heading in text and old in text
        text=text.replace(heading,'## Teleneuron-Endophalon-3 (training experiments)' if file=='README.md' else '## Teleneuron-Endophalon-3 (학습 실험)').replace(old,new);path.write_text(text)
    path=ROOT/'docs/model_commands.md';text=path.read_text().replace('3-A (base, untrained)','3-A (trained, experimental)').replace('3-B (base, untrained)','3-B (trained, experimental)').replace('all\nseven CLI/config combinations','all\nnine CLI/config combinations')
    text=text.replace('Endophalon-3 is a base without trained models. Add `--probe` for the untrained','Endophalon-3 training results: [endophalon3.md](endophalon3.md). Add `--probe` for the unchanged untrained')
    path.write_text(text)
    path=ROOT/'GOAL.md';path.write_text(path.read_text()+f'\n## Experiment evidence\n\nCompleted/audited {len(rows)}/9 models. A accuracy gates {passed}/{len(primary)}. See docs/endophalon3_training_report.ko.md; original source and evidence preserved. Infrastructure-interrupted attempts are recorded separately. No master merge.\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['train','finalize']);parser.add_argument('--seed',type=int,choices=[1,2]);args=parser.parse_args()
    if args.mode=='train':train(args.seed)
    else:finalize()
