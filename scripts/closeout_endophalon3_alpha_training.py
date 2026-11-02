"""Prepare reviewable trained artifacts/docs; publication guard lives in the workflow."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def closeout(out):
    # The workflow passes a relative path; all repository-relative hashes below
    # must derive from absolute paths, just like ROOT.
    out=out.resolve()
    from audit_endophalon3_alpha_training import audit
    audit(out)
    report=json.loads((out/'evaluation.json').read_text())
    destination=ROOT/'models/endophalon3_alpha/20261102_physics120_seed1_binary.json'
    if destination.exists():raise FileExistsError(destination)
    shutil.copyfile(out/'model.json',destination)
    out_rel=out.resolve().relative_to(ROOT)
    rows=[]
    for label,metric in [('Frozen baseline',report['baseline']['confirmation']),('Trained paired banks',report['metrics']['confirmation']),('Trained heads, reset-at-switch',report['reset_control']['confirmation'])]:
        rows.append('| '+label+' | '+' | '.join(f'{v*100:.2f}%' for v in (metric['accuracy'],metric['levels']['A']['accuracy'],metric['levels']['B']['accuracy']))+' |')
    text='''# Endophalon 3 — 120-generation physics training result

The owner-authorized new-3 paired-bank run completed **120 generations / 2,881 candidate pairs**. A/B parameters and heads remain separate; one production Session carries state through commands. This is not the historical 3T single-head experiment. The selection used training/validation only. [Frozen plan](endophalon3_alpha_120_plan.md).

| Fresh streams 40000–40063 | Overall | A | B |
| --- | --- | --- | --- |
'''+ '\n'.join(rows)+'\n\nFresh confirmation gate (>=90% each level and >=85% every operation): **'+('PASS' if report['confirmation_gate_passed'] else 'FAIL')+'**. A completed budget is distinct from a performance success. All per-operation and first/second-after-switch results, regularizers, failed candidates, switching energy and timing are in `'+str(out_rel)+'/evaluation.json`.\n\n'
    text+='''Both 77-dimensional bank theta vectors were optimized together, while each candidate's two heads were separately fitted on only the corresponding training events. Search seed 1, physical/data seed 1; one binary bank pair, no seed/coefficient search. The balanced-level MSE and existing noise/drift/weight penalties were frozen before training. Drift covers train+validation; switching energy is an external-control change, not a conservation residual. Fresh confirmation was first used after theta selection. The old test was previously observed and is descriptive. Confirmation is now observed and must not be reused as untouched evidence in adaptive follow-ups.

Reset controls change positions/velocities at bank switches with no extra warmup, preserving time/RNG and the same heads. They are state ablations, not separately trained reset models. Finite observations cover 64-event streams only; no infinite-horizon or multi-data-seed generalization claim. This one run cannot establish that a different seed/method/budget could never improve the result.

Artifacts include 121 archived optimizer states, full latest mean/sigma/RNG/best state and candidate metrics, raw log, environment, settings/source/start hashes, selected two-head bundle and held-out observations. The original model/configs/MVP/3T evidence is preserved. The standalone trained bundle is `models/endophalon3_alpha/20261102_physics120_seed1_binary.json`.

Read-only audit: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/audit_endophalon3_alpha_training.py '''+str(out_rel)+'''`. It verifies the budget, checkpoints, source/start fingerprints, train-only head refits, objective and all held-out metrics, plus exact physical replay on six held-out streams. Tests and model smoke are separate execution checks, not accuracy claims.
'''
    doc=ROOT/'docs/endophalon3_alpha_120_report.md'
    if doc.exists():raise FileExistsError(doc)
    doc.write_text(text)
    for name,extra in [('README.md','New 3 completed 120 generations; fresh per-level accuracy and gate outcome: '),('README.ko.md','새 3의 120세대 물리 학습 완료. 새 확인 세트 수준별 정확도와 목표 판정: ')]:
        p=ROOT/name;p.write_text(p.read_text()+'\n'+extra+'[report](docs/endophalon3_alpha_120_report.md).\n')
    p=ROOT/'GOAL.md';p.write_text(p.read_text()+'\n## 120-generation outcome\nRun completed and audited. Confirmation gate '+('PASS' if report['confirmation_gate_passed'] else 'FAIL')+'. See docs/endophalon3_alpha_120_report.md. No replacement search is authorized by this closeout.\n')
    p=ROOT/'docs/endophalon3_alpha.md';p.write_text(p.read_text()+'\n## Subsequent physics training\nThe separate 120-generation run is now complete. See [actual results](endophalon3_alpha_120_report.md); the initial MVP measurements above remain unchanged.\n')
    verification={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'saved_bundle_audit':'PASS','evaluations':2881,'generation_checkpoints':121,'source_and_start_hashes':'PASS','confirmation_gate':report['confirmation_gate_passed']}
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
    files[str(destination.relative_to(ROOT))]=hashlib.sha256(destination.read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(files,indent=2)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args();closeout(args.directory)
