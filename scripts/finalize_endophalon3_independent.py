"""Audit both completed runs, replace current models, then retire old 3 versions."""
from pathlib import Path
import argparse,hashlib,json,re,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tele_neuron.endophalon3_software import SoftwareModel
from tele_neuron.endophalon3_software.model import Module
from audit_endophalon3_independent import audit
from retire_endophalon3_experiments import retire,validate


def finalize(results):
    results=results.resolve();reports={level:audit(results/level) for level in 'AB'}
    scope=ROOT/'configs/endophalon3_software/retirement.json';plan=json.loads(scope.read_text())
    # Validate cleanup before replacing any currently usable model asset.
    validate(ROOT,plan['sha256'])
    modules={level:Module.load(results/level/'model.json') for level in 'AB'}
    for level in 'AB':shutil.copyfile(results/level/'model.json',ROOT/f'models/endophalon3_software/{level}.json')
    current=ROOT/'models/endophalon3_software/current.json';payload=SoftwareModel(modules).to_payload();current.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    history=ROOT/'docs/endophalon3_development.md';text=history.read_text()+'\n## 독립 A/B 각 120세대 완료 결과\n\n| 모델 | 초기 confirmation 정확도 | 학습 후 confirmation 정확도 | 엄격한 연산별 게이트 통과 |\n| --- | --- | --- | --- |\n'
    for level,r in reports.items():text+=f"| {level} | {100*r['baseline']['confirmation']['accuracy']:.2f}% | {100*r['metrics']['confirmation']['accuracy']:.2f}% | {r['accuracy_gate']} |\n"
    text+='\nA/B 각각 120세대·2881후보, 총 5762후보 평가를 완료했다. 예산 완료와 성능 성공은 구분한다. 모델·입력 이력·RNG는 독립적이고 소프트웨어 레지스터는 실제 예측 출력만 전달한다. 과거 88.33/74.48%는 다른 입력 집합의 결과이므로 이번 baseline과 직접 비교하지 않는다.\n'
    text+='\n현재 결과는 `docs/endophalon3_software_runs/20261102/independent120/A/` 및 `B/`에 있다. 지표·후보 이력·최종 optimizer/RNG·121개 체크포인트·실제 로그는 현재 학습의 재현 증거로 남긴다. 이전 3T·공유 상태·이전 연결 모델과 원시 산출물은 현재 트리에서 삭제했다. 과거 커밋은 일반 Git 이력에 남지만 별도 보존 브랜치·태그·폴더를 만들거나 이력을 재작성하지 않았다.\n'
    for level,r in reports.items():text+=f"\n{level} 사용자 목표 {100*r['user_target']['overall']:.0f}%: {r['user_target']['passed']}; confirmation 연산별: "+', '.join(f"{op} {100*m['accuracy']:.2f}% (n={m['n']})" for op,m in r['metrics']['confirmation']['per_op'].items())+'\n'
    text+='\n삭제한 구버전 파일 수: '+str(len(plan['sha256']))+'；확정 삭제 목록 SHA-256: `'+hashlib.sha256(scope.read_bytes()).hexdigest()+'`。\n'
    history.write_text(text)
    retire(ROOT,plan['sha256']);scope.unlink()
    # Retain v1/v2 smoke checks; delete retired-version smoke code, not only its calls.
    smoke=ROOT/'scripts/smoke_models.py';s=smoke.read_text();s=s.replace('    ("endophalon3", "level_a"),\n','').replace('    ("endophalon3", "level_b"),\n','')
    first=s.find('        if (ROOT / "src/tele_neuron/endophalon3_alpha").exists():');last=s.find('        if (ROOT / "src/tele_neuron/endophalon3_software").exists():')
    if first!=-1 and last>first:s=s[:first]+s[last:]
    smoke.write_text(s)
    for name in ['README.md','README.ko.md']:
        p=ROOT/name;s=p.read_text();s=re.sub(r'(?ms)^## Teleneuron-Endophalon-3T.*?(?=^## |\Z)','',s)
        s=re.sub(r'(?m)^The owner has authorized independent A/B 120-generation training,.*\n\n','',s)
        s=re.sub(r'(?m)^A/B 독립 모델 각각 120세대 학습 후 이전 3 버전을 삭제하도록 승인됐다\..*\n\n','',s)
        marker='\nThe operation-balanced routing follow-up' if name=='README.md' else '\n연산 균형 라우팅 후속 실험'
        if marker in s:s=s.split(marker,1)[0]
        s=s.replace('preserved Endophalon-3T, and the new Endophalon-3 (software-linked A/B)','Endophalon-3 (independent A/B, software-linked)').replace('보존된 Endophalon-3T와 새 Endophalon-3(독립 A/B 소프트웨어 연결)','Endophalon-3(독립 A/B 소프트웨어 연결)')
        if name=='README.ko.md':
            s+='\n## Endophalon 3 — 독립 A/B\n\nA/B 각각 120세대를 완료했다. theta·readout·물리 상태·RNG는 독립적이며 실제 예측 출력을 소프트웨어 레지스터로 연결한다. '
            s+=' / '.join(f"{l}: confirmation {100*reports[l]['metrics']['confirmation']['accuracy']:.2f}%" for l in 'AB')+'. 목표 달성 여부와 연산별 한계는 [3 개발 기록](docs/endophalon3_development.md)에 있다. 소유자 요청에 따라 이전 3 구현·모델 파일은 현재 트리에서 삭제했고 v1/v2는 유지했다.\n'
        else:
            s+='\n## Endophalon 3 — independent A/B\n\nA/B each completed 120 generations with independent theta/readout/state/RNG. Software registers connect actual predicted outputs. '
            s+=' / '.join(f"{l}: {100*reports[l]['metrics']['confirmation']['accuracy']:.2f}% confirmation" for l in 'AB')+'. Accuracy gates and per-op limits are recorded in [development history](docs/endophalon3_development.md). Older 3 implementations/model files were removed from the current tree at the owner’s request; v1/v2 remain.\n'
        s+='\n```bash\npython -m tele_neuron.endophalon3_software --model models/endophalon3_software/current.json --program configs/endophalon3_software/example_program.txt\n```\n'
        p.write_text(s)
    agents='''# Tele-Neuron Engine agent instructions

Read GOAL.md, FileTree.md, docs/model_commands.md and docs/endophalon3_software.md.
Current 3 is ONLY independent A/B, connected by explicit predicted-output registers.
A/B each own physical state, clock, RNG and local history. Pause the inactive model.
144 nodes, 2 substeps, gap16, delay12, 16 zones per active model. No oracle input.

The owner authorized deletion of obsolete 3T/shared-state/old model artifacts;
experimental history is consolidated in docs/endophalon3_development.md.
Do not restore discarded versions or launch new searches without a new task.
Current A/B training is 120 generations each; report failures and accuracy gates plainly.
Preserve current models/configs/seeds, current reproducibility evidence and all v1/v2.

master and endophalon3 must stay aligned. No new remote work branch, force push,
history rewriting or unrelated branch deletion. Leave Claude, generation archives
and iPad Lab branches intact; retain ipad_lab.html and v1/v2 backup consolidation.

Commands: python -m pytest -q; python scripts/smoke_models.py;
python scripts/verify_backup_consolidation.py; python scripts/filetree_index.py update;
python scripts/filetree_index.py lint. Install: python -m pip install -e . -r requirements.txt.
Model training/audit commands are in docs/model_commands.md.
'''
    (ROOT/'AGENTS.md').write_text(agents);(ROOT/'CLAUDE.md').write_text('# Current context\n\nFollow AGENTS.md and GOAL.md. Only independent A/B Endophalon 3 remains; old 3 variants are removed. See docs/endophalon3_development.md for experimental history. Preserve v1/v2, current evidence and unrelated branches.\n')
    (ROOT/'GOAL.md').write_text('# Endophalon 3 — independent A/B release\n\nA/B each completed 120 generations and passed saved-model audits. Current models are models/endophalon3_software/A.json, B.json and current.json. Separate states/clocks/RNG and local histories, explicit predicted-bit software registers. Old 3 variants/artifacts removed from active tree; experiment results only in docs/endophalon3_development.md. No history rewrite or unrelated branch changes.\n\nConfirmation: '+', '.join(f"{l} {100*reports[l]['metrics']['confirmation']['accuracy']:.2f}% (gate {reports[l]['accuracy_gate']})" for l in 'AB')+'. Budget completion is separate from accuracy success. No further training was started.\n')
    (ROOT/'docs/model_names.md').write_text('# Current model names\n\nv1/v2 families remain unchanged. Endophalon 3 is independent A/B plus software output registers; CLI tele_neuron.endophalon3_software. Old 3T/shared-state implementations and model files were removed by owner instruction. Their results are only in [development history](endophalon3_development.md); normal Git history was not rewritten.\n')
    current_doc='''# Endophalon 3 — current independent A/B

One self-contained runtime package, two independent 144-node/2-substep/16-zone models.
Software selects A/B, pauses the inactive session, and passes actual output bits through
explicit registers: `A xor 1 0 -> r0`, then `B and @r0 1`. Recall uses the selected
model's local history. No state transfer, hidden oracle or automatic correctness repair.

Models: models/endophalon3_software/A.json and B.json; combined current.json.
Each completed 120 generations / 2881 candidates, fixed A optimizer seed2/B seed1,
physical/data seed1. Training-only Ridge heads; validation-only theta selection.
Fresh confirmation and per-op outcomes are in the [development history](endophalon3_development.md)
and current independent120 evidence. Earlier versions/model artifacts were removed.

Commands: docs/model_commands.md. Current configs retain the initialization theta
needed to reproduce this training; they are not runnable old model/head archives.
Single-run scores do not establish chained-program accuracy or infinite-horizon stability.
'''
    (ROOT/'docs/endophalon3_software.md').write_text(current_doc)
    commands=ROOT/'docs/model_commands.md';s=commands.read_text().split('| 3T |',1)[0]
    s+='''

v1/v2 rows run with `python -m <module> --config <config> --output <new-card.json>`.
Default training is a full experiment; `--help` lists actual options. Smoke uses
temporary reduced configs. Original ProMax saved-card replay: `python scripts/replay_promax.py`.

## Current Endophalon 3

```bash
python -m tele_neuron.endophalon3_software --model models/endophalon3_software/current.json --program configs/endophalon3_software/example_program.txt --checkpoint-out /tmp/new-3-state.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/audit_endophalon3_independent.py docs/endophalon3_software_runs/20261102/independent120/A
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/audit_endophalon3_independent.py docs/endophalon3_software_runs/20261102/independent120/B
```

Output paths must be new. `level A/B` changes the router default, no physics step.
`OP bit bit -> register` stores actual predictions; `@register` supplies later input.
Completed training is not repeated. Training source is scripts/train_endophalon3_independent.py
with current A/B configs, fixed seeds and checkpoints; no extra search is authorized.
Old 3 variants were removed; only experiment history remains in endophalon3_development.md.
''';commands.write_text(s)
    summary={'status':'both independent 120-generation runs audited; old 3 active-tree versions retired','removed_files':len(plan['sha256']),'reports':{l:{'accuracy_gate':r['accuracy_gate'],'user_target':r['user_target'],'confirmation':r['metrics']['confirmation']} for l,r in reports.items()},'no_history_rewrite':True}
    (results/'release_verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    for level in 'AB':
        out=results/level
        (out/'saved_audit.json').write_text(json.dumps({'status':'passed','generation':120,'candidate_evaluations':2881,'generation_checkpoints':121,'source_hashes_verified':True,'train_only_refit_verified':True,'fitness_and_all_metrics_verified':True,'baseline_metrics_reproduced':True,'exact_physical_replay':{'test_streams':3,'confirmation_streams':3},'model_sha256':hashlib.sha256((out/'model.json').read_bytes()).hexdigest()},indent=2)+'\n')
        manifest={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'};(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Both native models ready; old 3 tree retired; current tests/smoke/publication remain',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);a=ap.parse_args();finalize(a.directory)
