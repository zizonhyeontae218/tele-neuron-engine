"""Owner-authorized tree cleanup: validate every old file before deleting any."""
from pathlib import Path
import hashlib,json,subprocess

PREFIXES=('src/tele_neuron/endophalon3/','src/tele_neuron/endophalon3_alpha/','src/tele_neuron/endophalon3t/','configs/endophalon3/','configs/endophalon3_alpha/','models/endophalon3/','models/endophalon3_alpha/','models/endophalon3_joint/','models/endophalon3_routing/','docs/endophalon3_runs/','docs/endophalon3_alpha_runs/','docs/endophalon3_routing_runs/','docs/endophalon3_joint_runs/')
KEEP_SCRIPTS={'train_endophalon3_independent.py','audit_endophalon3_independent.py','finalize_endophalon3_independent.py','retire_endophalon3_experiments.py'}
KEEP_DOCS={'docs/endophalon3_software.md','docs/endophalon3_development.md','docs/endophalon3_independent_plan.md'}
KEEP_TESTS={'test_endophalon3_software.py','test_endophalon3_independent.py','test_endophalon3_retirement.py'}


def eligible(p):
    if p.startswith(PREFIXES):return True
    if p.startswith('docs/endophalon3_software_runs/') and '/independent120/' not in p:return True
    if p.startswith('docs/endophalon3') and '/' not in p[5:] and p not in KEEP_DOCS:return True
    if p.startswith('scripts/') and 'endophalon3' in Path(p).name and Path(p).name not in KEEP_SCRIPTS:return True
    if p.startswith('tests/test_endophalon3') and Path(p).name not in KEEP_TESTS:return True
    if p.startswith('.github/workflows/endophalon3-') and Path(p).name!='endophalon3-independent.yml':return True
    return p in {'models/endophalon3_software/20261102_existing_cards.json','configs/endophalon3_software/existing_best_observed.json','docs/integration/2026-11-01-endophalon3.md'}


def plan(root):
    paths=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
    return {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths if p and eligible(p)}


def validate(root,expected):
    root=Path(root).resolve()
    for name,sha in expected.items():
        p=root/name
        if not eligible(name) or root not in p.resolve().parents:raise ValueError('Outside approved 3-version scope: '+name)
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('Changed/missing old file; refuse all deletions: '+name)
    current={p for p in subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root).decode().split('\0') if p and eligible(p)}
    if current!=set(expected):raise ValueError('New work or changed retirement scope; refuse deletion')


def retire(root,expected):
    root=Path(root).resolve();validate(root,expected)
    for name in expected:(root/name).unlink()
    print('Removed',len(expected),'obsolete 3-version files from current tree; no history rewrite')
