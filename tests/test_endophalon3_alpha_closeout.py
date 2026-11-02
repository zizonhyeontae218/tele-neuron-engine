"""Exercise actual closeout with workflow-style relative paths, without retraining."""
from pathlib import Path
import importlib.util
import json
import sys
from types import SimpleNamespace
import hashlib
import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('alpha_closeout',ROOT/'scripts/closeout_endophalon3_alpha_training.py')
closeout=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(closeout)


def test_relative_closeout_paths_hash_outputs_and_preserve_completed_model(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(closeout,'ROOT',tmp_path)
    monkeypatch.setitem(sys.modules,'audit_endophalon3_alpha_training',SimpleNamespace(audit=lambda out:None))
    monkeypatch.setattr(closeout.subprocess,'check_output',lambda *a,**k:'original-source-sha\n')
    out=Path('docs/run');out.mkdir(parents=True)
    for name in ['README.md','README.ko.md','GOAL.md','docs/endophalon3_alpha.md']:(tmp_path/name).write_text('original\n')
    (tmp_path/'models/endophalon3_alpha').mkdir(parents=True)
    (out/'model.json').write_text('{"original":true}\n')
    metric={'accuracy':.75,'levels':{'A':{'accuracy':.8},'B':{'accuracy':.7}}}
    report={key:{'confirmation':metric} for key in ['baseline','metrics','reset_control']};report['confirmation_gate_passed']=False
    (out/'evaluation.json').write_text(json.dumps(report));closeout.closeout(out)
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest and all(not Path(p).is_absolute() for p in manifest)
    assert all(hashlib.sha256((tmp_path/p).read_bytes()).hexdigest()==sha for p,sha in manifest.items())
    saved=tmp_path/'models/endophalon3_alpha/20261102_physics120_seed1_binary.json'
    assert saved.read_bytes()==(out/'model.json').read_bytes()
    original=saved.read_bytes()
    with pytest.raises(FileExistsError):closeout.closeout(out)
    assert saved.read_bytes()==original
