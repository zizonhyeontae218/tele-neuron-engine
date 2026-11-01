"""Joint scoring must not let one level or a rare operation hide failure."""
import importlib.util
from pathlib import Path
import numpy as np
from types import SimpleNamespace

spec=importlib.util.spec_from_file_location('joint',Path(__file__).resolve().parents[1]/'scripts/train_endophalon3_joint.py')
joint=importlib.util.module_from_spec(spec);spec.loader.exec_module(joint)


def test_gate_checks_both_levels_and_each_operation():
    result={'levels':{'A':{'accuracy':.95,'per_op':{'recall2':.84}},'B':{'accuracy':.99,'per_op':{'parity3w':1.}}}}
    assert not joint.passed(result)
    result['levels']['A']['per_op']['recall2']=.85
    assert joint.passed(result)
    result['levels']['B']['accuracy']=.89
    assert not joint.passed(result)


def test_grouped_scoring_uses_event_levels_and_fails_missing_operations(monkeypatch):
    cfg=SimpleNamespace(washout=2)
    data=SimpleNamespace(levels=np.array([[0,1,0,1,0,1]]),index=lambda _:np.array([0]))
    # A events are all correct; B events are all wrong; pooled=.5 hides both.
    monkeypatch.setattr(joint,'dataset',lambda *_:(np.zeros((4,16)),np.array([[1],[0],[1],[0]]),np.array([0,0,1,1])))
    head=SimpleNamespace(scores=lambda _:np.ones((4,1)))
    result=joint.grouped(cfg,data,range(1),head)
    assert result['accuracy']==.5
    assert result['levels']['A']['accuracy']==1.
    assert result['levels']['B']['accuracy']==0.
    assert result['levels']['A']['per_op']['recall2']==0.
    assert not joint.passed(result)


def test_atomic_checkpoint_roundtrip_preserves_rng(tmp_path):
    rng=np.random.default_rng(2);rng.normal(size=77)
    path=tmp_path/'checkpoint.json';joint.atomic(path,{'rng':rng.bit_generator.state})
    import json
    clone=np.random.default_rng();clone.bit_generator.state=json.loads(path.read_text())['rng']
    assert np.array_equal(rng.normal(size=(24,77)),clone.normal(size=(24,77)))
    assert not path.with_suffix('.tmp').exists()
