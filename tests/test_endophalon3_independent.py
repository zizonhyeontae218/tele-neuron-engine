"""Current independent training/runtime and deterministic continuation contracts."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
from tele_neuron.endophalon3_software.model import Module
from tele_neuron.endophalon3_software.training import new_state,advance,observe,fitness
ROOT=Path(__file__).resolve().parents[1]


def test_optimizer_resume_and_independent_seeds():
    a=json.loads((ROOT/'configs/endophalon3_software/A.json').read_text());b=json.loads((ROOT/'configs/endophalon3_software/B.json').read_text())
    assert a['outer']['seed']==2 and b['outer']['seed']==1
    def evaluate(v):return [{'j':float(np.mean(x*x)),'finite':True} for x in v]
    s=advance(new_state(a),evaluate,4,2);s=advance(s,evaluate,4,2)
    assert advance(s,evaluate,4,2)==advance(json.loads(json.dumps(s)),evaluate,4,2)
    assert len(s['mean'])==77 and s['evaluations']==5


def test_train_only_head_fit():
    m=Module(json.loads((ROOT/'configs/endophalon3_software/A.json').read_text()));d=observe(m,range(4),pulses=12);changed=deepcopy(d);changed['targets'][2:]=1-changed['targets'][2:]
    first,f=fitness(m,d,range(2),range(2,4));second,g=fitness(m,changed,range(2),range(2,4))
    for key in ['mean','scale','weights']:np.testing.assert_array_equal(getattr(first.head,key),getattr(second.head,key))
    assert f['j']!=g['j']


def test_selected_config_cpu_rounding_does_not_allow_model_or_metadata_changes():
    import importlib.util
    spec=importlib.util.spec_from_file_location('independent_audit',ROOT/'scripts/audit_endophalon3_independent.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    source=json.loads((ROOT/'configs/endophalon3_software/A.json').read_text());saved=deepcopy(source)
    saved['theta_init']['stiffness_slow']=float(np.nextafter(saved['theta_init']['stiffness_slow'],np.inf))
    audit.assert_selected_config(source,saved)
    saved['physics']['seed']=2
    with pytest.raises(AssertionError):audit.assert_selected_config(source,saved)
    saved=deepcopy(source);saved['theta_init']['stiffness_slow']+=1e-6
    with pytest.raises(AssertionError):audit.assert_selected_config(source,saved)
