from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
from tele_neuron.endophalon3.isa import Instruction
from tele_neuron.endophalon3_alpha.session import Session
from tele_neuron.endophalon3_software import SoftwareModel,Router

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def model():
    return SoftwareModel.from_cards(ROOT/'models/endophalon3/20261031_a_binary_s2.json',ROOT/'models/endophalon3/20261031_b_binary_s1.json',ROOT)


def test_interleaving_equals_standalone_local_streams_and_pauses_other_model(model):
    hub=Router(model);local={level:Session(model.parameters) for level in 'AB'}
    for text in ['A xor 1 0','B and 0 1','B recall1 1 0','A recall1 0 0','B parity3w 1 1','A recall2 0 0']:
        level=text[0];other='B' if level=='A' else 'A';before=hub.sessions[other].snapshot()
        actual=hub.execute(text);expected=local[level].execute(text)
        assert actual['score']==expected.score
        np.testing.assert_array_equal(actual['features'],expected.features)
        assert hub.sessions[other].snapshot()==before
    assert hub.sessions['A'].snapshot()==local['A'].snapshot()
    assert hub.sessions['B'].snapshot()==local['B'].snapshot()


def test_register_passes_actual_prediction_even_when_task_answer_is_different(model):
    payload=model.to_payload();head=payload['parameters']['banks']['A']['head'];head['weights']=np.zeros((17,1)).tolist();head['weights'][-1][0]=-1
    hub=Router(SoftwareModel.from_payload(payload))
    # XOR(1,0) is 1; deliberately wrong model emits 0. Wire must pass that 0.
    result=hub.execute('A xor 1 0 -> wire');assert result['output']==0 and hub.registers['wire']==0
    other=hub.execute('B and @wire 1 -> sink')
    assert other['instruction']['a']==0 and other['instruction']['b']==1
    assert hub.registers['sink']==other['output']


def test_invalid_commands_are_atomic_and_selection_advances_no_model(model):
    hub=Router(model);hub.execute('A xor 1 0 -> saved');before=hub.snapshot();hub.execute('level B');after=hub.snapshot()
    assert after.pop('selected')=='B';before.pop('selected');assert before==after
    for line in ['B xor @missing 0','B xor 0 0 -> bad name','A and 0 0','level A -> r','B xor 2 1','B xor 0 0 -> r -> s']:
        before=hub.snapshot()
        with pytest.raises(ValueError):hub.execute(line)
        assert hub.snapshot()==before


def test_checkpoint_continues_both_local_memories_and_registers(model):
    a=Router(model);a.execute('A xor 1 0 -> p');a.execute('B recall1 @p 0 -> q')
    b=Router.restore(model,json.loads(json.dumps(a.snapshot())))
    for line in ['A recall1 @q 1 -> r','B recall2 @r @p -> z']:
        assert a.execute(line)==b.execute(line)
    assert a.snapshot()==b.snapshot()
    bad=deepcopy(a.snapshot());bad['registers']['p']=2
    with pytest.raises(ValueError):Router.restore(model,bad)
    bad=deepcopy(a.snapshot());bad['sessions']['A']['active']=[1]
    with pytest.raises(ValueError):Router.restore(model,bad)


def test_software_bundle_and_cli(tmp_path,model):
    import os,subprocess,sys
    p=tmp_path/'bundle.json';model.save(p);assert SoftwareModel.load(p).fingerprint==model.fingerprint
    with pytest.raises(FileExistsError):model.save(p)
    program=ROOT/'configs/endophalon3_software/example_program.txt';checkpoint=tmp_path/'state.json'
    run=subprocess.run([sys.executable,'-m','tele_neuron.endophalon3_software','--model',str(p),'--program',str(program),'--checkpoint-out',str(checkpoint)],cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT/'src')),capture_output=True,text=True,check=True)
    data=json.loads(run.stdout);assert data['local_events']=={'A':2,'B':2} and data['events']==4
    assert len(data['registers'])==4 and checkpoint.exists()
