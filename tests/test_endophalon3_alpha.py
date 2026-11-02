"""Verify persistent state, bank selection and compatibility before model claims."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
from tele_neuron.endophalon.readout import RidgeReadout
from tele_neuron.endophalon3.core import collect
from tele_neuron.endophalon3.isa import Instruction, random_stream
from tele_neuron.endophalon3_alpha.model import Bank,Model
from tele_neuron.endophalon3_alpha.session import Session

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def model():
    return Model.from_cards(ROOT/'models/endophalon3/20261031_a_binary_s1.json',ROOT/'models/endophalon3/20261031_b_binary_s1.json',ROOT)

@pytest.mark.parametrize('level',['A','B'])
def test_fixed_level_matches_legacy_features_energy_and_scores(model,level):
    cfg=replace(model.configs[level],stream=replace(model.config.stream,pulses=8),workers=1)
    [data]=collect(cfg,[cfg.theta_init],range(2));session=Session(model,streams=(0,1));features=[];energy=[]
    inputs=[random_stream(level,cfg.physics.seed,s,8) for s in range(2)]
    from tele_neuron.endophalon3.isa import OPS
    for k in range(8):
        result=session.step([Instruction(level,OPS[int(x.ops[k])],int(x.a[k]),int(x.b[k])) for x in inputs])
        features.append([r.features for r in result]);energy.append([r.kinetic_before for r in result])
        expected=model.banks[level].head.scores(data.features[:,k])[:,0]
        np.testing.assert_allclose([r.score for r in result],expected,rtol=1e-13,atol=1e-13)
        np.testing.assert_array_equal([r.output for r in result],expected>=.5)
    np.testing.assert_array_equal(np.array(features).transpose(1,0,2),data.features)
    np.testing.assert_array_equal(np.array(energy).T,data.energy)


def test_select_and_invalid_commands_do_not_change_state_time_rng(model):
    s=Session(model);s.execute('A xor 1 0');before=s.snapshot();s.execute('level B');after=s.snapshot()
    assert after.pop('selected')=='B';before.pop('selected');assert before==after
    for bad in ['B nope 0 0','A and 0 0','B xor 2 0','level C','garbage']:
        before=s.snapshot()
        with pytest.raises(ValueError):s.execute(bad)
        assert s.snapshot()==before
    s.execute('recall1 0 0');assert s.active.tolist()==[1] and s.events==2


def test_checkpoint_split_program_equivalence_and_model_guard(model):
    a=Session(model);a.execute('A xor 1 0');saved=json.loads(json.dumps(a.snapshot()));b=Session.restore(model,saved)
    for line in ['B recall1 0 0','A recall2 1 0','B parity3w 1 1']:
        ra,rb=a.execute(line),b.execute(line)
        np.testing.assert_array_equal(ra.features,rb.features);assert ra.score==rb.score
    assert a.snapshot()==b.snapshot()
    other=deepcopy(model.to_payload());other['banks']['A']['head']['weights'][0][0]+=1
    with pytest.raises(ValueError):Session.restore(Model.from_payload(other),saved)
    malformed=deepcopy(saved);malformed['clock']+=1
    with pytest.raises(ValueError):Session.restore(model,malformed)


def test_shared_state_carries_different_a_histories_into_same_b_input(model):
    a,b=Session(model),Session(model)
    a.execute('A xor 1 0');b.execute('A xor 0 0')
    ra=a.execute('B recall1 0 0');rb=b.execute('B recall1 0 0')
    assert ra.switched and rb.switched
    assert not np.array_equal(ra.features,rb.features)
    assert not np.array_equal(a.positions,b.positions)
    # This proves physical dependence, not that either learned head is correct.
    assert np.isfinite([ra.switching_energy,rb.switching_energy]).all()


def test_bank_selects_its_own_head_and_no_hidden_runtime_history(model):
    payload=model.to_payload()
    for level,bias in [('A',-2.),('B',2.)]:
        h=payload['banks'][level]['head'];h['weights']=np.zeros((17,1)).tolist();h['weights'][-1][0]=bias
    s=Session(Model.from_payload(payload))
    assert s.execute('A xor 1 0').score==-2.
    assert s.execute('B xor 1 0').score==2.
    assert not {'history','inputs','targets','a1','a2'}&set(s.snapshot())


def test_bundle_roundtrip_rejects_incompatible_banks_and_overwrite(model,tmp_path):
    p=tmp_path/'model.json';model.save(p);restored=Model.load(p)
    assert restored.fingerprint==model.fingerprint
    with pytest.raises(FileExistsError):model.save(p)
    bad=deepcopy(model.to_payload());bad['banks']['B']['config']['physics']['seed']=2
    with pytest.raises(ValueError):Model.from_payload(bad)


def test_energy_jump_is_measured_at_unchanged_position_velocity(model):
    s=Session(model);s.execute('A xor 1 0')
    old=s._parameters(np.array([0]));new=s._parameters(np.array([1]));expected=float((s._energy(new)-s._energy(old))[0])
    x,v=s.positions.copy(),s.velocities.copy();clock=s.clock
    s.select('B');np.testing.assert_array_equal(s.positions,x);np.testing.assert_array_equal(s.velocities,v);assert s.clock==clock
    result=s.execute('xor 0 0');assert result.switching_energy==expected


def test_new_and_legacy_names_are_distinct():
    from tele_neuron.endophalon3_alpha import FAMILY
    from tele_neuron.endophalon3t import variant_name
    assert FAMILY=='Teleneuron-Endophalon-3'
    assert variant_name('AB')=='Teleneuron-Endophalon-3T-AB'


def test_cli_build_program_checkpoint_and_explicit_3t_name(tmp_path):
    import os,subprocess,sys
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'))
    def run(*args):
        return subprocess.run([sys.executable,'-m','tele_neuron.endophalon3_alpha',*map(str,args)],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
    bundle=tmp_path/'bundle.json';run('--config',ROOT/'configs/endophalon3_alpha/from_3t_seed1.json','--build',bundle)
    program=tmp_path/'program.txt';program.write_text('A xor 1 0\nlevel B\nrecall1 0 0\n')
    state=tmp_path/'state.json';result=run('--model',bundle,'--program',program,'--checkpoint-out',state)
    payload=json.loads(result.stdout);assert payload['events']==2 and payload['clock']==64
    assert payload['results'][1]['instruction']['level']=='B'
    program.write_text('A recall2 0 0\n');result=run('--model',bundle,'--program',program,'--checkpoint-in',state)
    assert json.loads(result.stdout)['events']==3
    result=subprocess.run([sys.executable,'-m','tele_neuron.endophalon3t','--help'],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
    assert '--config' in result.stdout


def test_explicit_3t_cli_serializes_new_name_without_rewriting_old_cards(tmp_path):
    import os,subprocess,sys
    raw=json.loads((ROOT/'configs/endophalon3/level_a.json').read_text())
    raw['network']['lattice']=[4,3,2]
    raw['events'].update(pulses=8,gap_min=6,gap_max=6,pulse_steps=2,warmup=4)
    raw['readout']['delay']=4
    raw['split'].update(train_streams=4,val_streams=2,test_streams=2,washout=2)
    raw['outer'].update(generations=0,population=3,elite=2)
    config=tmp_path/'3t.json';config.write_text(json.dumps(raw));out=tmp_path/'3t-card.json'
    subprocess.run([sys.executable,'-m','tele_neuron.endophalon3t','--config',str(config),'--output',str(out)],cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT/'src')),capture_output=True,text=True,check=True)
    assert json.loads(out.read_text())['family']=='Teleneuron-Endophalon-3T-A'
    assert json.loads((ROOT/'models/endophalon3/20261031_a_binary_s1.json').read_text())['family']=='Teleneuron-Endophalon-3-A'


def test_mixed_bank_batch_matches_individual_sessions(model):
    batch=Session(model,streams=(5,8));individual=[Session(model,(s,)) for s in (5,8)]
    # Alternation, then a block, then both switch directions; no reset between chunks.
    for k in range(12):
        levels=('A','B') if k%2 else ('B','A')
        if 4<=k<9:levels=('A','A')
        instructions=[Instruction(l,'recall1',k%2,(k+1)%2) for l in levels]
        batched=batch.step(instructions)
        for row,s in enumerate(individual):
            result=s.step([instructions[row]])[0]
            np.testing.assert_array_equal(result.features,batched[row].features)
            np.testing.assert_array_equal(s.positions[0],batch.positions[row])
            np.testing.assert_array_equal(s.velocities[0],batch.velocities[row])
            assert result.score==batched[row].score
