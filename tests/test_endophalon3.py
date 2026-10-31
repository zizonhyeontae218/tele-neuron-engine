from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from tele_neuron.endophalon2_pro.engine import EventRow, EventSpec, run_events
from tele_neuron.endophalon2_pro.theta import SpringTheta
from tele_neuron.endophalon3 import variant_name
from tele_neuron.endophalon3.config import load_config
from tele_neuron.endophalon3.core import capacity_probe, collect, dataset, single_readout, train
from tele_neuron.endophalon3.engine import run
from tele_neuron.endophalon3.isa import LEVELS, OPS, Instruction, evaluate, program, random_stream


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "endophalon3"


def _small(name: str = "level_a.json", **changes):
    config = load_config(CONFIGS / name)
    return replace(
        config, lattice=(4, 3, 2), stream=EventSpec(pulses=10, gap_min=6, gap_max=6, pulse_steps=2, warmup=4),
        readout_delay=4, washout=2, train_streams=4, val_streams=2, test_streams=2, workers=1, **changes,
    )


def test_instruction_parse_encode_and_levels() -> None:
    ins = Instruction.parse("A XOR 1 1")
    assert (ins.level, ins.op, ins.a, ins.b) == ("A", "xor", 1, 1)
    assert Instruction.parse("B:and_mem(0,1)") == Instruction("B", "and_mem", 0, 1)
    assert ins.encode("binary").tolist() == [0, 0, 0, 0, 1, 1]
    assert Instruction("B", "and_mem", 0, 1).encode("binary").tolist() == [1, 1, 1, 1, 0, 1]
    assert Instruction("A", "recall2", 0, 0).encode("onehot").tolist() == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError):
        Instruction("A", "and", 1, 1)  # level A has only xor, recall1, recall2, xor_mem
    assert LEVELS["A"] == OPS[:4] and len(LEVELS["B"]) == 8


def test_ground_truth_of_every_op() -> None:
    a = np.array([1, 0, 1, 1])
    b = np.array([1, 1, 0, 0])
    expected = {
        "xor": [0, 1, 1, 1], "recall1": [0, 1, 0, 1], "recall2": [0, 0, 1, 0], "xor_mem": [1, 1, 1, 0],
        "and": [1, 0, 0, 0], "or": [1, 1, 1, 1], "parity3w": [1, 1, 0, 0], "and_mem": [0, 0, 0, 1],
    }
    for name, values in expected.items():
        target, valid = evaluate(np.full(4, OPS.index(name)), a, b)
        assert target.tolist() == values, name
        assert valid.tolist() == [False, False, True, True]
    _, targets = program(["A XOR 1 0", "A recall1 0 0", "A recall2 1 1", "A xor_mem 0 1"])
    assert targets.tolist() == [1, 1, 1, 1]


def test_random_streams_keep_level_and_op_ranges() -> None:
    a_stream = random_stream("A", 1, 0, 200)
    b_stream = random_stream("B", 1, 0, 200)
    joint = random_stream("AB", 1, 0, 200)
    assert set(a_stream.levels) == {0} and set(a_stream.ops) == {0, 1, 2, 3}
    assert set(b_stream.levels) == {1} and set(b_stream.ops) == set(range(8))
    assert set(joint.levels) == {0, 1} and joint.ops[joint.levels == 0].max() <= 3
    assert a_stream.channels("binary").shape == (200, 6) and b_stream.channels("onehot").shape == (200, 11)


def test_all_configs_load_and_name_variants() -> None:
    for path in sorted(CONFIGS.glob("*.json")):
        config = load_config(path)
        assert config.physics.substeps == 2 and config.network.nodes == 144
        assert 0 < config.slow.sum() < config.network.nodes
    assert variant_name("A") == "Teleneuron-Endophalon-3-A"


def test_engine_equals_endophalon2_pro_without_slow_region() -> None:
    config = _small()
    t3 = config.theta_init
    pro = SpringTheta(**{k: getattr(t3, k) for k in SpringTheta.__dataclass_fields__})
    rows = [EventRow(0, random_stream("A", 1, s, 10).channels("binary"), config.stream.onsets(1, s), s, s) for s in range(3)]
    delays = tuple(range(1, config.stream.interval + 1))

    expected = run_events(config.network, config.physics, [pro], rows, config.stream)
    got = run(config.network, config.physics, [t3], rows, config.stream, slow=np.zeros(config.network.nodes, bool), delays=delays)
    only_n = run(config.network, config.physics, [t3], rows, config.stream, slow=np.zeros(config.network.nodes, bool), delays=(4,))

    assert np.array_equal(got.zones, expected.zones)
    assert np.array_equal(only_n.zones[:, :, 0], expected.zones[:, :, 3])
    assert np.array_equal(got.energy, expected.energy)


def test_slow_region_changes_dynamics_and_chunking_is_invariant() -> None:
    config = _small()
    rows = [EventRow(0, random_stream("A", 1, s, 10).channels("binary"), config.stream.onsets(1, s), s, s) for s in range(4)]
    theta = replace(config.theta_init, damping_slow=0.99, stiffness_slow=0.2)
    none = np.zeros(config.network.nodes, bool)

    homogeneous = run(config.network, config.physics, [theta], rows, config.stream, slow=none, delays=(4,))
    a = run(config.network, config.physics, [theta], rows, config.stream, slow=config.slow, delays=(4,))
    b = run(config.network, config.physics, [theta], rows, config.stream, slow=config.slow, delays=(4,), chunk_size=1, workers=2)

    assert config.slow.any()
    assert not np.allclose(a.zones, homogeneous.zones)
    assert np.array_equal(a.zones, b.zones)


@pytest.mark.parametrize("name", ["level_a.json", "level_b_onehot.json"])
def test_probe_single_readout_and_tiny_training(name: str, tmp_path) -> None:
    config = _small(name, outer={"generations": 1, "population": 3, "elite": 2, "seed": 1, "init_sigma": 0.5})

    [data] = collect(config, [config.theta_init], range(0, 6), nodes=True)
    x, y, ops = dataset(config, data, range(0, 4))
    result = single_readout(config, data, range(0, 4), range(4, 6))
    probe = capacity_probe(config, config.theta_init)
    report = train(config, random_baseline=True)

    assert x.shape[1] == config.zones and set(ops.tolist()) <= set(range(len(config.ops)))
    assert set(result["per_op"]) <= set(config.ops)
    assert set(probe["nodes_separate_per_op"]) == set(config.ops)
    assert report.fitness.j <= report.fitness_init.j and report.evaluations == 4
    assert {"single_readout", "single_readout_plus_instruction", "separate_readouts", "linear_nodes"} <= set(report.test["trained"])
