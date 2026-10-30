from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from tele_neuron.endophalon2_pro.engine import EventSpec
from tele_neuron.endophalon2_promax import variant_name
from tele_neuron.endophalon2_promax.config import load_config
from tele_neuron.endophalon2_promax.core import collect, dataset, separate_readouts, single_readout, train
from tele_neuron.endophalon2_promax.tasks import CHANNELS, OPS, channels, op_stream, op_targets


ROOT = Path(__file__).resolve().parents[1]


def _config(zones: int = 8):
    config = load_config(ROOT / "configs" / "endophalon2_promax" / f"promax{zones}.json")
    return replace(
        config,
        lattice=(4, 3, 2),
        stream=EventSpec(pulses=8, gap_min=6, gap_max=6, pulse_steps=2, warmup=4),
        readout_delay=4,
        washout=1,
        train_streams=4,
        val_streams=2,
        test_streams=2,
        workers=1,
    )


def test_opcode_targets_follow_each_events_own_operation() -> None:
    bits = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    ops = np.array([0, 1, 2, 0])  # xor, recall1, xor_mem, xor

    target, valid = op_targets(bits, ops)

    assert target.tolist() == [1, 1, 1, 0]  # 1^0, a_0=1, 0^1, 0^0
    assert valid.tolist() == [False, True, True, True]
    assert channels(bits, ops).tolist()[2] == [0, 1, 0, 0, 1]
    assert CHANNELS == 5 and OPS == ("xor", "recall1", "xor_mem")
    bits_a, ops_a = op_stream(1, 3, 50)
    assert set(ops_a.tolist()) == {0, 1, 2}
    assert np.array_equal(bits_a, op_stream(1, 3, 50)[0])


def test_variant_names_and_configs() -> None:
    assert variant_name(8) == "Teleneuron-Endophalon-2-ProMax8"
    assert variant_name(16) == "Teleneuron-Endophalon-2-ProMax16"
    assert load_config(ROOT / "configs" / "endophalon2_promax" / "promax8.json").zones == 8
    assert load_config(ROOT / "configs" / "endophalon2_promax" / "promax16.json").zones == 16


def test_single_readout_scores_every_operation_with_one_readout() -> None:
    config = _config()
    [data] = collect(config, [config.theta_init], range(0, 6), nodes=True)

    result = single_readout(config, data, range(0, 4), range(4, 6))
    x, y, ops = dataset(config, data, range(0, 4))
    _, _, ops_plus = dataset(config, data, range(0, 4), features="zones+op")

    assert result["readout"].weights.shape == (config.zones + 1, 1)
    assert set(result["per_op"]) == set(OPS)
    assert x.shape[1] == config.zones and len(y) == len(ops) == len(ops_plus)
    assert dataset(config, data, range(0, 4), features="zones+op")[0].shape[1] == config.zones + 3
    assert dataset(config, data, range(0, 4), features="nodes")[0].shape[1] == config.network.nodes * 3
    assert set(separate_readouts(config, data, range(0, 4), range(4, 6))["per_op"]) == set(OPS)


def test_training_end_to_end_and_model_card(tmp_path) -> None:
    from tele_neuron.endophalon2_promax.__main__ import save_model

    config = replace(_config(), outer={"generations": 1, "population": 3, "elite": 2, "seed": 1, "init_sigma": 0.5})

    report = train(config, random_baseline=True)
    save_model("tiny", "inline", config, report, tmp_path / "card.json")

    assert report.fitness.j <= report.fitness_init.j
    assert report.evaluations == 4
    assert {"single_readout", "single_readout_plus_opcode", "separate_readouts", "linear_nodes"} <= set(report.test["trained"])
    card = json.loads((tmp_path / "card.json").read_text())
    assert card["family"] == "Teleneuron-Endophalon-2-ProMax8"
    assert len(card["theta"]["stimulus_points"]) == 5
