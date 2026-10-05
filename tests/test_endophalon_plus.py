from pathlib import Path
import json

import numpy as np
import pytest

from tele_neuron.endophalon_plus.readout import WideRidgeReadout
from tele_neuron.endophalon_plus.tasks import task_cases
from tele_neuron.endophalon_plus.config import load_config, parse_config
from tele_neuron.endophalon_plus.gate import LinearGate
from tele_neuron.endophalon_plus.model import load_model, model_from_report, save_model
from tele_neuron.endophalon_plus.train import train


ROOT = Path(__file__).resolve().parents[1]


def _tiny_payload() -> dict:
    payload = json.loads((ROOT / "configs" / "endophalon_plus" / "full_adder.json").read_text(encoding="utf-8"))
    payload["task"] = "xor"
    payload["world"]["space"]["size"] = [4.0, 4.0, 3.0]
    payload["world"]["input"]["points"] = [[0.75, 0.75, 1.5], [3.25, 0.75, 1.5]]
    payload["world"]["balls"]["count"] = 40
    payload["world"]["simulation"]["steps"] = 8
    payload["observer"] = {"grids": [[2, 2, 1], [4, 4, 1]], "taps": [4, 8], "speed": True}
    payload["gate"].update({"restarts": 2, "iterations": 3, "validation_realizations": 2})
    payload["noise"].update({"train_realizations": 6, "test_realizations": 2})
    payload["runtime"] = {"workers": 1, "chunk_size": 64}
    return payload


def test_new_task_truth_tables() -> None:
    adder = dict(task_cases("adder2"))
    mult = dict(task_cases("mult2"))
    parity = dict(task_cases("parity4"))

    assert adder[(1, 1, 1, 1)] == (1, 1, 0)  # 3 + 3 = 6
    assert adder[(0, 1, 1, 0)] == (0, 1, 1)  # 1 + 2 = 3
    assert mult[(1, 1, 1, 1)] == (1, 0, 0, 1)  # 3 * 3 = 9
    assert mult[(1, 0, 1, 1)] == (0, 1, 1, 0)  # 2 * 3 = 6
    assert parity[(1, 0, 1, 1)] == (1,)
    assert len(parity) == 16


def test_bundled_configs_parse() -> None:
    for path in sorted((ROOT / "configs" / "endophalon_plus").glob("*.json")):
        config = load_config(path)
        assert len(config.presets) == 3
        assert config.input_width == len(config.world.input_points)


def test_config_requires_exactly_three_presets() -> None:
    payload = _tiny_payload()
    payload["presets"] = payload["presets"][:2]

    with pytest.raises(ValueError, match="exactly 3 presets"):
        parse_config(payload)


def test_linear_gate_argmax_and_round_trip() -> None:
    gate = LinearGate(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]]))

    assert gate.choose((1, 0)) == 0
    assert gate.choose((0, 1)) == 1
    assert gate.choose((0, 0)) == 2
    assert LinearGate.from_payload(gate.to_payload()).assignment(task_cases("xor")) == gate.assignment(task_cases("xor"))
    assert set(LinearGate.constant(1, 3, 2).assignment(task_cases("xor"))) == {1}


def test_dual_ridge_matches_primal() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(20, 50))
    targets = (rng.normal(size=(20, 2)) > 0).astype(float)

    dual = WideRidgeReadout(alpha=2.0).fit(features, targets)
    primal = WideRidgeReadout(alpha=2.0)
    primal.mean = features.mean(axis=0)
    primal.scale = features.std(axis=0)
    design = np.hstack([(features - primal.mean) / primal.scale, np.ones((20, 1))])
    penalty = 2.0 * np.eye(51)
    penalty[-1, -1] = 0.0
    primal.weights = np.linalg.solve(design.T @ design + penalty, design.T @ targets)

    assert np.allclose(dual.weights, primal.weights)


def test_train_end_to_end_and_model_card_round_trip(tmp_path: Path) -> None:
    config = parse_config(_tiny_payload())

    report = train(config)
    model = model_from_report("tiny", "inline", config, report)
    path = tmp_path / "tiny.json"
    save_model(model, config, path)
    loaded = load_model(path)

    assert len(report.assignment) == 4
    assert {"baseline_linear_on_bits", "gate_only", "fixed_elastic", "fixed_viscous", "fixed_heavy_mix"} <= set(report.controls)
    assert report.controls["baseline_linear_on_bits"]["train"].accuracy == pytest.approx(0.5)
    assert loaded.metrics["test"]["accuracy"] == report.test.accuracy
    assert loaded.predict(config, (0, 1)) == model.predict(config, (0, 1))


def test_fixed_gate_mode_uses_one_preset() -> None:
    payload = _tiny_payload()
    payload["gate"] = {"mode": "fixed", "fixed_preset": 2}

    report = train(parse_config(payload))

    assert set(report.assignment) == {2}
    assert report.gate_evaluations == 0
