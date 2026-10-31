from pathlib import Path
import json

import numpy as np
import pytest

from tele_neuron.endophalon.config import load_config, parse_config
from tele_neuron.endophalon.model import load_model, model_from_report, save_model
from tele_neuron.endophalon.observer import VoxelObserver
from tele_neuron.endophalon.readout import RidgeReadout
from tele_neuron.endophalon.reservoir import BallReservoir
from tele_neuron.endophalon.tasks import task_cases
from tele_neuron.endophalon.train import train


ROOT = Path(__file__).resolve().parents[1]


def _tiny_payload() -> dict:
    payload = json.loads((ROOT / "configs" / "endophalon" / "xor.json").read_text(encoding="utf-8"))
    payload["reservoir"]["balls"]["count"] = 32
    payload["reservoir"]["simulation"]["steps"] = 12
    payload["observer"]["taps"] = [6, 12]
    payload["noise"]["train_realizations"] = 6
    payload["noise"]["test_realizations"] = 2
    return payload


def test_full_adder_truth_table() -> None:
    cases = dict(task_cases("full_adder"))

    assert cases[(0, 0, 0)] == (0, 0)
    assert cases[(1, 0, 0)] == (1, 0)
    assert cases[(1, 1, 0)] == (0, 1)
    assert cases[(1, 1, 1)] == (1, 1)
    assert len(cases) == 8


def test_bundled_configs_parse() -> None:
    for path in sorted((ROOT / "configs" / "endophalon").glob("*.json")):
        config = load_config(path)
        assert config.input_width == len(config.reservoir.input_points)


def test_config_rejects_input_width_mismatch() -> None:
    payload = _tiny_payload()
    payload["task"] = "parity3"

    with pytest.raises(ValueError, match="input width"):
        parse_config(payload)


def test_observer_occupancy_sums_to_one_per_tap() -> None:
    observer = VoxelObserver((4.0, 4.0, 3.0), (4, 4, 2))
    rng = np.random.default_rng(0)
    snapshots = (rng.uniform(0, 3, (50, 3)), rng.uniform(0, 3, (50, 3)))

    features = observer.features(snapshots)

    assert features.shape == (2 * 32,)
    assert features[:32].sum() == pytest.approx(1.0)
    assert features[32:].sum() == pytest.approx(1.0)


def test_ridge_readout_fits_linear_data_and_round_trips() -> None:
    rng = np.random.default_rng(3)
    features = rng.normal(size=(64, 5))
    targets = np.stack([features[:, 0] > 0, features[:, 1] - features[:, 2] > 0], axis=1).astype(int)

    readout = RidgeReadout(alpha=0.01).fit(features, targets)
    restored = RidgeReadout.from_payload(readout.to_payload())

    assert np.mean(readout.predict(features) == targets) > 0.9
    assert np.array_equal(restored.predict(features), readout.predict(features))


def test_reservoir_is_deterministic_per_realization() -> None:
    config = parse_config(_tiny_payload())
    reservoir = BallReservoir(config.reservoir, config.observer.taps, position_jitter=0.1, velocity_jitter=0.02)

    first = reservoir.run((1, 0), realization=3)
    again = reservoir.run((1, 0), realization=3)
    other = reservoir.run((1, 0), realization=4)

    assert len(first) == 2
    assert all(np.array_equal(a, b) for a, b in zip(first, again))
    assert not np.array_equal(first[-1], other[-1])


def test_train_end_to_end_and_model_card_round_trip(tmp_path: Path) -> None:
    config = parse_config(_tiny_payload())

    report = train(config)
    model = model_from_report("tiny", "inline", config, report)
    path = tmp_path / "tiny.json"
    save_model(model, path)
    loaded = load_model(path)

    assert report.feature_count == 2 * 16
    assert report.baseline_train.accuracy == pytest.approx(0.5)
    assert loaded.metrics["train"]["accuracy"] == report.train.accuracy
    assert loaded.predict(config, (0, 1)) == model.predict(config, (0, 1))
