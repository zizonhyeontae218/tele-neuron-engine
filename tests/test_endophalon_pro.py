from pathlib import Path
import json

import numpy as np
import pytest

from tele_neuron.endophalon_pro.config import load_config, parse_config
from tele_neuron.endophalon_pro.model import load_model, model_from_report, save_model
from tele_neuron.endophalon_pro.objective import CaseFeatures, fitness
from tele_neuron.endophalon_pro.physics import ProBackend, ProJob, base_masses, build_state, simulate
from tele_neuron.endophalon_pro.theta import Theta, ThetaSpace
from tele_neuron.endophalon_pro.train import train
from tele_neuron.plus import Job, Preset
from tele_neuron.plus.engine import NumpyBackend
from tele_neuron.plus.runner import simulate as plus_simulate


ROOT = Path(__file__).resolve().parents[1]


def _tiny_payload() -> dict:
    payload = json.loads((ROOT / "configs" / "endophalon_pro" / "full_adder.json").read_text(encoding="utf-8"))
    payload["task"] = "xor"
    payload["theta_init"]["stimulus_points"] = [[1.0, 1.0, 1.5], [4.0, 1.0, 1.5]]
    payload["world"]["balls"]["count"] = 30
    payload["world"]["simulation"]["steps"] = 8
    payload["readout"] = {"step": 8, "grid": [2, 2, 1], "ridge_alpha": 1.0}
    payload["outer"].update({"generations": 2, "population": 4, "elite": 2})
    payload["noise"].update({"train_realizations": 4, "val_realizations": 2, "test_realizations": 2})
    payload["sweep"] = {"grids": [[2, 1, 1], [2, 2, 1]]}
    payload["runtime"] = {"workers": 1, "chunk_size": 64}
    return payload


def test_bundled_configs_parse() -> None:
    for path in sorted((ROOT / "configs" / "endophalon_pro").glob("*.json")):
        config = load_config(path)
        assert config.world.ball_count == 144
        assert len(config.theta_init.stimulus_points) == config.input_width


def test_theta_space_round_trips_and_always_decodes_in_range() -> None:
    space = ThetaSpace((5.0, 5.0, 3.0), inputs=2)
    theta = Theta(0.97, 0.6, 0.2, 0.03, 0.4, ((1.0, 2.0, 1.5), (3.5, 4.0, 1.0)))

    decoded = space.decode(space.encode(theta))
    assert decoded.damping == pytest.approx(theta.damping)
    assert decoded.input_strength == pytest.approx(theta.input_strength)
    assert np.allclose(decoded.stimulus_points, theta.stimulus_points)

    for u in np.random.default_rng(0).normal(0, 10, (50, space.dim)):
        wild = space.decode(u)
        assert 0.8 <= wild.damping <= 1.0 and 0 <= wild.restitution <= 1
        assert 0.01 <= wild.input_strength <= 2.0
        assert np.all(np.array(wild.stimulus_points) >= 0.25)
        assert np.all(np.array(wild.stimulus_points) <= np.array([4.75, 4.75, 2.75]))


def test_pro_backend_matches_plus_when_points_and_masses_agree() -> None:
    config = parse_config(_tiny_payload())
    points = ((1.0, 1.0, 1.5), (4.0, 1.0, 1.5))
    theta = Theta(0.99, 0.9, 0.08, 0.05, 0.0, points)
    from dataclasses import replace

    world = replace(config.world, input_points=points)
    preset = Preset("p", 0.99, 0.9, 0.08, 0.05)
    expected, _ = plus_simulate(world, (preset,), [Job(0, (1, 1), 2)], (8,), NumpyBackend())
    got = simulate(world, [theta], [ProJob(0, (1, 1), 2)], 8)

    assert np.allclose(got.positions[0], expected[0, 0])


def test_rows_with_different_thetas_do_not_interact() -> None:
    config = parse_config(_tiny_payload())
    a = Theta(0.99, 0.9, 0.08, 0.05, 0.0, ((1.0, 1.0, 1.5), (4.0, 1.0, 1.5)))
    b = Theta(0.9, 0.3, 0.5, 0.0, 1.0, ((2.5, 2.5, 1.0), (0.5, 4.5, 2.0)))

    alone = simulate(config.world, [b], [ProJob(0, (0, 1), 1)], 8)
    mixed = simulate(config.world, [a, b], [ProJob(0, (1, 0), 0), ProJob(1, (0, 1), 1)], 8, chunk_size=1, workers=2)

    assert np.array_equal(alone.positions[0], mixed.positions[1])


def test_mass_profile_has_mean_one_and_spread_zero_is_uniform() -> None:
    config = parse_config(_tiny_payload())

    assert np.allclose(base_masses(config.world, 0.0), 1.0)
    spread = base_masses(config.world, 1.0)
    assert spread.mean() == pytest.approx(1.0)
    assert spread.std() > 0.3


def test_fitness_combines_terms_with_lambdas() -> None:
    config = parse_config(_tiny_payload())
    rng = np.random.default_rng(0)
    targets = np.array([[0], [1], [1], [0]])
    features = rng.normal(0.0, 0.1, size=(6, 4, 3)) + targets[None, :, :] * 2.0
    data = CaseFeatures(features, np.full(6, 0.2), targets)

    fit, _ = fitness(config, data, range(0, 4), range(4, 6))

    spec = config.objective
    expected = fit.l_val + spec.lambda_noise * fit.r_noise + spec.lambda_stability * 0.2 + spec.lambda_complexity * fit.r_complexity
    assert fit.j == pytest.approx(expected)
    assert fit.r_stability == pytest.approx(0.2)
    assert fit.val_accuracy == 1.0


def test_train_end_to_end_never_worsens_j_and_card_round_trips(tmp_path: Path) -> None:
    config = parse_config(_tiny_payload())

    report = train(config, random_baseline=True)
    model = model_from_report("tiny", "inline", config, report)
    path = tmp_path / "tiny.json"
    save_model(model, config, path)
    loaded = load_model(path)

    assert report.fitness.j <= report.fitness_init.j
    assert [g.best_j for g in report.history] == sorted([g.best_j for g in report.history], reverse=True)
    assert report.evaluations == 1 + 2 * (config.outer.population)
    assert {"trained", "initial", "bits_linear", "random_search"} <= set(report.test)
    assert [row["features"] for row in report.sweep] == [2, 4]
    assert loaded.theta == report.theta
    assert loaded.predict(config, (1, 0)) == model.predict(config, (1, 0))
