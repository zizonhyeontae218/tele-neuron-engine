from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from tele_neuron.endophalon2_pro.config import load_config, with_events
from tele_neuron.endophalon2_pro.core import collect, drift, train
from tele_neuron.endophalon2_pro.engine import EventRow, run_events, spring_forces
from tele_neuron.endophalon2_pro.network import build_network


ROOT = Path(__file__).resolve().parents[1]


def _config(**events):
    config = load_config(ROOT / "configs" / "endophalon2_pro" / "irregular.json")
    config = with_events(config, **({"pulses": 6, "gap_min": 5, "gap_max": 8, "pulse_steps": 2, "warmup": 4} | events))
    return replace(
        config, lattice=(4, 3, 2), readout_delay=4, washout=1, train_streams=4, val_streams=2, test_streams=2, workers=1
    )


def test_network_is_deterministic_and_connected_both_ways() -> None:
    a = build_network((5, 5, 3), (6, 6, 4), neighbors=6, jitter=0.3, seed=1)
    b = build_network((5, 5, 3), (6, 6, 4), neighbors=6, jitter=0.3, seed=1)

    assert a.nodes == 144
    assert np.array_equal(a.left, b.left) and np.array_equal(a.home, b.home)
    assert np.all(a.left < a.right)
    degree = np.bincount(np.concatenate([a.left, a.right]), minlength=a.nodes)
    assert degree.min() >= 6


def test_spring_forces_match_per_edge_loop_and_sum_to_zero() -> None:
    network = build_network((5, 5, 3), (4, 3, 2), neighbors=4, jitter=0.3, seed=2)
    x = network.home[None] + np.random.default_rng(0).normal(0, 0.2, (3, network.nodes, 3))
    k1, k3 = np.array([0.05, 0.1, 0.2]), np.array([0.5, 0.0, 2.0])

    forces = spring_forces(x, network, k1, k3)

    expected = np.zeros_like(x)
    for e, (i, j) in enumerate(zip(network.left, network.right)):
        d = x[:, j] - x[:, i]
        length = np.linalg.norm(d, axis=1)
        ext = length - network.rest[e]
        m = (k1 * ext + k3 * ext**3) / length
        expected[:, i] += d * m[:, None]
        expected[:, j] -= d * m[:, None]
    assert np.allclose(forces, expected)
    assert np.allclose(forces.sum(axis=1), 0.0)
    assert np.allclose(spring_forces(network.home[None], network, k1[:1], k3[:1]), 0.0)


def test_irregular_onsets_stay_within_gap_bounds_and_are_per_stream() -> None:
    config = _config()
    onsets = config.stream.onsets(1, 3)

    gaps = np.diff(onsets)
    assert onsets[0] == config.stream.warmup
    assert gaps.min() >= 5 and gaps.max() <= 8
    assert not np.array_equal(onsets, config.stream.onsets(1, 4))


def test_results_do_not_depend_on_chunking_or_workers() -> None:
    config = _config()
    theta = replace(config.theta_init, bath=0.01)
    rows = [EventRow(0, np.random.default_rng(s).integers(0, 2, (6, 2)), config.stream.onsets(1, s), s, s) for s in range(4)]

    whole = run_events(config.network, config.physics, [theta], rows, config.stream, chunk_size=64)
    split = run_events(config.network, config.physics, [theta], rows, config.stream, chunk_size=1, workers=2)

    assert np.array_equal(whole.zones, split.zones)
    assert whole.finite.all()


def test_blown_up_theta_is_flagged_not_crashing() -> None:
    config = _config()
    wild = replace(config.theta_init, k_linear=50.0, damping=1.0, pulse_strength=5.0)

    [item] = collect(config, [wild], range(0, 2))

    assert not item.finite
    assert np.isfinite(item.data.features).all()


def test_per_stream_drift_does_not_cancel_across_streams() -> None:
    config = _config(pulses=20)
    [item] = collect(config, [config.theta_init], range(0, 2))
    k = np.arange(20, dtype=np.float64)
    energy = np.stack([np.exp(0.1 * k), np.exp(-0.1 * k)])  # one heats, one cools
    data = replace(item.data, energy=energy)

    assert drift(config, data, range(0, 2)) == pytest.approx(0.1 * (20 - config.washout), rel=1e-6)


def test_training_never_worsens_j_and_reports_controls(tmp_path) -> None:
    from tele_neuron.endophalon2_pro.__main__ import save_model

    config = replace(_config(), outer={"generations": 1, "population": 3, "elite": 2, "seed": 1})

    report = train(config, tasks=("xor", "recall1"), random_baseline=True)
    save_model("tiny", "inline", config, report, ("xor", "recall1"), tmp_path / "card.json")

    assert report.fitness.j <= report.fitness_init.j
    assert report.evaluations == 4
    for key in ("trained/continuous", "initial/continuous", "trained/reset", "history_baseline"):
        assert set(report.test[key]) == set(config.tasks)
    card = json.loads((tmp_path / "card.json").read_text())
    assert card["family"] == "Teleneuron-Endophalon-2-Pro"
