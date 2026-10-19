from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tele_neuron.endophalon2.config import load_config
from tele_neuron.endophalon2.engine import RowSpec, StreamSpec, run_rows, soft_zones
from tele_neuron.endophalon2.evaluate import collect, fit_score
from tele_neuron.endophalon2.tasks import history_features, stream_bits, task_targets
from tele_neuron.endophalon2.theta import Theta2, Theta2Space
from tele_neuron.endophalon_pro.physics import ProJob, simulate as pro_simulate
from tele_neuron.endophalon_pro.theta import Theta


ROOT = Path(__file__).resolve().parents[1]


def _config(**stream):
    config = load_config(ROOT / "configs" / "endophalon2" / "base.json")
    world = replace(config.world, ball_count=40)
    spec = replace(config.stream, **({"pulses": 6, "interval": 6, "pulse_steps": 2, "warmup": 4} | stream))
    return replace(
        config, world=world, stream=spec, washout=1, train_streams=4, val_streams=2, test_streams=2, workers=1,
        readout_delay=min(config.readout_delay, spec.interval),
    )


def test_stream_tasks_follow_their_definitions() -> None:
    bits = np.array([[1, 0], [1, 1], [0, 1], [1, 1], [0, 0]])

    xor, valid = task_targets("xor", bits)
    assert xor.tolist() == [1, 0, 1, 0, 0] and valid.all()
    recall, valid = task_targets("recall2", bits)
    assert recall[2:].tolist() == [1, 1, 0] and valid.tolist() == [False, False, True, True, True]
    mem, _ = task_targets("xor_mem", bits)
    assert mem[1:].tolist() == [0, 1, 1, 1]
    parity, _ = task_targets("parity3w", bits)
    assert parity[2:].tolist() == [0, 0, 1]
    assert history_features(bits).shape == (5, 6)
    assert np.array_equal(stream_bits(1, 3, 10), stream_bits(1, 3, 10))


def test_theta2_space_round_trip() -> None:
    config = _config()
    space = config.space
    decoded = space.decode(space.encode(config.theta_init))

    assert decoded.anchor == pytest.approx(config.theta_init.anchor)
    assert np.allclose(decoded.readout_centers, config.theta_init.readout_centers)
    assert space.dim == 7 + 3 * (2 + 4)


def test_without_rest_dynamics_matches_pro_engine() -> None:
    config = _config(pulses=1, interval=10, pulse_steps=10, warmup=0)
    t2 = replace(config.theta_init, anchor=0.0, bath=0.0)
    theta = Theta(t2.damping, t2.restitution, t2.pulse_strength, 0.0, t2.mass_spread, t2.stimulus_points)

    pro = pro_simulate(replace(config.world, steps=10), [theta], [ProJob(0, (1, 1), 5)], 10)
    record = run_rows(config.world, [t2], [RowSpec(0, np.array([[1, 1]]), 5, 5)], config.stream)

    centers = np.array([t2.readout_centers])
    expected = soft_zones(pro.positions, centers, np.array([t2.readout_width]))
    assert np.allclose(record.zones[0, 0, -1], expected[0])


def test_chunks_and_workers_do_not_change_results_with_heat_bath() -> None:
    config = _config()
    theta = replace(config.theta_init, bath=0.01)
    rows = [RowSpec(0, stream_bits(1, s, 6), s, s) for s in range(4)]

    whole = run_rows(config.world, [theta], rows, config.stream, chunk_size=64)
    split = run_rows(config.world, [theta], rows, config.stream, chunk_size=1, workers=2)

    assert np.array_equal(whole.zones, split.zones)
    assert np.array_equal(whole.energy, split.energy)


def test_heat_bath_keeps_a_damped_system_moving() -> None:
    config = _config(pulses=8, pulse_steps=1)
    cold = replace(config.theta_init, damping=0.9, anchor=0.0, bath=0.0, pulse_strength=0.01)
    warm = replace(cold, bath=0.01)
    rows = [RowSpec(0, np.zeros((8, 2), dtype=int), 0, 0), RowSpec(1, np.zeros((8, 2), dtype=int), 0, 0)]

    record = run_rows(config.world, [cold, warm], rows, config.stream)

    assert record.energy[0, -1] < 1e-7
    assert record.energy[1, -1] > 100 * record.energy[0, -1]


def test_anchor_pulls_pushed_balls_back_toward_home() -> None:
    config = _config(pulses=10, interval=10, pulse_steps=1)
    push = np.zeros((10, 2), dtype=int)
    push[0] = 1
    loose = replace(config.theta_init, anchor=0.0, bath=0.0, pulse_strength=0.5, damping=0.95)
    tied = replace(loose, anchor=0.02)
    rows = [RowSpec(0, push, 0, 0), RowSpec(1, push, 0, 0)]

    record = run_rows(config.world, [loose, tied], rows, config.stream, grids=((4, 4, 1),))
    rest = run_rows(config.world, [tied], [RowSpec(0, np.zeros((10, 2), dtype=int), 0, 0)], config.stream, grids=((4, 4, 1),))

    def distance(a, b):
        return np.abs(a - b).sum()

    final_rest = rest.grid[0, -1, -1]
    assert distance(record.grid[1, -1, -1], final_rest) < distance(record.grid[0, -1, -1], final_rest)


def test_reset_mode_runs_each_pulse_alone_and_scores() -> None:
    config = _config()
    streams = range(0, 8)

    continuous = collect(config, [config.theta_init], streams)[0]
    reset = collect(config, [config.theta_init], streams, reset=True)[0]

    assert continuous.features.shape == reset.features.shape == (8, 6, 6, 4)
    assert np.array_equal(continuous.bits, reset.bits)
    accuracy, mse, _ = fit_score(config, continuous, "xor", range(0, 4), range(6, 8), 4)
    assert 0.0 <= accuracy <= 1.0 and mse >= 0.0


def test_stage2_training_never_worsens_j_and_reports_controls(tmp_path) -> None:
    from tele_neuron.endophalon2.model import save_model
    from tele_neuron.endophalon2.train import train

    config = replace(_config(), outer={"generations": 1, "population": 3, "elite": 2, "seed": 1})

    report = train(config, tasks=("xor", "recall1"), random_baseline=True)
    save_model("tiny", "inline", config, report, ("xor", "recall1"), tmp_path / "card.json")

    assert report.fitness.j <= report.fitness_init.j
    assert report.evaluations == 1 + 3
    for key in ("trained/continuous", "initial/continuous", "trained/reset", "history_baseline"):
        assert set(report.test[key]) == set(config.tasks)
    assert set(report.random_search["test"]) == set(config.tasks)
    assert (tmp_path / "card.json").exists()
