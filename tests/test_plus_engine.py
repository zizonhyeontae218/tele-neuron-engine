from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from tele_neuron.collision import spatial_hash_candidates
from tele_neuron.endophalon.config import load_config as load_endophalon
from tele_neuron.endophalon.reservoir import BallReservoir
from tele_neuron.plus import Job, Preset, WorldSpec
from tele_neuron.plus.batch import build_batch
from tele_neuron.plus.engine import NumpyBackend, candidate_pairs
from tele_neuron.plus.observe import MultiScaleObserver
from tele_neuron.plus.runner import run_features, simulate


ROOT = Path(__file__).resolve().parents[1]


def _small_world(**overrides) -> WorldSpec:
    values = dict(
        space_size=(4.0, 4.0, 3.0),
        ball_count=48,
        ball_radius=0.08,
        input_points=((0.75, 0.75, 1.5), (3.25, 0.75, 1.5)),
        input_epsilon=1e-6,
        steps=10,
        dt=1.0,
        cell_size=0.5,
        seed=2,
        position_jitter=0.1,
        velocity_jitter=0.02,
    )
    values.update(overrides)
    return WorldSpec(**values)


PRESETS = (
    Preset("a", 0.99, 0.9, 0.08, 0.05),
    Preset("b", 0.9, 0.5, 0.2, 0.02, mass_mode="bimodal", mass_low=0.3, mass_high=3.0, heavy_fraction=0.3),
)


def test_matches_endophalon_reservoir_without_collisions() -> None:
    config = load_endophalon(ROOT / "configs" / "endophalon" / "full_adder.json").reservoir
    world = WorldSpec(
        config.space_size, config.ball_count, config.ball_radius, config.input_points, config.input_epsilon,
        config.steps, config.dt, config.cell_size, config.seed, 0.1, 0.02,
    )
    preset = Preset("p", config.damping, config.restitution, config.input_strength, config.initial_speed)
    reservoir = BallReservoir(config, (8, 24), position_jitter=0.1, velocity_jitter=0.02)

    with mock.patch("tele_neuron.endophalon.reservoir.resolve_collisions", return_value=0):
        expected = reservoir.run((1, 0, 1), 3)
    positions, _ = simulate(world, (preset,), [Job(0, (1, 0, 1), 3)], (8, 24), NumpyBackend(collisions=False))

    for tap, snapshot in enumerate(expected):
        assert np.allclose(positions[0, tap], snapshot, atol=1e-10)


def test_candidate_pairs_match_phase1_spatial_hash_per_row() -> None:
    rng = np.random.default_rng(0)
    positions = rng.uniform(0, 1, (3, 120, 3)) * np.array([4.0, 4.0, 3.0])

    left, right = candidate_pairs(positions, (4.0, 4.0, 3.0), 0.5)

    for row in range(3):
        mine = (left // 120 == row)
        got = sorted(zip((left[mine] % 120).tolist(), (right[mine] % 120).tolist()))
        assert got == spatial_hash_candidates(positions[row], 0.5)
    assert np.all(left // 120 == right // 120)


def test_rows_are_independent_of_batch_mates() -> None:
    world = _small_world()
    alone, _ = simulate(world, PRESETS, [Job(1, (1, 1), 4)], (10,))
    mixed, _ = simulate(world, PRESETS, [Job(0, (0, 1), 0), Job(1, (1, 1), 4), Job(0, (1, 0), 9)], (10,))

    assert np.array_equal(alone[0], mixed[1])


def test_workers_and_chunking_do_not_change_features() -> None:
    world = _small_world()
    observer = MultiScaleObserver(world.space_size, ((2, 2, 1), (4, 4, 2)))
    jobs = [Job(index % 2, ((index >> 1) & 1, index & 1), index) for index in range(6)]

    serial = run_features(world, PRESETS, jobs, (5, 10), observer, chunk_size=64)
    parallel = run_features(world, PRESETS, jobs, (5, 10), observer, workers=2, chunk_size=2)

    assert serial.shape == (6, 2 * observer.features_per_tap)
    assert np.array_equal(serial, parallel)


def test_elastic_collision_conserves_momentum_and_separates_overlap() -> None:
    world = _small_world(ball_count=2, position_jitter=0.0, velocity_jitter=0.0)
    state = build_batch(world, (Preset("e", 1.0, 1.0, 0.0, 0.0, mass_low=1.0, mass_high=2.0),), [Job(0, (0, 0), 0)])
    state.positions[0] = [[2.0, 2.0, 1.5], [2.1, 2.0, 1.5]]
    state.velocities[0] = [[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]]
    state.masses[0] = [1.0, 2.0]
    momentum = (state.masses[0, :, None] * state.velocities[0]).sum(axis=0)

    hits = NumpyBackend()._collide(state, world)

    assert hits.tolist() == [1]
    assert np.allclose((state.masses[0, :, None] * state.velocities[0]).sum(axis=0), momentum)
    gap = np.linalg.norm(state.positions[0, 1] - state.positions[0, 0])
    assert gap == pytest.approx(2 * world.ball_radius)


def test_bimodal_preset_masses_are_fixed_and_split() -> None:
    world = _small_world(ball_count=100)
    masses = PRESETS[1].masses(world)

    assert np.array_equal(masses, PRESETS[1].masses(world))
    assert np.count_nonzero(masses == 3.0) == 30
    assert np.count_nonzero(masses == 0.3) == 70


def test_observer_occupancy_sums_to_one_per_grid() -> None:
    observer = MultiScaleObserver((4.0, 4.0, 3.0), ((2, 2, 1), (4, 4, 2)), speed=True)
    rng = np.random.default_rng(1)
    positions = rng.uniform(0, 3, (2, 30, 3))

    features = observer.observe(positions, np.ones_like(positions))

    assert features.shape == (2, 4 + 32 + 1)
    assert np.allclose(features[:, :4].sum(axis=1), 1.0)
    assert np.allclose(features[:, 4:36].sum(axis=1), 1.0)
    assert np.allclose(features[:, -1], np.sqrt(3.0))
