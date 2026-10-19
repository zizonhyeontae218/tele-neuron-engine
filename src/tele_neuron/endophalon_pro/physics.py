from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from tele_neuron.endophalon_pro.theta import Theta
from tele_neuron.plus.batch import BatchState
from tele_neuron.plus.engine import NumpyBackend
from tele_neuron.plus.world import WorldSpec


@dataclass(slots=True)
class ProBatchState(BatchState):
    """Plus batch state plus per-row stimulus points (each row has its own theta)."""

    points: np.ndarray = None  # type: ignore[assignment]  # (B, I, 3)


class ProBackend(NumpyBackend):
    """Plus NumPy backend with stimulus points taken per row instead of per world."""

    def _input_forces(self, state: BatchState, world: WorldSpec) -> np.ndarray:
        assert isinstance(state, ProBatchState)
        forces = np.zeros_like(state.positions)
        eps = world.input_epsilon
        for index in range(state.points.shape[1]):
            gain = state.strength * state.bits[:, index]
            active = gain != 0
            if not np.any(active):
                continue
            delta = state.positions[active] - state.points[active, index][:, None, :]
            distance_sq = np.maximum(np.sum(delta * delta, axis=2), eps)
            scale = gain[active, None] / (distance_sq * np.sqrt(distance_sq))
            forces[active] += delta * scale[:, :, None]
        return forces


@dataclass(frozen=True, slots=True)
class ProJob:
    theta: int  # index into the thetas list
    bits: tuple[int, ...]
    realization: int


@dataclass(frozen=True, slots=True)
class Readings:
    """What the physics leaves behind at the readout time, per job."""

    positions: np.ndarray  # (jobs, N, 3)
    velocities: np.ndarray  # (jobs, N, 3)
    wall_fraction: np.ndarray  # (jobs,) share of balls pressed against a wall


def base_masses(world: WorldSpec, spread: float) -> np.ndarray:
    """Log-normal mass profile with mean 1; the shape z is fixed per world seed."""
    z = np.random.default_rng([world.seed, 0x5052]).standard_normal(world.ball_count)
    masses = np.exp(spread * z)
    return masses / masses.mean()


def build_state(world: WorldSpec, thetas: list[Theta], jobs: list[ProJob]) -> ProBatchState:
    base_positions, directions = world.base_state()
    low = world.ball_radius
    high = np.array(world.space_size, dtype=np.float64) - world.ball_radius
    count, inputs = len(jobs), len(thetas[0].stimulus_points)
    positions = np.empty((count, world.ball_count, 3))
    velocities = np.empty_like(positions)
    masses = np.empty((count, world.ball_count))
    bits = np.zeros((count, inputs))
    points = np.empty((count, inputs, 3))
    mass_cache: dict[int, np.ndarray] = {}
    for row, job in enumerate(jobs):
        theta = thetas[job.theta]
        if len(job.bits) != inputs:
            raise ValueError("job bits must match the number of stimulus points")
        position_noise, velocity_noise = world.jitter(job.realization)
        positions[row] = np.clip(base_positions + position_noise, low, high)
        velocities[row] = directions * theta.initial_speed + velocity_noise
        if job.theta not in mass_cache:
            mass_cache[job.theta] = base_masses(world, theta.mass_spread)
        masses[row] = mass_cache[job.theta]
        bits[row] = job.bits
        points[row] = theta.stimulus_points
    return ProBatchState(
        positions=positions,
        velocities=velocities,
        masses=masses,
        damping=np.array([thetas[job.theta].damping for job in jobs]),
        restitution=np.array([thetas[job.theta].restitution for job in jobs]),
        strength=np.array([thetas[job.theta].input_strength for job in jobs]),
        bits=bits,
        points=points,
    )


def simulate(
    world: WorldSpec,
    thetas: list[Theta],
    jobs: list[ProJob],
    readout_step: int,
    *,
    workers: int = 1,
    chunk_size: int = 128,
) -> Readings:
    """Run every job to ``readout_step`` and return the state at that moment.

    Rows are independent, so chunking and worker count never change results.
    """
    if not 1 <= readout_step <= world.steps:
        raise ValueError(f"readout_step must be within [1, {world.steps}]")
    chunks = [jobs[start : start + chunk_size] for start in range(0, len(jobs), chunk_size)]
    args = [(world, thetas, chunk, readout_step) for chunk in chunks]
    if workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_run_chunk, args))
    else:
        parts = [_run_chunk(item) for item in args]
    return Readings(
        positions=np.concatenate([part[0] for part in parts]),
        velocities=np.concatenate([part[1] for part in parts]),
        wall_fraction=np.concatenate([part[2] for part in parts]),
    )


def _run_chunk(args: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world, thetas, jobs, readout_step = args
    state = build_state(world, thetas, jobs)
    backend = ProBackend()
    for _ in range(readout_step):
        backend.step(state, world)
    lower = world.ball_radius + 1e-9
    upper = np.array(world.space_size) - world.ball_radius - 1e-9
    on_wall = np.any((state.positions <= lower) | (state.positions >= upper), axis=2)
    return state.positions, state.velocities, on_wall.mean(axis=1)
