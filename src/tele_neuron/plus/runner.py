from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from tele_neuron.plus.batch import build_batch
from tele_neuron.plus.engine import Backend, NumpyBackend
from tele_neuron.plus.observe import MultiScaleObserver
from tele_neuron.plus.world import Job, Preset, WorldSpec


def simulate(
    world: WorldSpec,
    presets: tuple[Preset, ...],
    jobs: list[Job],
    taps: tuple[int, ...],
    backend: Backend | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one batch and return ``(positions, velocities)`` shaped ``(B, T, N, 3)``."""
    taps = _check_taps(taps, world.steps)
    backend = backend or NumpyBackend()
    state = build_batch(world, presets, jobs)
    positions = np.empty((state.size, len(taps), world.ball_count, 3))
    velocities = np.empty_like(positions)
    slot = {tap: index for index, tap in enumerate(taps)}
    for step in range(1, taps[-1] + 1):
        backend.step(state, world)
        if step in slot:
            positions[:, slot[step]] = state.positions
            velocities[:, slot[step]] = state.velocities
    return positions, velocities


def run_features(
    world: WorldSpec,
    presets: tuple[Preset, ...],
    jobs: list[Job],
    taps: tuple[int, ...],
    observer: MultiScaleObserver,
    *,
    workers: int = 1,
    chunk_size: int = 64,
    collisions: bool = True,
) -> np.ndarray:
    """Simulate every job and return observed features, ``(len(jobs), T * F)``.

    Jobs are split into chunks of ``chunk_size`` rows; chunks run in parallel
    when ``workers > 1``. Each row depends only on its own job, so the result is
    identical for any chunking or worker count.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunks = [jobs[start : start + chunk_size] for start in range(0, len(jobs), chunk_size)]
    args = [(world, presets, chunk, taps, observer, collisions) for chunk in chunks]
    if workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_chunk_features, args))
    else:
        parts = [_chunk_features(item) for item in args]
    return np.vstack(parts)


def _chunk_features(args: tuple) -> np.ndarray:
    world, presets, jobs, taps, observer, collisions = args
    positions, velocities = simulate(world, presets, jobs, taps, NumpyBackend(collisions=collisions))
    return np.hstack(
        [observer.observe(positions[:, tap], velocities[:, tap]) for tap in range(positions.shape[1])]
    )


def _check_taps(taps: tuple[int, ...], steps: int) -> tuple[int, ...]:
    ordered = tuple(sorted(set(int(tap) for tap in taps)))
    if not ordered or ordered[0] < 1 or ordered[-1] > steps:
        raise ValueError(f"taps must be non-empty steps within [1, {steps}]")
    return ordered
