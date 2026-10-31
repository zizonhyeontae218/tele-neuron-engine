from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tele_neuron.plus.world import Job, Preset, WorldSpec


@dataclass(slots=True)
class BatchState:
    """B independent simulations stored as stacked arrays.

    Rows never interact: each row carries its own preset parameters, input bits
    and noise realization, so any mix of jobs can share one batch.
    """

    positions: np.ndarray  # (B, N, 3)
    velocities: np.ndarray  # (B, N, 3)
    masses: np.ndarray  # (B, N)
    damping: np.ndarray  # (B,)
    restitution: np.ndarray  # (B,)
    strength: np.ndarray  # (B,)
    bits: np.ndarray  # (B, I) float, 0/1 per input point

    @property
    def size(self) -> int:
        return int(self.positions.shape[0])


def build_batch(world: WorldSpec, presets: tuple[Preset, ...], jobs: list[Job]) -> BatchState:
    if not jobs:
        raise ValueError("a batch needs at least one job")
    inputs = len(world.input_points)
    base_positions, directions = world.base_state()
    preset_masses = [preset.masses(world) for preset in presets]

    count = len(jobs)
    positions = np.empty((count, world.ball_count, 3), dtype=np.float64)
    velocities = np.empty_like(positions)
    masses = np.empty((count, world.ball_count), dtype=np.float64)
    bits = np.zeros((count, inputs), dtype=np.float64)
    low = world.ball_radius
    high = np.array(world.space_size, dtype=np.float64) - world.ball_radius

    for row, job in enumerate(jobs):
        if not 0 <= job.preset < len(presets):
            raise ValueError(f"job preset index {job.preset} is out of range")
        if len(job.bits) > inputs:
            raise ValueError("more bits were provided than configured input points")
        preset = presets[job.preset]
        position_noise, velocity_noise = world.jitter(job.realization)
        positions[row] = np.clip(base_positions + position_noise, low, high)
        velocities[row] = directions * preset.initial_speed + velocity_noise
        masses[row] = preset_masses[job.preset]
        bits[row, : len(job.bits)] = job.bits

    return BatchState(
        positions=positions,
        velocities=velocities,
        masses=masses,
        damping=np.array([presets[job.preset].damping for job in jobs], dtype=np.float64),
        restitution=np.array([presets[job.preset].restitution for job in jobs], dtype=np.float64),
        strength=np.array([presets[job.preset].input_strength for job in jobs], dtype=np.float64),
        bits=bits,
    )
