from __future__ import annotations

from typing import Protocol

import numpy as np

from tele_neuron.collision import resolve_collisions
from tele_neuron.endophalon.config import ReservoirConfig
from tele_neuron.input_encoder import input_forces
from tele_neuron.space import reflect_bounds


Snapshots = tuple[np.ndarray, ...]


class Reservoir(Protocol):
    """A fixed dynamical system that turns input bits into observable states."""

    def run(self, bits: tuple[int, ...], realization: int) -> Snapshots:
        """Return particle positions at each requested tap step."""
        ...


class BallReservoir:
    """The Phase 1 ball physics, used as an untrained reservoir.

    Every realization starts from the same seeded base state plus a small,
    deterministic jitter, so a readout must rely on input-driven structure
    rather than on one exact microstate.
    """

    def __init__(
        self,
        config: ReservoirConfig,
        taps: tuple[int, ...],
        *,
        position_jitter: float = 0.0,
        velocity_jitter: float = 0.0,
    ) -> None:
        self.config = config
        self.taps = taps
        self.position_jitter = position_jitter
        self.velocity_jitter = velocity_jitter
        self._base_positions, self._base_velocities = self._base_state()

    def run(self, bits: tuple[int, ...], realization: int) -> Snapshots:
        positions, velocities = self.initial_state(realization)
        masses = np.full(self.config.ball_count, self.config.ball_mass, dtype=np.float64)
        wanted = set(self.taps)
        snapshots: list[np.ndarray] = []
        for step in range(1, max(self.taps) + 1):
            self._step(positions, velocities, masses, bits)
            if step in wanted:
                snapshots.append(positions.copy())
        return tuple(snapshots)

    def initial_state(self, realization: int) -> tuple[np.ndarray, np.ndarray]:
        positions = self._base_positions.copy()
        velocities = self._base_velocities.copy()
        if realization < 0:
            raise ValueError("realization must be non-negative")
        rng = np.random.default_rng([self.config.seed, realization])
        positions += rng.normal(0.0, self.position_jitter, positions.shape)
        velocities += rng.normal(0.0, self.velocity_jitter, velocities.shape)
        radius = self.config.ball_radius
        np.clip(positions, radius, np.array(self.config.space_size) - radius, out=positions)
        return positions, velocities

    def _base_state(self) -> tuple[np.ndarray, np.ndarray]:
        config = self.config
        rng = np.random.default_rng(config.seed)
        radius = config.ball_radius
        positions = rng.uniform(
            low=np.full(3, radius),
            high=np.array(config.space_size, dtype=np.float64) - radius,
            size=(config.ball_count, 3),
        )
        directions = rng.normal(size=(config.ball_count, 3))
        norms = np.linalg.norm(directions, axis=1)
        directions[norms > 0] /= norms[norms > 0, np.newaxis]
        return positions, directions * config.initial_speed

    def _step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        bits: tuple[int, ...],
    ) -> None:
        config = self.config
        forces = input_forces(
            positions=positions,
            bits=bits,
            input_points=config.input_points,
            strength=config.input_strength,
            epsilon=config.input_epsilon,
        )
        velocities *= config.damping
        velocities += forces / masses[:, np.newaxis] * config.dt
        positions += velocities * config.dt
        reflect_bounds(positions, velocities, config.space_size, config.ball_radius)
        resolve_collisions(
            positions=positions,
            velocities=velocities,
            masses=masses,
            radius=config.ball_radius,
            cell_size=config.cell_size,
            restitution=config.restitution,
        )
        reflect_bounds(positions, velocities, config.space_size, config.ball_radius)
