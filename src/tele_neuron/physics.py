from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tele_neuron.collision import resolve_collisions
from tele_neuron.config import ExperimentConfig
from tele_neuron.input_encoder import input_forces
from tele_neuron.space import reflect_bounds


@dataclass(slots=True)
class SimulationState:
    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    radius: float
    collision_count: int = 0


def initialize_state(config: ExperimentConfig) -> SimulationState:
    rng = np.random.default_rng(config.simulation.seed)
    space_size = np.array(config.space.size, dtype=np.float64)
    radius = config.balls.radius

    positions = rng.uniform(
        low=np.full(3, radius),
        high=space_size - radius,
        size=(config.balls.count, 3),
    )
    directions = rng.normal(size=(config.balls.count, 3))
    norms = np.linalg.norm(directions, axis=1)
    directions[norms > 0] /= norms[norms > 0, np.newaxis]
    velocities = directions * config.balls.initial_speed
    masses = np.full(config.balls.count, config.balls.mass, dtype=np.float64)

    return SimulationState(
        positions=positions,
        velocities=velocities,
        masses=masses,
        radius=radius,
    )


def initialize_state_with_masses(
    config: ExperimentConfig,
    masses: np.ndarray,
) -> SimulationState:
    state = initialize_state(config)
    if masses.shape != (config.balls.count,):
        raise ValueError("model masses length must match balls.count")
    if np.any(masses <= 0):
        raise ValueError("model masses must all be positive")
    state.masses = masses.astype(np.float64, copy=True)
    return state


def step_state(
    state: SimulationState,
    config: ExperimentConfig,
    bits: tuple[int, ...],
) -> None:
    forces = input_forces(
        positions=state.positions,
        bits=bits,
        input_points=config.input.points,
        strength=config.input.strength,
        epsilon=config.input.epsilon,
    )
    acceleration = forces / state.masses[:, np.newaxis]
    state.velocities *= config.simulation.damping
    state.velocities += acceleration * config.simulation.dt
    state.positions += state.velocities * config.simulation.dt
    reflect_bounds(
        positions=state.positions,
        velocities=state.velocities,
        space_size=config.space.size,
        radius=state.radius,
    )
    state.collision_count += resolve_collisions(
        positions=state.positions,
        velocities=state.velocities,
        masses=state.masses,
        radius=state.radius,
        cell_size=config.simulation.cell_size,
        restitution=config.simulation.restitution,
    )
    reflect_bounds(
        positions=state.positions,
        velocities=state.velocities,
        space_size=config.space.size,
        radius=state.radius,
    )
