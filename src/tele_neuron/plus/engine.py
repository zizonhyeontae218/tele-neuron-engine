from __future__ import annotations

from typing import Protocol

import numpy as np

from tele_neuron.plus.batch import BatchState
from tele_neuron.plus.world import WorldSpec


# The own cell plus 13 of the 26 neighbors: every adjacent cell pair is visited
# from exactly one side, so each ball pair is found once.
_HALF_OFFSETS = np.array(
    [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) >= (0, 0, 0)
    ],
    dtype=np.int64,
)


class Backend(Protocol):
    """Advances every row of a batch by one time step."""

    def step(self, state: BatchState, world: WorldSpec) -> np.ndarray:
        """Step in place and return the per-row collision count for this step."""
        ...


class NumpyBackend:
    """Vectorized CPU backend.

    Forces, integration and wall reflection are plain array math over
    ``(B, N, 3)``. Collisions use one global spatial hash whose key includes the
    batch row, so rows can never collide with each other. All contact impulses
    in a step are computed from the same pre-collision state and summed
    (Jacobi style). Phase 1 resolves pairs one after another instead, so the two
    engines agree exactly only when collisions are off.
    """

    def __init__(self, *, collisions: bool = True, epsilon: float = 1e-9) -> None:
        self.collisions = collisions
        self.epsilon = epsilon

    def step(self, state: BatchState, world: WorldSpec) -> np.ndarray:
        forces = self._input_forces(state, world)
        state.velocities *= state.damping[:, None, None]
        state.velocities += forces / state.masses[:, :, None] * world.dt
        state.positions += state.velocities * world.dt
        self._reflect(state, world)
        hits = np.zeros(state.size, dtype=np.int64)
        if self.collisions:
            hits = self._collide(state, world)
            self._reflect(state, world)
        return hits

    def _input_forces(self, state: BatchState, world: WorldSpec) -> np.ndarray:
        forces = np.zeros_like(state.positions)
        eps = world.input_epsilon
        for index, point in enumerate(world.input_points):
            gain = state.strength * state.bits[:, index]
            active = gain != 0
            if not np.any(active):
                continue
            delta = state.positions[active] - np.array(point, dtype=np.float64)
            distance_sq = np.maximum(np.sum(delta * delta, axis=2), eps)
            scale = gain[active, None] / (distance_sq * np.sqrt(distance_sq))
            forces[active] += delta * scale[:, :, None]
        return forces

    @staticmethod
    def _reflect(state: BatchState, world: WorldSpec) -> None:
        lower = world.ball_radius
        upper = np.array(world.space_size, dtype=np.float64) - world.ball_radius
        hit = (state.positions < lower) | (state.positions > upper)
        np.clip(state.positions, lower, upper, out=state.positions)
        state.velocities[hit] *= -1.0

    def _collide(self, state: BatchState, world: WorldSpec) -> np.ndarray:
        rows, count = state.size, world.ball_count
        positions = state.positions.reshape(-1, 3)
        velocities = state.velocities.reshape(-1, 3)
        masses = state.masses.reshape(-1)

        left, right = candidate_pairs(state.positions, world.space_size, world.cell_size)
        if left.size == 0:
            return np.zeros(rows, dtype=np.int64)

        min_distance = 2.0 * world.ball_radius
        delta = positions[right] - positions[left]
        distance_sq = np.einsum("ij,ij->i", delta, delta)
        touching = (distance_sq > self.epsilon) & (distance_sq <= min_distance * min_distance)
        left, right, delta, distance_sq = left[touching], right[touching], delta[touching], distance_sq[touching]
        if left.size == 0:
            return np.zeros(rows, dtype=np.int64)

        distance = np.sqrt(distance_sq)
        normal = delta / distance[:, None]
        mass_left, mass_right = masses[left], masses[right]
        total = mass_left + mass_right

        overlap = (min_distance - distance)[:, None] * normal
        position_shift = np.zeros_like(positions)
        np.add.at(position_shift, left, -overlap * (mass_right / total)[:, None])
        np.add.at(position_shift, right, overlap * (mass_left / total)[:, None])

        impact = np.einsum("ij,ij->i", velocities[left] - velocities[right], normal)
        approaching = impact > 0
        row = left // count
        impulse = np.where(
            approaching,
            (1.0 + state.restitution[row]) * impact / (1.0 / mass_left + 1.0 / mass_right),
            0.0,
        )[:, None] * normal
        velocity_shift = np.zeros_like(velocities)
        np.add.at(velocity_shift, left, -impulse / mass_left[:, None])
        np.add.at(velocity_shift, right, impulse / mass_right[:, None])

        positions += position_shift
        velocities += velocity_shift
        return np.bincount(row[approaching], minlength=rows)


def candidate_pairs(
    positions: np.ndarray,
    space_size: tuple[float, float, float],
    cell_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """All same-row ball pairs in the same or adjacent hash cells.

    ``positions`` is ``(B, N, 3)``. Returned indices are into the flattened
    ``(B * N)`` ball list, with ``left < right`` and each pair listed once.
    """
    rows, count, _ = positions.shape
    dims = np.floor(np.array(space_size) / cell_size).astype(np.int64) + 1
    cells_per_row = int(np.prod(dims))
    cells = np.clip(np.floor(positions.reshape(-1, 3) / cell_size).astype(np.int64), 0, dims - 1)
    row = np.repeat(np.arange(rows, dtype=np.int64), count)
    strides = np.array([dims[1] * dims[2], dims[2], 1], dtype=np.int64)
    keys = row * cells_per_row + cells @ strides

    # Bucket balls by key: balls of cell k are order[cell_start[k]:cell_start[k] + cell_count[k]].
    order = np.argsort(keys, kind="stable")
    cell_count = np.bincount(keys, minlength=rows * cells_per_row)
    cell_start = np.cumsum(cell_count) - cell_count

    lefts: list[np.ndarray] = []
    rights: list[np.ndarray] = []
    for offset in _HALF_OFFSETS:
        own_cell = not offset.any()
        neighbor = cells + offset
        valid = np.all((neighbor >= 0) & (neighbor < dims), axis=1)
        source = np.nonzero(valid)[0]
        neighbor_keys = keys[source] + int(offset @ strides)
        sizes = cell_count[neighbor_keys]
        occupied = sizes > 0
        source, neighbor_keys, sizes = source[occupied], neighbor_keys[occupied], sizes[occupied]
        total = int(sizes.sum())
        if total == 0:
            continue
        first = np.repeat(cell_start[neighbor_keys], sizes)
        within = np.arange(total) - np.repeat(np.cumsum(sizes) - sizes, sizes)
        left = np.repeat(source, sizes)
        right = order[first + within]
        if own_cell:
            keep = left < right
            left, right = left[keep], right[keep]
        lefts.append(np.minimum(left, right))
        rights.append(np.maximum(left, right))
    if not lefts:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(lefts), np.concatenate(rights)
