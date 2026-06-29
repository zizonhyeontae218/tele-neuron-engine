from __future__ import annotations

from collections import defaultdict
from itertools import product

import numpy as np


Cell = tuple[int, int, int]


def spatial_hash_candidates(positions: np.ndarray, cell_size: float) -> list[tuple[int, int]]:
    cells: dict[Cell, list[int]] = defaultdict(list)
    cell_coords = np.floor(positions / cell_size).astype(np.int64)

    for index, coord in enumerate(cell_coords):
        cells[(int(coord[0]), int(coord[1]), int(coord[2]))].append(index)

    pairs: set[tuple[int, int]] = set()
    for cell, indexes in cells.items():
        for offset in product((-1, 0, 1), repeat=3):
            neighbor = (cell[0] + offset[0], cell[1] + offset[1], cell[2] + offset[2])
            neighbor_indexes = cells.get(neighbor)
            if not neighbor_indexes:
                continue

            for left in indexes:
                for right in neighbor_indexes:
                    if left < right:
                        pairs.add((left, right))

    return sorted(pairs)


def resolve_collisions(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    radius: float,
    cell_size: float,
    restitution: float,
    epsilon: float = 1e-9,
) -> int:
    collision_count = 0
    min_distance = radius * 2.0
    min_distance_sq = min_distance * min_distance

    for left, right in spatial_hash_candidates(positions, cell_size):
        delta = positions[right] - positions[left]
        distance_sq = float(np.dot(delta, delta))
        if distance_sq <= epsilon or distance_sq > min_distance_sq:
            continue

        distance = float(np.sqrt(distance_sq))
        normal = delta / distance
        relative_velocity = velocities[left] - velocities[right]
        impact = float(np.dot(relative_velocity, normal))

        overlap = min_distance - distance
        if overlap > 0:
            total_mass = masses[left] + masses[right]
            positions[left] -= normal * overlap * (masses[right] / total_mass)
            positions[right] += normal * overlap * (masses[left] / total_mass)

        if impact <= 0:
            continue

        impulse = (1.0 + restitution) * impact / ((1.0 / masses[left]) + (1.0 / masses[right]))
        velocities[left] -= normal * impulse / masses[left]
        velocities[right] += normal * impulse / masses[right]
        collision_count += 1

    return collision_count
