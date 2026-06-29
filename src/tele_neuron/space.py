from __future__ import annotations

import numpy as np


def reflect_bounds(
    positions: np.ndarray,
    velocities: np.ndarray,
    space_size: tuple[float, float, float],
    radius: float,
) -> None:
    lower = np.full(3, radius, dtype=np.float64)
    upper = np.array(space_size, dtype=np.float64) - radius

    for axis in range(3):
        low_hits = positions[:, axis] < lower[axis]
        high_hits = positions[:, axis] > upper[axis]

        positions[low_hits, axis] = lower[axis]
        positions[high_hits, axis] = upper[axis]
        velocities[low_hits | high_hits, axis] *= -1.0
