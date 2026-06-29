from __future__ import annotations

import numpy as np


def parse_bits(text: str) -> tuple[int, ...]:
    bits = tuple(int(char) for char in text.strip())
    if not bits or any(bit not in (0, 1) for bit in bits):
        raise ValueError("bits must be a non-empty string of 0 and 1")
    return bits


def input_forces(
    positions: np.ndarray,
    bits: tuple[int, ...],
    input_points: tuple[tuple[float, float, float], ...],
    strength: float,
    epsilon: float,
) -> np.ndarray:
    if len(bits) > len(input_points):
        raise ValueError("more bits were provided than configured input points")

    forces = np.zeros_like(positions, dtype=np.float64)
    for bit, point in zip(bits, input_points, strict=False):
        if bit == 0:
            continue

        origin = np.array(point, dtype=np.float64)
        delta = positions - origin
        distance_sq = np.sum(delta * delta, axis=1)
        distance = np.sqrt(np.maximum(distance_sq, epsilon))
        direction = delta / distance[:, np.newaxis]
        forces += direction * (strength / np.maximum(distance_sq, epsilon))[:, np.newaxis]

    return forces
