from __future__ import annotations

import numpy as np


class MultiScaleObserver:
    """Voxel occupancy at several grid resolutions, plus mean speed.

    For each tap it emits, per batch row: the fraction of balls in every voxel
    of every grid, then (optionally) the mean ball speed.
    """

    def __init__(
        self,
        space_size: tuple[float, float, float],
        grids: tuple[tuple[int, int, int], ...],
        *,
        speed: bool = True,
    ) -> None:
        if not grids:
            raise ValueError("observer needs at least one grid")
        self.space_size = np.array(space_size, dtype=np.float64)
        self.grids = tuple(np.array(grid, dtype=np.int64) for grid in grids)
        self.speed = speed

    @property
    def features_per_tap(self) -> int:
        return sum(int(np.prod(grid)) for grid in self.grids) + int(self.speed)

    def observe(self, positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        """``(B, N, 3)`` positions/velocities -> ``(B, features_per_tap)``."""
        rows, count, _ = positions.shape
        parts = [self._occupancy(positions, grid) for grid in self.grids]
        if self.speed:
            parts.append(np.linalg.norm(velocities, axis=2).mean(axis=1, keepdims=True))
        return np.hstack(parts)

    def _occupancy(self, positions: np.ndarray, grid: np.ndarray) -> np.ndarray:
        rows, count, _ = positions.shape
        voxels = int(np.prod(grid))
        cells = np.floor(positions / self.space_size * grid).astype(np.int64)
        cells = np.clip(cells, 0, grid - 1)
        flat = np.ravel_multi_index(np.moveaxis(cells, 2, 0), tuple(grid))
        keys = (np.arange(rows, dtype=np.int64)[:, None] * voxels + flat).ravel()
        counts = np.bincount(keys, minlength=rows * voxels).reshape(rows, voxels)
        return counts / count
