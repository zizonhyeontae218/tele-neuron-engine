from __future__ import annotations

import numpy as np

from tele_neuron.endophalon.reservoir import Snapshots


class VoxelObserver:
    """Turns position snapshots into a vector of voxel occupancy fractions."""

    def __init__(self, space_size: tuple[float, float, float], grid: tuple[int, int, int]) -> None:
        self.space_size = np.array(space_size, dtype=np.float64)
        self.grid = np.array(grid, dtype=np.int64)

    @property
    def voxels(self) -> int:
        return int(np.prod(self.grid))

    def features(self, snapshots: Snapshots) -> np.ndarray:
        return np.concatenate([self.occupancy(positions) for positions in snapshots])

    def occupancy(self, positions: np.ndarray) -> np.ndarray:
        cells = np.floor(positions / self.space_size * self.grid).astype(np.int64)
        cells = np.clip(cells, 0, self.grid - 1)
        flat = np.ravel_multi_index(cells.T, tuple(self.grid))
        counts = np.bincount(flat, minlength=self.voxels).astype(np.float64)
        return counts / max(len(positions), 1)
