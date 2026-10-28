from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Network:
    """A fixed spring network: node home positions and edges with rest lengths."""

    home: np.ndarray  # (N, 3)
    left: np.ndarray  # (E,) edge endpoints, left < right
    right: np.ndarray  # (E,)
    rest: np.ndarray  # (E,) rest lengths

    @property
    def nodes(self) -> int:
        return int(self.home.shape[0])

    @property
    def edges(self) -> int:
        return int(self.left.shape[0])


def build_network(
    space_size: tuple[float, float, float],
    lattice: tuple[int, int, int],
    *,
    neighbors: int,
    jitter: float,
    seed: int,
) -> Network:
    """Jittered lattice of nodes, each joined to its ``neighbors`` nearest nodes.

    Edges are symmetric (an edge exists if either end picked the other), so
    nodes may have more than ``neighbors`` springs. The layout depends only on
    ``seed``.
    """
    size = np.array(space_size, dtype=np.float64)
    axes = [(np.arange(n) + 0.5) * (s / n) for n, s in zip(lattice, size, strict=True)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    rng = np.random.default_rng([seed, 0x4E45])
    spacing = (size / np.array(lattice)).min()
    home = grid + rng.uniform(-jitter, jitter, grid.shape) * spacing
    home = np.clip(home, 0.0, size)

    distance = np.linalg.norm(home[:, None, :] - home[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    nearest = np.argsort(distance, axis=1)[:, :neighbors]
    pairs = {(min(i, int(j)), max(i, int(j))) for i in range(len(home)) for j in nearest[i]}
    left, right = (np.array(side, dtype=np.int64) for side in zip(*sorted(pairs)))
    return Network(home=home, left=left, right=right, rest=distance[left, right])
