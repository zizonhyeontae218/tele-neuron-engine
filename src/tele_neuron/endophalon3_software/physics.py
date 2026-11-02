"""Shared spring-force primitives for the current independent 3 runtime."""
import numpy as np
from tele_neuron.endophalon2_pro.network import Network

def slow_mask(network: Network, regions: dict | None) -> np.ndarray:
    """Boolean (N,) mask of slow nodes: home coordinate ``axis`` above ``threshold``."""
    if not regions:
        return np.zeros(network.nodes, dtype=bool)
    return network.home[:, int(regions["axis"])] > float(regions["threshold"])

def _spring_forces(positions, network, k1_edge, k3_edge, scatter):  # noqa: ANN001
    """Endophalon-2-Pro spring forces with per-row, per-edge stiffness (B, E)."""
    rows, nodes, _ = positions.shape
    delta = positions[:, network.right] - positions[:, network.left]
    length = np.sqrt(np.einsum("bed,bed->be", delta, delta)) + 1e-12
    extension = length - network.rest[None]
    pull = delta * ((k1_edge * extension + k3_edge * extension**3) / length)[:, :, None]
    forces = np.empty_like(positions)
    for axis in range(3):
        component = pull[:, :, axis].ravel()
        forces[:, :, axis] = (
            np.bincount(scatter.left, weights=component, minlength=scatter.size)
            - np.bincount(scatter.right, weights=component, minlength=scatter.size)
        ).reshape(rows, nodes)
    return forces
