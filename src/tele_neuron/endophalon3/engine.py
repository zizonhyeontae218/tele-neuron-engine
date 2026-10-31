"""Endophalon-3 engine: the Endophalon-2-Pro spring integrator, plus a slow region
and readout-only recording.

Changes from ``tele_neuron.endophalon2_pro.engine`` (that module is untouched):

- **Slow region.** Nodes in the slow region use ``damping_slow``; a spring's
  stiffness is scaled by the mean of its two endpoints' multipliers (1 for
  fast nodes, ``stiffness_slow`` for slow ones). With no slow nodes this is
  exactly the Endophalon-2-Pro physics.
- **Readout-only recording.** Soft zones are computed only at the requested
  delays (training needs one), not at every delay of every window. Profiling
  of ProMax16 put soft zones at ~27% of simulation time.
- Substeps stay at 2 (the configured value), as in Endophalon-2-Pro.

The random number use (initial jitter, heat-bath kicks) mirrors
Endophalon-2-Pro, so with no slow region and every delay recorded both engines
give identical results (tested).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from tele_neuron.endophalon2.engine import soft_zones
from tele_neuron.endophalon2_pro.engine import EventRow, EventSpec, PhysicsSpec, SpringScatter, masses_for
from tele_neuron.endophalon2_pro.network import Network
from tele_neuron.endophalon3.theta import Theta3


@dataclass(frozen=True, slots=True)
class Record:
    zones: np.ndarray  # (rows, events, len(delays), zones)
    energy: np.ndarray  # (rows, events) kinetic energy just before each event
    stretch: np.ndarray  # (rows, events) mean relative spring extension just before each event
    finite: np.ndarray  # (rows,)
    nodes: np.ndarray | None  # (rows, events, N, 3) displacement at the first recorded delay, if asked


def slow_mask(network: Network, regions: dict | None) -> np.ndarray:
    """Boolean (N,) mask of slow nodes: home coordinate ``axis`` above ``threshold``."""
    if not regions:
        return np.zeros(network.nodes, dtype=bool)
    return network.home[:, int(regions["axis"])] > float(regions["threshold"])


def run(
    network: Network,
    physics: PhysicsSpec,
    thetas: list[Theta3],
    rows: list[EventRow],
    spec: EventSpec,
    *,
    slow: np.ndarray,
    delays: tuple[int, ...],
    record_nodes: bool = False,
    workers: int = 1,
    chunk_size: int = 96,
) -> Record:
    if not delays or min(delays) < 1 or max(delays) > spec.interval:
        raise ValueError(f"delays must be within [1, {spec.interval}]")
    chunks = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]
    args = [(network, physics, thetas, chunk, spec, slow, tuple(delays), record_nodes) for chunk in chunks]
    if workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_run_chunk, args))
    else:
        parts = [_run_chunk(a) for a in args]
    cat = [np.concatenate([p[i] for p in parts]) for i in range(4)]
    nodes = np.concatenate([p[4] for p in parts]) if record_nodes else None
    return Record(*cat, nodes=nodes)


def _run_chunk(args: tuple) -> tuple[np.ndarray, ...]:
    with np.errstate(all="ignore"):
        return _simulate(*args)


def _simulate(network, physics, thetas, rows, spec, slow, delays, record_nodes):  # noqa: ANN001
    count, nodes = len(rows), network.nodes

    def per_row(name: str) -> np.ndarray:
        return np.array([getattr(thetas[row.theta], name) for row in rows], dtype=np.float64)

    damping, damping_slow = per_row("damping"), per_row("damping_slow")
    k1, k3, k_slow = per_row("k_linear"), per_row("k_cubic"), per_row("stiffness_slow")
    anchor, bath, strength = per_row("anchor"), per_row("bath"), per_row("pulse_strength")
    stim_width, read_width = per_row("stimulus_width"), per_row("readout_width")
    points = np.array([thetas[row.theta].stimulus_points for row in rows], dtype=np.float64)
    centers = np.array([thetas[row.theta].readout_centers for row in rows], dtype=np.float64)
    cache: dict[int, np.ndarray] = {}
    masses = np.stack([cache.setdefault(r.theta, masses_for(network, thetas[r.theta].mass_spread, physics.seed)) for r in rows])
    bits = np.stack([row.bits for row in rows]).astype(np.float64)
    onsets = np.stack([row.onsets for row in rows])

    # Per-node damping and per-edge stiffness multipliers (slow region).
    node_scale = np.where(slow[None, :], k_slow[:, None], 1.0)  # (B, N)
    edge_scale = 0.5 * (node_scale[:, network.left] + node_scale[:, network.right])  # (B, E)
    k1_edge, k3_edge = k1[:, None] * edge_scale, k3[:, None] * edge_scale
    node_damping = np.where(slow[None, :], damping_slow[:, None], damping[:, None])  # (B, N)
    damp_sub = node_damping ** (1.0 / physics.substeps)

    positions = np.empty((count, nodes, 3))
    velocities = np.empty_like(positions)
    for index, row in enumerate(rows):
        rng = np.random.default_rng([physics.seed, 0x4A49, row.jitter])
        positions[index] = network.home + rng.normal(0.0, physics.position_jitter, (nodes, 3))
        velocities[index] = rng.normal(0.0, physics.velocity_jitter, (nodes, 3))
    keys = sorted({row.noise for row in rows})
    generators = {key: np.random.default_rng([physics.seed, 0x42, key]) for key in keys}
    key_index = np.array([keys.index(row.noise) for row in rows])

    scatter = SpringScatter(network, count)
    slot = {delay - 1: i for i, delay in enumerate(delays)}  # delay index after the step -> output slot
    zones = np.zeros((count, spec.pulses, len(delays), centers.shape[1]))
    nodes_out = np.zeros((count, spec.pulses, nodes, 3)) if record_nodes else np.zeros((count, 0))
    energy = np.zeros((count, spec.pulses))
    stretch = np.zeros((count, spec.pulses))
    h = physics.dt / physics.substeps
    total = int(onsets[:, -1].max()) + spec.interval
    all_rows = np.arange(count)

    for step in range(total):
        current = (onsets <= step).sum(axis=1) - 1
        started = current >= 0
        safe = np.maximum(current, 0)
        delay = np.where(started, step - onsets[all_rows, safe], -1)
        at_onset = started & (delay == 0)
        if np.any(at_onset):
            r = all_rows[at_onset]
            energy[r, current[r]] = 0.5 * np.mean(masses[r] * np.sum(velocities[r] ** 2, axis=2), axis=1)
            d = positions[r][:, network.right] - positions[r][:, network.left]
            stretch[r, current[r]] = (np.abs(np.linalg.norm(d, axis=2) - network.rest[None]) / network.rest[None]).mean(axis=1)
        drive = (started & (delay < spec.pulse_steps)).astype(np.float64)[:, None] * bits[all_rows, safe]
        kicks = np.stack([generators[key].standard_normal((nodes, 3)) for key in keys])[key_index]

        for sub in range(physics.substeps):
            forces = _spring_forces(positions, network, k1_edge, k3_edge, scatter)
            forces -= anchor[:, None, None] * (positions - network.home[None])
            if np.any(drive):
                for channel in range(points.shape[1]):
                    gain = strength * drive[:, channel]
                    if not np.any(gain):
                        continue
                    away = positions - points[:, channel][:, None, :]
                    r2 = np.einsum("bnd,bnd->bn", away, away)
                    unit = away / (np.sqrt(r2)[:, :, None] + 1e-9)
                    forces += unit * (gain[:, None] * np.exp(-r2 / (2.0 * stim_width[:, None] ** 2)))[:, :, None]
            velocities *= damp_sub[:, :, None]
            velocities += forces / masses[:, :, None] * h
            if sub == 0:
                velocities += bath[:, None, None] * kicks
            positions += velocities * h

        index = np.where(started, delay, -1)
        for d_index, out in slot.items():
            hit = all_rows[started & (index == d_index)]
            if hit.size:
                zones[hit, current[hit], out] = soft_zones(positions[hit], centers[hit], read_width[hit])
                if record_nodes and out == 0:
                    nodes_out[hit, current[hit]] = positions[hit] - network.home[None]

    finite = np.isfinite(zones).all(axis=(1, 2, 3)) & np.isfinite(positions).all(axis=(1, 2))
    return np.nan_to_num(zones), energy, stretch, finite, np.nan_to_num(nodes_out)


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
