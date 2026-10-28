from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from tele_neuron.endophalon2.engine import soft_zones
from tele_neuron.endophalon2_pro.network import Network
from tele_neuron.endophalon2_pro.theta import SpringTheta


@dataclass(frozen=True, slots=True)
class EventSpec:
    """Input events: after ``warmup`` steps, event k starts ``gap_k`` steps after event k-1.

    Gaps are drawn per stream from [gap_min, gap_max]; equal bounds give a
    regular clock. The state is recorded at every delay 1..``window`` after
    each onset, where ``window = gap_min``, so any readout delay n <= gap_min
    can be chosen afterwards and never runs into the next event.
    """

    pulses: int
    gap_min: int
    gap_max: int
    pulse_steps: int
    warmup: int

    def __post_init__(self) -> None:
        if self.pulses <= 0 or self.warmup < 0:
            raise ValueError("need positive pulses and non-negative warmup")
        if not 1 <= self.gap_min <= self.gap_max:
            raise ValueError("need 1 <= gap_min <= gap_max")
        if not 1 <= self.pulse_steps <= self.gap_min:
            raise ValueError("pulse_steps must be within [1, gap_min]")

    @property
    def interval(self) -> int:
        """Recording window per event (named like Endophalon-2's StreamSpec.interval)."""
        return self.gap_min

    def onsets(self, seed: int, stream: int) -> np.ndarray:
        gaps = np.random.default_rng([seed, 0x4741, stream]).integers(self.gap_min, self.gap_max + 1, self.pulses)
        gaps[0] = 0
        return self.warmup + np.cumsum(gaps)


@dataclass(frozen=True, slots=True)
class PhysicsSpec:
    dt: float
    substeps: int
    position_jitter: float
    velocity_jitter: float
    seed: int


@dataclass(frozen=True, slots=True)
class EventRow:
    theta: int
    bits: np.ndarray  # (pulses, inputs)
    onsets: np.ndarray  # (pulses,) step of each event
    jitter: int  # initial-state jitter realization
    noise: int  # heat-bath noise key (rows sharing a key share kicks)


@dataclass(frozen=True, slots=True)
class EventRecord:
    zones: np.ndarray  # (rows, pulses, window, zones)
    energy: np.ndarray  # (rows, pulses): kinetic energy just before each event
    stretch: np.ndarray  # (rows, pulses): mean |extension| / rest length just before each event
    finite: np.ndarray  # (rows,) False if the integration blew up


def masses_for(network: Network, spread: float, seed: int) -> np.ndarray:
    z = np.random.default_rng([seed, 0x5052]).standard_normal(network.nodes)
    masses = np.exp(spread * z)
    return masses / masses.mean()


class SpringScatter:
    """Precomputed indices to add per-edge forces to their nodes for a batch of B rows."""

    def __init__(self, network: Network, rows: int) -> None:
        base = (np.arange(rows) * network.nodes)[:, None]
        self.left = (base + network.left[None]).ravel()
        self.right = (base + network.right[None]).ravel()
        self.size = rows * network.nodes


def spring_forces(
    positions: np.ndarray,
    network: Network,
    k1: np.ndarray,
    k3: np.ndarray,
    scatter: SpringScatter | None = None,
) -> np.ndarray:
    """(B, N, 3) positions -> (B, N, 3) forces from all springs (linear + cubic)."""
    rows, nodes, _ = positions.shape
    scatter = scatter or SpringScatter(network, rows)
    delta = positions[:, network.right] - positions[:, network.left]  # (B, E, 3)
    length = np.sqrt(np.einsum("bed,bed->be", delta, delta)) + 1e-12
    extension = length - network.rest[None]
    magnitude = k1[:, None] * extension + k3[:, None] * extension**3
    pull = delta * (magnitude / length)[:, :, None]  # force on left, toward right when stretched
    forces = np.empty_like(positions)
    for axis in range(3):
        component = pull[:, :, axis].ravel()
        forces[:, :, axis] = (
            np.bincount(scatter.left, weights=component, minlength=scatter.size)
            - np.bincount(scatter.right, weights=component, minlength=scatter.size)
        ).reshape(rows, nodes)
    return forces


def run_events(
    network: Network,
    physics: PhysicsSpec,
    thetas: list[SpringTheta],
    rows: list[EventRow],
    spec: EventSpec,
    *,
    workers: int = 1,
    chunk_size: int = 128,
) -> EventRecord:
    chunks = [rows[start : start + chunk_size] for start in range(0, len(rows), chunk_size)]
    args = [(network, physics, thetas, chunk, spec) for chunk in chunks]
    if workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_run_chunk, args))
    else:
        parts = [_run_chunk(item) for item in args]
    return EventRecord(*(np.concatenate([part[i] for part in parts]) for i in range(4)))


def _run_chunk(args: tuple) -> tuple[np.ndarray, ...]:
    # A stiff theta can blow up; that row is flagged non-finite instead of warning.
    with np.errstate(all="ignore"):
        return _simulate(*args)


def _simulate(network: Network, physics: PhysicsSpec, thetas: list[SpringTheta], rows: list[EventRow], spec: EventSpec) -> tuple[np.ndarray, ...]:
    count, nodes = len(rows), network.nodes

    def per_row(name: str) -> np.ndarray:
        return np.array([getattr(thetas[row.theta], name) for row in rows], dtype=np.float64)

    damping, k1, k3 = per_row("damping"), per_row("k_linear"), per_row("k_cubic")
    anchor, bath, strength = per_row("anchor"), per_row("bath"), per_row("pulse_strength")
    stim_width, read_width = per_row("stimulus_width"), per_row("readout_width")
    points = np.array([thetas[row.theta].stimulus_points for row in rows], dtype=np.float64)  # (B, I, 3)
    centers = np.array([thetas[row.theta].readout_centers for row in rows], dtype=np.float64)  # (B, K, 3)
    mass_cache: dict[int, np.ndarray] = {}
    masses = np.stack(
        [mass_cache.setdefault(row.theta, masses_for(network, thetas[row.theta].mass_spread, physics.seed)) for row in rows]
    )
    bits = np.stack([row.bits for row in rows]).astype(np.float64)  # (B, P, I)
    onsets = np.stack([row.onsets for row in rows])  # (B, P)

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
    window = spec.interval
    zones = np.zeros((count, spec.pulses, window, centers.shape[1]))
    energy = np.zeros((count, spec.pulses))
    stretch = np.zeros((count, spec.pulses))
    h = physics.dt / physics.substeps
    damp_sub = damping ** (1.0 / physics.substeps)
    total = int(onsets[:, -1].max()) + window
    all_rows = np.arange(count)

    for step in range(total):
        current = (onsets <= step).sum(axis=1) - 1  # (B,) latest event index, -1 before the first
        started = current >= 0
        safe = np.maximum(current, 0)
        delay = np.where(started, step - onsets[all_rows, safe], -1)
        at_onset = started & (delay == 0)
        if np.any(at_onset):
            r = all_rows[at_onset]
            energy[r, current[r]] = 0.5 * np.mean(masses[r] * np.sum(velocities[r] ** 2, axis=2), axis=1)
            d = positions[r][:, network.right] - positions[r][:, network.left]
            ext = np.abs(np.linalg.norm(d, axis=2) - network.rest[None]) / network.rest[None]
            stretch[r, current[r]] = ext.mean(axis=1)
        drive = (started & (delay < spec.pulse_steps)).astype(np.float64)[:, None] * bits[all_rows, safe]  # (B, I)
        kicks = np.stack([generators[key].standard_normal((nodes, 3)) for key in keys])[key_index]

        for sub in range(physics.substeps):
            forces = spring_forces(positions, network, k1, k3, scatter)
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
            velocities *= damp_sub[:, None, None]
            velocities += forces / masses[:, :, None] * h
            if sub == 0:
                velocities += bath[:, None, None] * kicks
            positions += velocities * h

        delay_after = np.where(started, delay, -1)
        recording = started & (delay_after < window)
        if np.any(recording):
            r = all_rows[recording]
            zones[r, current[r], delay_after[r]] = soft_zones(positions[r], centers[r], read_width[r])

    finite = np.isfinite(zones).all(axis=(1, 2, 3)) & np.isfinite(positions).all(axis=(1, 2))
    zones = np.nan_to_num(zones, nan=0.0, posinf=0.0, neginf=0.0)
    return zones, energy, stretch, finite
