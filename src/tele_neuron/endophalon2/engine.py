from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from tele_neuron.endophalon2.theta import Theta2
from tele_neuron.endophalon_pro.physics import ProBackend, ProBatchState, base_masses
from tele_neuron.plus.observe import MultiScaleObserver
from tele_neuron.plus.world import WorldSpec


@dataclass(frozen=True, slots=True)
class StreamSpec:
    """Timeline of one continuous run.

    The physics warms up for ``warmup`` steps, then pulse k starts at
    ``warmup + k * interval`` and pushes for ``pulse_steps`` steps. The state is
    recorded at every delay 1..interval after each pulse onset, so any readout
    delay n can be chosen afterwards.
    """

    pulses: int
    interval: int
    pulse_steps: int
    warmup: int

    def __post_init__(self) -> None:
        if self.pulses <= 0 or self.interval <= 0 or self.warmup < 0:
            raise ValueError("stream needs positive pulses and interval, and non-negative warmup")
        if not 1 <= self.pulse_steps <= self.interval:
            raise ValueError("pulse_steps must be within [1, interval]")

    @property
    def total_steps(self) -> int:
        return self.warmup + self.pulses * self.interval


@dataclass(frozen=True, slots=True)
class RowSpec:
    """One simulation row: a theta, a pulse sequence, and its noise sources."""

    theta: int
    bits: np.ndarray  # (pulses, inputs) of 0/1
    jitter: int  # realization used for the initial jitter
    noise: int  # key of the heat-bath noise sequence (rows with equal keys share it)


@dataclass(frozen=True, slots=True)
class StreamRecord:
    zones: np.ndarray  # (rows, pulses, interval, zones): soft readout zone occupancy at each delay
    grid: np.ndarray  # (rows, pulses, interval, G): optional fixed-grid occupancy (G may be 0)
    energy: np.ndarray  # (rows, pulses): mean kinetic energy at each pulse onset
    wall: np.ndarray  # (rows, pulses): share of balls pinned to a wall at each pulse onset


@dataclass(slots=True)
class StreamState(ProBatchState):
    home: np.ndarray = None  # type: ignore[assignment]  # (N, 3) anchor targets
    anchor: np.ndarray = None  # type: ignore[assignment]  # (B,) spring stiffness
    bath: np.ndarray = None  # type: ignore[assignment]  # (B,) velocity kick std per step


class StreamBackend(ProBackend):
    """Pro backend plus rest dynamics: anchor springs and a heat bath.

    Both extra terms cost O(N) per step, small next to collision handling.
    """

    def step_stream(self, state: StreamState, world: WorldSpec, kicks: np.ndarray | None) -> None:
        forces = self._input_forces(state, world)
        forces -= state.anchor[:, None, None] * (state.positions - state.home[None])
        state.velocities *= state.damping[:, None, None]
        state.velocities += forces / state.masses[:, :, None] * world.dt
        if kicks is not None:
            state.velocities += state.bath[:, None, None] * kicks
        state.positions += state.velocities * world.dt
        self._reflect(state, world)
        self._collide(state, world)
        self._reflect(state, world)


def soft_zones(positions: np.ndarray, centers: np.ndarray, width: np.ndarray) -> np.ndarray:
    """(B, N, 3) positions, (B, K, 3) centers, (B,) widths -> (B, K) Gaussian-weighted occupancy."""
    delta = positions[:, :, None, :] - centers[:, None, :, :]
    distance_sq = np.einsum("bnkd,bnkd->bnk", delta, delta)
    return np.exp(-distance_sq / (2.0 * width[:, None, None] ** 2)).mean(axis=1)


def run_rows(
    world: WorldSpec,
    thetas: list[Theta2],
    rows: list[RowSpec],
    spec: StreamSpec,
    *,
    grids: tuple[tuple[int, int, int], ...] = (),
    workers: int = 1,
    chunk_size: int = 128,
) -> StreamRecord:
    """Run every row through the whole timeline. Rows never interact, so the
    result does not depend on chunking or worker count."""
    chunks = [rows[start : start + chunk_size] for start in range(0, len(rows), chunk_size)]
    args = [(world, thetas, chunk, spec, grids) for chunk in chunks]
    if workers > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_run_chunk, args))
    else:
        parts = [_run_chunk(item) for item in args]
    return StreamRecord(*(np.concatenate([part[i] for part in parts]) for i in range(4)))


def _build(world: WorldSpec, thetas: list[Theta2], rows: list[RowSpec]) -> StreamState:
    base_positions, directions = world.base_state()
    low = world.ball_radius
    high = np.array(world.space_size) - world.ball_radius
    count, inputs = len(rows), len(thetas[0].stimulus_points)
    positions = np.empty((count, world.ball_count, 3))
    velocities = np.empty_like(positions)
    masses = np.empty((count, world.ball_count))
    mass_cache: dict[int, np.ndarray] = {}
    for index, row in enumerate(rows):
        position_noise, velocity_noise = world.jitter(row.jitter)
        positions[index] = np.clip(base_positions + position_noise, low, high)
        velocities[index] = velocity_noise
        if row.theta not in mass_cache:
            mass_cache[row.theta] = base_masses(world, thetas[row.theta].mass_spread)
        masses[index] = mass_cache[row.theta]

    def per_row(name: str) -> np.ndarray:
        return np.array([getattr(thetas[row.theta], name) for row in rows], dtype=np.float64)

    return StreamState(
        positions=positions,
        velocities=velocities,
        masses=masses,
        damping=per_row("damping"),
        restitution=per_row("restitution"),
        strength=per_row("pulse_strength"),
        bits=np.zeros((count, inputs)),
        points=np.array([thetas[row.theta].stimulus_points for row in rows], dtype=np.float64),
        home=base_positions,
        anchor=per_row("anchor"),
        bath=per_row("bath"),
    )


def _run_chunk(args: tuple) -> tuple[np.ndarray, ...]:
    world, thetas, rows, spec, grids = args
    state = _build(world, thetas, rows)
    backend = StreamBackend()
    count = len(rows)
    sequences = np.stack([row.bits for row in rows]).astype(np.float64)  # (B, P, I)
    if sequences.shape[1] != spec.pulses:
        raise ValueError("every row needs one bit vector per pulse")
    centers = np.array([thetas[row.theta].readout_centers for row in rows], dtype=np.float64)
    widths = np.array([thetas[row.theta].readout_width for row in rows], dtype=np.float64)
    observer = MultiScaleObserver(world.space_size, grids, speed=False) if grids else None

    # One heat-bath generator per noise key: rows sharing a key share the kicks.
    keys = sorted({row.noise for row in rows})
    generators = {key: np.random.default_rng([world.seed, 0x42, key]) for key in keys}
    key_index = np.array([keys.index(row.noise) for row in rows])
    use_bath = bool(np.any(state.bath > 0))

    zones = np.empty((count, spec.pulses, spec.interval, centers.shape[1]))
    grid = np.empty((count, spec.pulses, spec.interval, observer.features_per_tap if observer else 0))
    energy = np.empty((count, spec.pulses))
    wall = np.empty((count, spec.pulses))
    lower = world.ball_radius + 1e-9
    upper = np.array(world.space_size) - world.ball_radius - 1e-9

    for step in range(spec.total_steps):
        offset = step - spec.warmup
        pulse, delay = divmod(offset, spec.interval) if offset >= 0 else (-1, -1)
        if delay == 0:
            energy[:, pulse] = 0.5 * np.mean(state.masses * np.sum(state.velocities**2, axis=2), axis=1)
            pinned = np.any((state.positions <= lower) | (state.positions >= upper), axis=2)
            wall[:, pulse] = pinned.mean(axis=1)
        state.bits = sequences[:, pulse] if pulse >= 0 and delay < spec.pulse_steps else np.zeros_like(state.bits)
        kicks = None
        if use_bath:
            draws = np.stack([generators[key].standard_normal((world.ball_count, 3)) for key in keys])
            kicks = draws[key_index]
        backend.step_stream(state, world, kicks)
        if pulse >= 0:
            # Recorded after the step: delay index d means d + 1 steps after onset.
            zones[:, pulse, delay] = soft_zones(state.positions, centers, widths)
            if observer:
                grid[:, pulse, delay] = observer.observe(state.positions, state.velocities)
    return zones, grid, energy, wall
