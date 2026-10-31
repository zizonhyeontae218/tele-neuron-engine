from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from tele_neuron.endophalon.readout import RidgeReadout
from tele_neuron.endophalon2.config import E2Config
from tele_neuron.endophalon2.engine import RowSpec, StreamSpec, run_rows
from tele_neuron.endophalon2.tasks import history_features, stream_bits, task_targets
from tele_neuron.endophalon2.theta import Theta2


@dataclass(frozen=True, slots=True)
class StreamData:
    """Recorded features of one theta, laid out per stream and pulse."""

    features: np.ndarray  # (streams, pulses, interval, F)
    bits: np.ndarray  # (streams, pulses, 2)
    energy: np.ndarray  # (streams, pulses)
    wall: np.ndarray  # (streams, pulses)
    streams: tuple[int, ...]  # stream ids, row order

    def index(self, streams: range) -> np.ndarray:
        position = {stream: i for i, stream in enumerate(self.streams)}
        return np.array([position[s] for s in streams], dtype=np.int64)


def collect(
    config: E2Config,
    thetas: list[Theta2],
    streams: range,
    *,
    reset: bool = False,
    grids: tuple[tuple[int, int, int], ...] = (),
) -> list[StreamData]:
    """Simulate every theta on the given streams.

    ``reset=True`` is the reset baseline: every pulse runs alone from a fresh,
    warmed-up start, as in earlier versions, instead of on top of the running
    state.
    """
    spec, pulses = config.stream, config.stream.pulses
    bits = {s: stream_bits(config.world.seed, s, pulses) for s in streams}
    rows: list[RowSpec] = []
    for index in range(len(thetas)):
        for s in streams:
            if reset:
                rows += [RowSpec(index, bits[s][k : k + 1], 1_000_000 + s * pulses + k, 1_000_000 + s * pulses + k) for k in range(pulses)]
            else:
                rows.append(RowSpec(index, bits[s], s, s))
    run_spec = replace(spec, pulses=1) if reset else spec
    record = run_rows(config.world, thetas, rows, run_spec, grids=grids, workers=config.workers, chunk_size=config.chunk_size)
    features = record.grid if grids else record.zones
    per_theta = len(streams) * (pulses if reset else 1)
    out = []
    for index in range(len(thetas)):
        part = slice(index * per_theta, (index + 1) * per_theta)
        out.append(
            StreamData(
                features=features[part].reshape(len(streams), pulses, spec.interval, -1),
                bits=np.stack([bits[s] for s in streams]),
                energy=record.energy[part].reshape(len(streams), pulses),
                wall=record.wall[part].reshape(len(streams), pulses),
                streams=tuple(streams),
            )
        )
    return out


def dataset(
    config: E2Config,
    data: StreamData,
    task: str,
    streams: range,
    delay: int,
    pulses: slice | None = None,
    *,
    history: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Features at readout delay ``delay`` (or bit-history features) and targets, washout removed."""
    rows = data.index(streams)
    window = pulses or slice(config.washout, config.stream.pulses)
    xs, ys = [], []
    for row in rows:
        target, valid = task_targets(task, data.bits[row])
        feats = history_features(data.bits[row]) if history else data.features[row, :, delay - 1]
        keep = np.zeros(config.stream.pulses, dtype=bool)
        keep[window] = True
        keep &= valid
        xs.append(feats[keep])
        ys.append(target[keep])
    return np.vstack(xs), np.concatenate(ys)[:, None]


def fit_score(
    config: E2Config,
    data: StreamData,
    task: str,
    fit: range,
    score: range,
    delay: int,
    *,
    history: bool = False,
    score_pulses: slice | None = None,
) -> tuple[float, float, RidgeReadout]:
    """Inner loop for one task: exact ridge readout on ``fit`` streams; accuracy and MSE on ``score`` streams."""
    x, y = dataset(config, data, task, fit, delay, history=history)
    readout = RidgeReadout(alpha=config.ridge_alpha).fit(x, y)
    sx, sy = dataset(config, data, task, score, delay, score_pulses, history=history)
    raw = readout.scores(sx)
    return float(np.mean((raw >= 0.5) == sy)), float(np.mean((raw - sy) ** 2)), readout
