from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import json
from pathlib import Path
from typing import Any

import numpy as np

from tele_neuron.endophalon.config import _require, _triple
from tele_neuron.endophalon2_pro.engine import EventSpec, PhysicsSpec
from tele_neuron.endophalon2_pro.network import Network, build_network
from tele_neuron.endophalon3.engine import slow_mask
from tele_neuron.endophalon3.isa import ENCODINGS, LEVELS, channel_names
from tele_neuron.endophalon3.theta import Theta3, Theta3Space


@dataclass(frozen=True)
class E3Config:
    name: str
    level: str  # "A", "B" (separate training) or "AB" (joint, level drawn per event)
    encoding: str  # "binary" or "onehot"
    regions: dict | None  # {"axis": 1, "threshold": 3.0} -> slow nodes; None -> homogeneous physics
    readout_delay: int
    space_size: tuple[float, float, float]
    lattice: tuple[int, int, int]
    neighbors: int
    lattice_jitter: float
    physics: PhysicsSpec
    stream: EventSpec
    theta_init: Theta3
    ridge_alpha: float
    washout: int
    train_streams: int
    val_streams: int
    test_streams: int
    objective: dict[str, float]
    outer: dict[str, Any]
    workers: int
    chunk_size: int

    @cached_property
    def network(self) -> Network:
        return build_network(self.space_size, self.lattice, neighbors=self.neighbors, jitter=self.lattice_jitter, seed=self.physics.seed)

    @cached_property
    def slow(self) -> np.ndarray:
        return slow_mask(self.network, self.regions)

    @property
    def channels(self) -> tuple[str, ...]:
        return channel_names(self.encoding)

    @property
    def ops(self) -> tuple[str, ...]:
        return LEVELS["B"] if self.level in ("B", "AB") else LEVELS["A"]

    @property
    def zones(self) -> int:
        return len(self.theta_init.readout_centers)

    @property
    def space(self) -> Theta3Space:
        return Theta3Space(self.space_size, len(self.channels), self.zones)

    def stream_ranges(self) -> tuple[range, range, range]:
        a, b = self.train_streams, self.train_streams + self.val_streams
        return range(0, a), range(a, b), range(b, b + self.test_streams)


def load_config(path: str | Path) -> E3Config:
    with Path(path).open("r", encoding="utf-8") as file:
        return parse_config(json.load(file))


def parse_config(raw: dict[str, Any]) -> E3Config:
    _require(raw, "name", "level", "encoding", "world", "network", "physics", "events", "readout", "theta_init", "split")
    net, phys, events, readout, split = raw["network"], raw["physics"], raw["events"], raw["readout"], raw["split"]
    runtime = raw.get("runtime", {})
    config = E3Config(
        name=str(raw["name"]),
        level=str(raw["level"]),
        encoding=str(raw["encoding"]),
        regions=raw.get("regions"),
        readout_delay=int(readout["delay"]),
        space_size=_triple(raw["world"]["space"]["size"], "world.space.size"),
        lattice=tuple(int(v) for v in net["lattice"]),  # type: ignore[arg-type]
        neighbors=int(net["neighbors"]),
        lattice_jitter=float(net["jitter"]),
        physics=PhysicsSpec(
            dt=float(phys["dt"]),
            substeps=max(1, int(phys["substeps"])),
            position_jitter=float(phys.get("position_jitter", 0.05)),
            velocity_jitter=float(phys.get("velocity_jitter", 0.01)),
            seed=int(phys["seed"]),
        ),
        stream=EventSpec(**{k: int(events[k]) for k in ("pulses", "gap_min", "gap_max", "pulse_steps", "warmup")}),
        theta_init=Theta3.from_payload(raw["theta_init"]),
        ridge_alpha=float(readout["ridge_alpha"]),
        washout=int(split["washout"]),
        train_streams=int(split["train_streams"]),
        val_streams=int(split["val_streams"]),
        test_streams=int(split["test_streams"]),
        objective={k: float(v) for k, v in raw.get("objective", {}).items()},
        outer=dict(raw.get("outer", {})),
        workers=max(1, int(runtime.get("workers", 1))),
        chunk_size=max(1, int(runtime.get("chunk_size", 96))),
    )
    if config.level not in ("A", "B", "AB"):
        raise ValueError("level must be 'A', 'B' or 'AB'")
    if config.encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {ENCODINGS}")
    if len(config.theta_init.stimulus_points) != len(config.channels):
        raise ValueError(f"theta_init needs {len(config.channels)} stimulus points for {config.encoding}: {config.channels}")
    if not 1 <= config.readout_delay <= config.stream.gap_min:
        raise ValueError(f"readout.delay must be within [1, {config.stream.gap_min}]")
    if not 2 <= config.washout < config.stream.pulses:
        raise ValueError("split.washout must be >= 2 (recall2 needs two events of history) and leave scored events")
    config.space.decode(config.space.encode(config.theta_init))
    return config
