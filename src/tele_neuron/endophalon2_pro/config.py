from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
import json
from pathlib import Path
from typing import Any

from tele_neuron.endophalon.config import _require, _triple
from tele_neuron.endophalon2.tasks import STREAM_TASKS
from tele_neuron.endophalon2_pro.engine import EventSpec, PhysicsSpec
from tele_neuron.endophalon2_pro.network import Network, build_network
from tele_neuron.endophalon2_pro.theta import SpringTheta, SpringThetaSpace


INPUTS = 2


@dataclass(frozen=True)
class E2PConfig:
    name: str
    tasks: tuple[str, ...]
    readout_delay: int
    space_size: tuple[float, float, float]
    lattice: tuple[int, int, int]
    neighbors: int
    lattice_jitter: float
    physics: PhysicsSpec
    stream: EventSpec
    theta_init: SpringTheta
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

    @property
    def zones(self) -> int:
        return len(self.theta_init.readout_centers)

    @property
    def space(self) -> SpringThetaSpace:
        return SpringThetaSpace(self.space_size, INPUTS, self.zones)

    def stream_ranges(self) -> tuple[range, range, range]:
        a, b = self.train_streams, self.train_streams + self.val_streams
        return range(0, a), range(a, b), range(b, b + self.test_streams)


def load_config(path: str | Path) -> E2PConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        return parse_config(json.load(file))


def parse_config(raw: dict[str, Any]) -> E2PConfig:
    _require(raw, "name", "tasks", "world", "network", "physics", "events", "readout", "theta_init", "split")
    net, phys, events, readout, split = raw["network"], raw["physics"], raw["events"], raw["readout"], raw["split"]
    _require(net, "lattice", "neighbors", "jitter")
    _require(phys, "dt", "substeps", "seed")
    _require(events, "pulses", "gap_min", "gap_max", "pulse_steps", "warmup")
    _require(readout, "delay", "ridge_alpha")
    _require(split, "washout", "train_streams", "val_streams", "test_streams")
    runtime = raw.get("runtime", {})
    config = E2PConfig(
        name=str(raw["name"]),
        tasks=tuple(str(task) for task in raw["tasks"]),
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
        stream=EventSpec(
            pulses=int(events["pulses"]),
            gap_min=int(events["gap_min"]),
            gap_max=int(events["gap_max"]),
            pulse_steps=int(events["pulse_steps"]),
            warmup=int(events["warmup"]),
        ),
        theta_init=SpringTheta.from_payload(raw["theta_init"]),
        ridge_alpha=float(readout["ridge_alpha"]),
        washout=int(split["washout"]),
        train_streams=int(split["train_streams"]),
        val_streams=int(split["val_streams"]),
        test_streams=int(split["test_streams"]),
        objective={key: float(value) for key, value in raw.get("objective", {}).items()},
        outer=dict(raw.get("outer", {})),
        workers=max(1, int(runtime.get("workers", 1))),
        chunk_size=max(1, int(runtime.get("chunk_size", 128))),
    )
    _validate(config)
    return config


def with_events(config: E2PConfig, **changes: int) -> E2PConfig:
    return replace(config, stream=replace(config.stream, **changes))


def _validate(config: E2PConfig) -> None:
    unknown = [task for task in config.tasks if task not in STREAM_TASKS]
    if unknown or not config.tasks:
        raise ValueError(f"unknown or missing stream tasks: {unknown}")
    if len(config.lattice) != 3 or min(config.lattice) <= 0:
        raise ValueError("network.lattice must contain 3 positive integers")
    if not 1 <= config.readout_delay <= config.stream.gap_min:
        raise ValueError(f"readout.delay must be within [1, {config.stream.gap_min}] (gap_min)")
    if not 0 <= config.washout < config.stream.pulses:
        raise ValueError("split.washout must leave scored pulses")
    if config.train_streams <= 0 or config.val_streams < 0 or config.test_streams < 0:
        raise ValueError("need positive train_streams and non-negative val/test streams")
    if len(config.theta_init.stimulus_points) != INPUTS:
        raise ValueError(f"theta_init needs exactly {INPUTS} stimulus points")
    space = config.space
    space.decode(space.encode(config.theta_init))
