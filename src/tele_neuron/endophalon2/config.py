from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

from tele_neuron.endophalon.config import _require, _triple
from tele_neuron.endophalon2.engine import StreamSpec
from tele_neuron.endophalon2.tasks import STREAM_TASKS
from tele_neuron.endophalon2.theta import Theta2, Theta2Space
from tele_neuron.plus.world import WorldSpec


INPUTS = 2  # every stream task uses two input channels


@dataclass(frozen=True, slots=True)
class E2Config:
    name: str
    tasks: tuple[str, ...]
    readout_delay: int  # n: readout this many steps after each pulse onset
    world: WorldSpec
    stream: StreamSpec
    theta_init: Theta2
    ridge_alpha: float
    washout: int  # pulses at the start of every stream left out of fitting and scoring
    train_streams: int
    val_streams: int
    test_streams: int
    objective: dict[str, float]
    outer: dict[str, Any]
    workers: int
    chunk_size: int

    @property
    def zones(self) -> int:
        return len(self.theta_init.readout_centers)

    @property
    def space(self) -> Theta2Space:
        return Theta2Space(self.world.space_size, INPUTS, self.zones)

    def stream_ranges(self) -> tuple[range, range, range]:
        a, b = self.train_streams, self.train_streams + self.val_streams
        return range(0, a), range(a, b), range(b, b + self.test_streams)

    def with_theta(self, theta: Theta2) -> "E2Config":
        return replace(self, theta_init=theta)


def load_config(path: str | Path) -> E2Config:
    with Path(path).open("r", encoding="utf-8") as file:
        return parse_config(json.load(file))


def parse_config(raw: dict[str, Any]) -> E2Config:
    _require(raw, "name", "tasks", "world", "stream", "readout", "theta_init", "split")
    world_raw, stream_raw, readout, split = raw["world"], raw["stream"], raw["readout"], raw["split"]
    _require(world_raw, "space", "balls", "simulation")
    _require(stream_raw, "pulses", "interval", "pulse_steps", "warmup")
    _require(readout, "delay", "ridge_alpha")
    _require(split, "washout", "train_streams", "val_streams", "test_streams")
    sim, balls = world_raw["simulation"], world_raw["balls"]
    size = _triple(world_raw["space"]["size"], "world.space.size")
    world = WorldSpec(
        space_size=size,
        ball_count=int(balls["count"]),
        ball_radius=float(balls["radius"]),
        input_points=tuple((size[0] / 2, size[1] / 2, size[2] / 2) for _ in range(INPUTS)),
        input_epsilon=float(world_raw.get("input", {}).get("epsilon", 1e-6)),
        steps=1,  # unused: stream length comes from the stream timeline
        dt=float(sim.get("dt", 1.0)),
        cell_size=float(sim["cell_size"]),
        seed=int(sim["seed"]),
        position_jitter=float(world_raw.get("noise", {}).get("position_jitter", 0.1)),
        velocity_jitter=float(world_raw.get("noise", {}).get("velocity_jitter", 0.02)),
    )
    stream = StreamSpec(
        pulses=int(stream_raw["pulses"]),
        interval=int(stream_raw["interval"]),
        pulse_steps=int(stream_raw["pulse_steps"]),
        warmup=int(stream_raw["warmup"]),
    )
    runtime = raw.get("runtime", {})
    config = E2Config(
        name=str(raw["name"]),
        tasks=tuple(str(task) for task in raw["tasks"]),
        readout_delay=int(readout["delay"]),
        world=world,
        stream=stream,
        theta_init=Theta2.from_payload(raw["theta_init"]),
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


def _validate(config: E2Config) -> None:
    unknown = [task for task in config.tasks if task not in STREAM_TASKS]
    if unknown or not config.tasks:
        raise ValueError(f"unknown or missing stream tasks: {unknown}")
    if not 1 <= config.readout_delay <= config.stream.interval:
        raise ValueError(f"readout.delay must be within [1, {config.stream.interval}]")
    if not 0 <= config.washout < config.stream.pulses:
        raise ValueError("split.washout must leave at least one scored pulse")
    if config.train_streams <= 0 or config.val_streams < 0 or config.test_streams < 0:
        raise ValueError("need positive train_streams and non-negative val/test streams")
    if config.ridge_alpha <= 0:
        raise ValueError("readout.ridge_alpha must be positive")
    if len(config.theta_init.stimulus_points) != INPUTS:
        raise ValueError(f"theta_init needs exactly {INPUTS} stimulus points")
    space = config.space
    space.decode(space.encode(config.theta_init))
