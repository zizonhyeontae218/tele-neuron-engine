from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

from tele_neuron.endophalon.config import _parse_cases as _parse_explicit_cases, _require, _triple
from tele_neuron.endophalon.tasks import Case
from tele_neuron.endophalon_plus.tasks import task_cases
from tele_neuron.plus.world import Preset, WorldSpec, preset_from_payload


PRESET_COUNT = 3


@dataclass(frozen=True, slots=True)
class ObserverSpec:
    grids: tuple[tuple[int, int, int], ...]
    taps: tuple[int, ...]
    speed: bool


@dataclass(frozen=True, slots=True)
class GateSpec:
    mode: str  # "linear" (learned) or "fixed"
    fixed_preset: int
    restarts: int
    iterations: int
    seed: int
    validation_realizations: int


@dataclass(frozen=True, slots=True)
class PlusConfig:
    name: str
    task: str
    world: WorldSpec
    presets: tuple[Preset, ...]
    observer: ObserverSpec
    gate: GateSpec
    ridge_alpha: float
    train_realizations: int
    test_realizations: int
    workers: int
    chunk_size: int
    cases: tuple[Case, ...]

    @property
    def input_width(self) -> int:
        return len(self.cases[0][0])

    @property
    def output_width(self) -> int:
        return len(self.cases[0][1])

    def with_noise(self, position_jitter: float, velocity_jitter: float) -> "PlusConfig":
        world = replace(self.world, position_jitter=position_jitter, velocity_jitter=velocity_jitter)
        return replace(self, world=world)


def load_config(path: str | Path) -> PlusConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        return parse_config(json.load(file))


def parse_config(raw: dict[str, Any]) -> PlusConfig:
    _require(raw, "name", "task", "world", "presets", "observer", "gate", "readout", "noise")
    noise = raw["noise"]
    _require(noise, "position_jitter", "velocity_jitter", "train_realizations", "test_realizations")
    world = _parse_world(raw["world"], noise)
    presets = tuple(preset_from_payload(item) for item in raw["presets"])
    if len(presets) != PRESET_COUNT:
        raise ValueError(f"exactly {PRESET_COUNT} presets are required, got {len(presets)}")
    runtime = raw.get("runtime", {})
    config = PlusConfig(
        name=str(raw["name"]),
        task=str(raw["task"]),
        world=world,
        presets=presets,
        observer=_parse_observer(raw["observer"], world.steps),
        gate=_parse_gate(raw["gate"]),
        ridge_alpha=float(raw["readout"]["ridge_alpha"]),
        train_realizations=int(noise["train_realizations"]),
        test_realizations=int(noise["test_realizations"]),
        workers=max(1, int(runtime.get("workers", 1))),
        chunk_size=max(1, int(runtime.get("chunk_size", 64))),
        cases=_parse_cases(raw),
    )
    if config.input_width != len(world.input_points):
        raise ValueError(
            f"task input width must match world.input.points length "
            f"({config.input_width} != {len(world.input_points)})"
        )
    if config.ridge_alpha <= 0:
        raise ValueError("readout.ridge_alpha must be positive")
    if config.test_realizations < 0:
        raise ValueError("noise.test_realizations must be non-negative")
    if config.gate.mode == "linear" and not 0 < config.gate.validation_realizations < config.train_realizations:
        raise ValueError("gate.validation_realizations must be between 1 and train_realizations - 1")
    if config.gate.mode == "fixed" and config.train_realizations <= 0:
        raise ValueError("noise.train_realizations must be positive")
    return config


def _parse_cases(raw: dict[str, Any]) -> tuple[Case, ...]:
    if "cases" in raw:
        return _parse_explicit_cases(raw)
    return task_cases(str(raw["task"]))


def _parse_world(raw: dict[str, Any], noise: dict[str, Any]) -> WorldSpec:
    _require(raw, "space", "balls", "input", "simulation")
    sim, balls, inputs = raw["simulation"], raw["balls"], raw["input"]
    _require(sim, "steps", "dt", "cell_size", "seed")
    _require(balls, "count", "radius")
    _require(inputs, "points", "epsilon")
    return WorldSpec(
        space_size=_triple(raw["space"]["size"], "world.space.size"),
        ball_count=int(balls["count"]),
        ball_radius=float(balls["radius"]),
        input_points=tuple(_triple(point, "world.input.points[]") for point in inputs["points"]),
        input_epsilon=float(inputs["epsilon"]),
        steps=int(sim["steps"]),
        dt=float(sim["dt"]),
        cell_size=float(sim["cell_size"]),
        seed=int(sim["seed"]),
        position_jitter=float(noise["position_jitter"]),
        velocity_jitter=float(noise["velocity_jitter"]),
    )


def _parse_observer(raw: dict[str, Any], steps: int) -> ObserverSpec:
    _require(raw, "grids", "taps")
    grids = tuple(tuple(int(value) for value in grid) for grid in raw["grids"])
    if not grids or any(len(grid) != 3 or min(grid) <= 0 for grid in grids):
        raise ValueError("observer.grids must be a non-empty list of 3 positive integers each")
    taps = tuple(sorted({int(tap) for tap in raw["taps"]}))
    if not taps or taps[0] < 1 or taps[-1] > steps:
        raise ValueError(f"observer.taps must be non-empty steps within [1, {steps}]")
    return ObserverSpec(grids=grids, taps=taps, speed=bool(raw.get("speed", True)))  # type: ignore[arg-type]


def _parse_gate(raw: dict[str, Any]) -> GateSpec:
    mode = str(raw.get("mode", "linear"))
    if mode not in {"linear", "fixed"}:
        raise ValueError("gate.mode must be 'linear' or 'fixed'")
    gate = GateSpec(
        mode=mode,
        fixed_preset=int(raw.get("fixed_preset", 0)),
        restarts=max(1, int(raw.get("restarts", 24))),
        iterations=max(0, int(raw.get("iterations", 40))),
        seed=int(raw.get("seed", 1)),
        validation_realizations=int(raw.get("validation_realizations", 0)),
    )
    if not 0 <= gate.fixed_preset < PRESET_COUNT:
        raise ValueError(f"gate.fixed_preset must be in [0, {PRESET_COUNT})")
    return gate
