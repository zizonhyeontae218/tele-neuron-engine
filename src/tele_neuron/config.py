from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SpaceConfig:
    size: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    steps: int
    dt: float
    damping: float
    restitution: float
    cell_size: float
    seed: int


@dataclass(frozen=True, slots=True)
class BallInitConfig:
    count: int
    mass: float
    radius: float
    initial_speed: float


@dataclass(frozen=True, slots=True)
class InputConfig:
    points: tuple[tuple[float, float, float], ...]
    strength: float
    epsilon: float


@dataclass(frozen=True, slots=True)
class ReadoutZoneConfig:
    name: str
    center: tuple[float, float, float]
    half_extents: tuple[float, float, float]
    threshold: int


@dataclass(frozen=True, slots=True)
class CaseConfig:
    bits: tuple[int, ...]
    target: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    task: str
    space: SpaceConfig
    simulation: SimulationConfig
    balls: BallInitConfig
    input: InputConfig
    readout: tuple[ReadoutZoneConfig, ...]
    cases: tuple[CaseConfig, ...]


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> ExperimentConfig:
    _require(raw, "name", "task", "space", "simulation", "balls", "input", "readout", "cases")

    config = ExperimentConfig(
        name=str(raw["name"]),
        task=str(raw["task"]),
        space=SpaceConfig(size=_triple(raw["space"]["size"], "space.size")),
        simulation=_parse_simulation(raw["simulation"]),
        balls=_parse_balls(raw["balls"]),
        input=_parse_input(raw["input"]),
        readout=tuple(_parse_readout_zone(item) for item in raw["readout"]),
        cases=tuple(_parse_case(item) for item in raw["cases"]),
    )
    _validate_config_shape(config)
    return config


def _parse_simulation(raw: dict[str, Any]) -> SimulationConfig:
    _require(raw, "steps", "dt", "damping", "restitution", "cell_size", "seed")
    config = SimulationConfig(
        steps=int(raw["steps"]),
        dt=float(raw["dt"]),
        damping=float(raw["damping"]),
        restitution=float(raw["restitution"]),
        cell_size=float(raw["cell_size"]),
        seed=int(raw["seed"]),
    )
    if config.steps <= 0:
        raise ValueError("simulation.steps must be positive")
    if config.dt <= 0:
        raise ValueError("simulation.dt must be positive")
    if not 0 < config.damping <= 1:
        raise ValueError("simulation.damping must be in (0, 1]")
    if not 0 <= config.restitution <= 1:
        raise ValueError("simulation.restitution must be in [0, 1]")
    if config.cell_size <= 0:
        raise ValueError("simulation.cell_size must be positive")
    return config


def _parse_balls(raw: dict[str, Any]) -> BallInitConfig:
    _require(raw, "count", "mass", "radius", "initial_speed")
    config = BallInitConfig(
        count=int(raw["count"]),
        mass=float(raw["mass"]),
        radius=float(raw["radius"]),
        initial_speed=float(raw["initial_speed"]),
    )
    if config.count <= 0:
        raise ValueError("balls.count must be positive")
    if config.mass <= 0:
        raise ValueError("balls.mass must be positive")
    if config.radius <= 0:
        raise ValueError("balls.radius must be positive")
    if config.initial_speed < 0:
        raise ValueError("balls.initial_speed must be non-negative")
    return config


def _parse_input(raw: dict[str, Any]) -> InputConfig:
    _require(raw, "points", "strength", "epsilon")
    config = InputConfig(
        points=tuple(_triple(point, "input.points[]") for point in raw["points"]),
        strength=float(raw["strength"]),
        epsilon=float(raw["epsilon"]),
    )
    if not config.points:
        raise ValueError("input.points must not be empty")
    if config.strength < 0:
        raise ValueError("input.strength must be non-negative")
    if config.epsilon <= 0:
        raise ValueError("input.epsilon must be positive")
    return config


def _parse_readout_zone(raw: dict[str, Any]) -> ReadoutZoneConfig:
    _require(raw, "name", "center", "half_extents", "threshold")
    config = ReadoutZoneConfig(
        name=str(raw["name"]),
        center=_triple(raw["center"], "readout.center"),
        half_extents=_triple(raw["half_extents"], "readout.half_extents"),
        threshold=int(raw["threshold"]),
    )
    if config.threshold < 0:
        raise ValueError("readout.threshold must be non-negative")
    if any(value <= 0 for value in config.half_extents):
        raise ValueError("readout.half_extents values must be positive")
    return config


def _parse_case(raw: dict[str, Any]) -> CaseConfig:
    _require(raw, "bits", "target")
    bits = tuple(int(bit) for bit in raw["bits"])
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("case bits must be 0 or 1")
    target = _bits_value(raw["target"], "case target")
    if any(bit not in (0, 1) for bit in target):
        raise ValueError("case target must be 0 or 1")
    return CaseConfig(bits=bits, target=target)


def _bits_value(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        return (int(value),)
    if not isinstance(value, list | tuple):
        raise ValueError(f"{name} must be an int or list of bits")
    bits = tuple(int(bit) for bit in value)
    if not bits:
        raise ValueError(f"{name} must not be empty")
    return bits


def _triple(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly 3 numbers")
    return (float(value[0]), float(value[1]), float(value[2]))


def _require(raw: dict[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in raw]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"missing required config field(s): {joined}")


def _validate_config_shape(config: ExperimentConfig) -> None:
    if not config.readout:
        raise ValueError("readout must not be empty")
    for index, case in enumerate(config.cases):
        if len(case.bits) != len(config.input.points):
            raise ValueError(
                f"case {index} bits length must match input.points length "
                f"({len(case.bits)} != {len(config.input.points)})"
            )
        if len(case.target) != len(config.readout):
            raise ValueError(
                f"case {index} target length must match readout length "
                f"({len(case.target)} != {len(config.readout)})"
            )
