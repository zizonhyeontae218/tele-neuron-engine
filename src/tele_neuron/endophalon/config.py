from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from tele_neuron.endophalon.tasks import Case, task_cases


@dataclass(frozen=True, slots=True)
class ReservoirConfig:
    space_size: tuple[float, float, float]
    steps: int
    dt: float
    damping: float
    restitution: float
    cell_size: float
    seed: int
    ball_count: int
    ball_mass: float
    ball_radius: float
    initial_speed: float
    input_points: tuple[tuple[float, float, float], ...]
    input_strength: float
    input_epsilon: float


@dataclass(frozen=True, slots=True)
class ObserverConfig:
    grid: tuple[int, int, int]
    taps: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ReadoutConfig:
    ridge_alpha: float


@dataclass(frozen=True, slots=True)
class NoiseConfig:
    position_jitter: float
    velocity_jitter: float
    train_realizations: int
    test_realizations: int


@dataclass(frozen=True, slots=True)
class EndophalonConfig:
    name: str
    task: str
    reservoir: ReservoirConfig
    observer: ObserverConfig
    readout: ReadoutConfig
    noise: NoiseConfig
    cases: tuple[Case, ...]

    @property
    def input_width(self) -> int:
        return len(self.cases[0][0])

    @property
    def output_width(self) -> int:
        return len(self.cases[0][1])


def load_config(path: str | Path) -> EndophalonConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        return parse_config(json.load(file))


def parse_config(raw: dict[str, Any]) -> EndophalonConfig:
    _require(raw, "name", "task", "reservoir", "observer", "readout", "noise")
    reservoir = _parse_reservoir(raw["reservoir"])
    cases = _parse_cases(raw)
    config = EndophalonConfig(
        name=str(raw["name"]),
        task=str(raw["task"]),
        reservoir=reservoir,
        observer=_parse_observer(raw["observer"], reservoir.steps),
        readout=_parse_readout(raw["readout"]),
        noise=_parse_noise(raw["noise"]),
        cases=cases,
    )
    if config.input_width != len(reservoir.input_points):
        raise ValueError(
            f"task input width must match reservoir.input.points length "
            f"({config.input_width} != {len(reservoir.input_points)})"
        )
    return config


def _parse_reservoir(raw: dict[str, Any]) -> ReservoirConfig:
    _require(raw, "space", "simulation", "balls", "input")
    sim, balls, inputs = raw["simulation"], raw["balls"], raw["input"]
    _require(sim, "steps", "dt", "damping", "restitution", "cell_size", "seed")
    _require(balls, "count", "mass", "radius", "initial_speed")
    _require(inputs, "points", "strength", "epsilon")
    config = ReservoirConfig(
        space_size=_triple(raw["space"]["size"], "reservoir.space.size"),
        steps=int(sim["steps"]),
        dt=float(sim["dt"]),
        damping=float(sim["damping"]),
        restitution=float(sim["restitution"]),
        cell_size=float(sim["cell_size"]),
        seed=int(sim["seed"]),
        ball_count=int(balls["count"]),
        ball_mass=float(balls["mass"]),
        ball_radius=float(balls["radius"]),
        initial_speed=float(balls["initial_speed"]),
        input_points=tuple(_triple(point, "reservoir.input.points[]") for point in inputs["points"]),
        input_strength=float(inputs["strength"]),
        input_epsilon=float(inputs["epsilon"]),
    )
    if config.steps <= 0 or config.dt <= 0:
        raise ValueError("reservoir.simulation steps and dt must be positive")
    if not 0 < config.damping <= 1:
        raise ValueError("reservoir.simulation.damping must be in (0, 1]")
    if not 0 <= config.restitution <= 1:
        raise ValueError("reservoir.simulation.restitution must be in [0, 1]")
    if config.cell_size <= 0:
        raise ValueError("reservoir.simulation.cell_size must be positive")
    if config.ball_count <= 0 or config.ball_mass <= 0 or config.ball_radius <= 0:
        raise ValueError("reservoir.balls count, mass and radius must be positive")
    if config.initial_speed < 0 or config.input_strength < 0:
        raise ValueError("reservoir initial_speed and input strength must be non-negative")
    if config.input_epsilon <= 0:
        raise ValueError("reservoir.input.epsilon must be positive")
    if not config.input_points:
        raise ValueError("reservoir.input.points must not be empty")
    return config


def _parse_observer(raw: dict[str, Any], steps: int) -> ObserverConfig:
    _require(raw, "grid", "taps")
    grid = tuple(int(value) for value in raw["grid"])
    taps = tuple(sorted({int(value) for value in raw["taps"]}))
    if len(grid) != 3 or any(value <= 0 for value in grid):
        raise ValueError("observer.grid must contain 3 positive integers")
    if not taps or any(tap < 1 or tap > steps for tap in taps):
        raise ValueError(f"observer.taps must be non-empty steps within [1, {steps}]")
    return ObserverConfig(grid=grid, taps=taps)  # type: ignore[arg-type]


def _parse_readout(raw: dict[str, Any]) -> ReadoutConfig:
    _require(raw, "ridge_alpha")
    alpha = float(raw["ridge_alpha"])
    if alpha < 0:
        raise ValueError("readout.ridge_alpha must be non-negative")
    return ReadoutConfig(ridge_alpha=alpha)


def _parse_noise(raw: dict[str, Any]) -> NoiseConfig:
    _require(raw, "position_jitter", "velocity_jitter", "train_realizations", "test_realizations")
    config = NoiseConfig(
        position_jitter=float(raw["position_jitter"]),
        velocity_jitter=float(raw["velocity_jitter"]),
        train_realizations=int(raw["train_realizations"]),
        test_realizations=int(raw["test_realizations"]),
    )
    if config.position_jitter < 0 or config.velocity_jitter < 0:
        raise ValueError("noise jitters must be non-negative")
    if config.train_realizations <= 0 or config.test_realizations < 0:
        raise ValueError("noise.train_realizations must be positive and test_realizations non-negative")
    return config


def _parse_cases(raw: dict[str, Any]) -> tuple[Case, ...]:
    if "cases" not in raw:
        return task_cases(str(raw["task"]))
    cases: list[Case] = []
    for item in raw["cases"]:
        _require(item, "bits", "target")
        bits = tuple(int(bit) for bit in item["bits"])
        target_raw = item["target"]
        target = (int(target_raw),) if isinstance(target_raw, int) else tuple(int(bit) for bit in target_raw)
        if any(bit not in (0, 1) for bit in (*bits, *target)) or not bits or not target:
            raise ValueError("case bits and target must be non-empty 0/1 values")
        cases.append((bits, target))
    if not cases:
        raise ValueError("cases must not be empty")
    if len({len(bits) for bits, _ in cases}) != 1 or len({len(target) for _, target in cases}) != 1:
        raise ValueError("all cases must share the same bits and target widths")
    return tuple(cases)


def _triple(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly 3 numbers")
    return (float(value[0]), float(value[1]), float(value[2]))


def _require(raw: dict[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in raw]
    if missing:
        raise ValueError(f"missing required config field(s): {', '.join(missing)}")
