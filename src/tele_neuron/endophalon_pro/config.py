from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

from tele_neuron.endophalon.config import _parse_cases as _parse_explicit_cases, _require, _triple
from tele_neuron.endophalon.tasks import Case
from tele_neuron.endophalon_plus.tasks import task_cases
from tele_neuron.endophalon_pro.theta import Theta, ThetaSpace
from tele_neuron.plus.world import WorldSpec


@dataclass(frozen=True, slots=True)
class ReadoutSpec:
    step: int  # the readout reads the physics this many steps after the input starts
    grid: tuple[int, int, int]
    ridge_alpha: float


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    lambda_noise: float
    lambda_stability: float
    lambda_complexity: float


@dataclass(frozen=True, slots=True)
class OuterSpec:
    generations: int
    population: int
    elite: int
    init_sigma: float
    min_sigma: float
    seed: int


@dataclass(frozen=True, slots=True)
class SplitSpec:
    train: int
    val: int
    test: int

    def ranges(self) -> tuple[range, range, range]:
        return (
            range(0, self.train),
            range(self.train, self.train + self.val),
            range(self.train + self.val, self.train + self.val + self.test),
        )


@dataclass(frozen=True, slots=True)
class ProConfig:
    name: str
    task: str
    world: WorldSpec
    readout: ReadoutSpec
    theta_init: Theta
    objective: ObjectiveSpec
    outer: OuterSpec
    splits: SplitSpec
    sweep_grids: tuple[tuple[int, int, int], ...]
    workers: int
    chunk_size: int
    cases: tuple[Case, ...]

    @property
    def input_width(self) -> int:
        return len(self.cases[0][0])

    @property
    def space(self) -> ThetaSpace:
        return ThetaSpace(self.world.space_size, self.input_width)

    def with_noise(self, position_jitter: float, velocity_jitter: float) -> "ProConfig":
        return replace(self, world=replace(self.world, position_jitter=position_jitter, velocity_jitter=velocity_jitter))


def load_config(path: str | Path) -> ProConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        return parse_config(json.load(file))


def parse_config(raw: dict[str, Any]) -> ProConfig:
    _require(raw, "name", "task", "world", "readout", "theta_init", "objective", "outer", "noise")
    noise = raw["noise"]
    _require(noise, "position_jitter", "velocity_jitter", "train_realizations", "val_realizations", "test_realizations")
    cases = tuple(_parse_explicit_cases(raw)) if "cases" in raw else task_cases(str(raw["task"]))
    world = _parse_world(raw["world"], noise, len(cases[0][0]))
    readout = raw["readout"]
    _require(readout, "step", "grid", "ridge_alpha")
    objective = raw["objective"]
    outer = raw["outer"]
    runtime = raw.get("runtime", {})
    config = ProConfig(
        name=str(raw["name"]),
        task=str(raw["task"]),
        world=world,
        readout=ReadoutSpec(
            step=int(readout["step"]),
            grid=_grid(readout["grid"], "readout.grid"),
            ridge_alpha=float(readout["ridge_alpha"]),
        ),
        theta_init=Theta.from_payload(raw["theta_init"]),
        objective=ObjectiveSpec(
            lambda_noise=float(objective.get("lambda_noise", 0.0)),
            lambda_stability=float(objective.get("lambda_stability", 0.0)),
            lambda_complexity=float(objective.get("lambda_complexity", 0.0)),
        ),
        outer=OuterSpec(
            generations=max(0, int(outer.get("generations", 20))),
            population=max(2, int(outer.get("population", 16))),
            elite=max(1, int(outer.get("elite", 4))),
            init_sigma=float(outer.get("init_sigma", 0.8)),
            min_sigma=float(outer.get("min_sigma", 0.05)),
            seed=int(outer.get("seed", 1)),
        ),
        splits=SplitSpec(
            train=int(noise["train_realizations"]),
            val=int(noise["val_realizations"]),
            test=int(noise["test_realizations"]),
        ),
        sweep_grids=tuple(_grid(grid, "sweep.grids[]") for grid in raw.get("sweep", {}).get("grids", [])),
        workers=max(1, int(runtime.get("workers", 1))),
        chunk_size=max(1, int(runtime.get("chunk_size", 128))),
        cases=cases,
    )
    _validate(config)
    return config


def _parse_world(raw: dict[str, Any], noise: dict[str, Any], inputs: int) -> WorldSpec:
    _require(raw, "space", "balls", "simulation")
    sim, balls = raw["simulation"], raw["balls"]
    _require(sim, "steps", "dt", "cell_size", "seed")
    _require(balls, "count", "radius")
    size = _triple(raw["space"]["size"], "world.space.size")
    return WorldSpec(
        space_size=size,
        ball_count=int(balls["count"]),
        ball_radius=float(balls["radius"]),
        # Stimulus points are part of theta; the world only needs placeholders.
        input_points=tuple((size[0] / 2, size[1] / 2, size[2] / 2) for _ in range(inputs)),
        input_epsilon=float(raw.get("input", {}).get("epsilon", 1e-6)),
        steps=int(sim["steps"]),
        dt=float(sim["dt"]),
        cell_size=float(sim["cell_size"]),
        seed=int(sim["seed"]),
        position_jitter=float(noise["position_jitter"]),
        velocity_jitter=float(noise["velocity_jitter"]),
    )


def _grid(value: Any, name: str) -> tuple[int, int, int]:
    grid = tuple(int(v) for v in value)
    if len(grid) != 3 or min(grid) <= 0:
        raise ValueError(f"{name} must contain 3 positive integers")
    return grid  # type: ignore[return-value]


def _validate(config: ProConfig) -> None:
    if not 1 <= config.readout.step <= config.world.steps:
        raise ValueError(f"readout.step must be within [1, {config.world.steps}]")
    if config.readout.ridge_alpha <= 0:
        raise ValueError("readout.ridge_alpha must be positive")
    if len(config.theta_init.stimulus_points) != config.input_width:
        raise ValueError("theta_init.stimulus_points must have one point per input bit")
    if config.splits.train <= 0 or config.splits.val <= 0 or config.splits.test < 0:
        raise ValueError("need positive train and val realizations and non-negative test realizations")
    if config.outer.elite > config.outer.population:
        raise ValueError("outer.elite must not exceed outer.population")
    space = config.space
    space.decode(space.encode(config.theta_init))  # raises if theta_init is out of range
