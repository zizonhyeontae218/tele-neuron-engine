from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class WorldSpec:
    """The fixed stage every preset plays on."""

    space_size: tuple[float, float, float]
    ball_count: int
    ball_radius: float
    input_points: tuple[tuple[float, float, float], ...]
    input_epsilon: float
    steps: int
    dt: float
    cell_size: float
    seed: int
    position_jitter: float = 0.0
    velocity_jitter: float = 0.0

    def __post_init__(self) -> None:
        if self.ball_count <= 0 or self.ball_radius <= 0:
            raise ValueError("world ball count and radius must be positive")
        if self.steps <= 0 or self.dt <= 0:
            raise ValueError("world steps and dt must be positive")
        if self.cell_size < 2 * self.ball_radius:
            raise ValueError("world cell_size must be at least one ball diameter")
        if not self.input_points:
            raise ValueError("world input_points must not be empty")
        if self.input_epsilon <= 0:
            raise ValueError("world input_epsilon must be positive")
        if self.position_jitter < 0 or self.velocity_jitter < 0:
            raise ValueError("world jitters must be non-negative")

    def base_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Seeded positions and unit directions shared by every realization."""
        rng = np.random.default_rng(self.seed)
        radius = self.ball_radius
        positions = rng.uniform(
            low=np.full(3, radius),
            high=np.array(self.space_size, dtype=np.float64) - radius,
            size=(self.ball_count, 3),
        )
        directions = rng.normal(size=(self.ball_count, 3))
        norms = np.linalg.norm(directions, axis=1)
        directions[norms > 0] /= norms[norms > 0, np.newaxis]
        return positions, directions

    def jitter(self, realization: int) -> tuple[np.ndarray, np.ndarray]:
        if realization < 0:
            raise ValueError("realization must be non-negative")
        rng = np.random.default_rng([self.seed, realization])
        shape = (self.ball_count, 3)
        return rng.normal(0.0, self.position_jitter, shape), rng.normal(0.0, self.velocity_jitter, shape)


@dataclass(frozen=True, slots=True)
class Preset:
    """One set of physical laws: how the world damps, bounces, and weighs."""

    name: str
    damping: float
    restitution: float
    input_strength: float
    initial_speed: float
    mass_mode: str = "uniform"
    mass_low: float = 1.0
    mass_high: float = 1.0
    heavy_fraction: float = 0.5

    def __post_init__(self) -> None:
        if not 0 < self.damping <= 1:
            raise ValueError(f"preset {self.name}: damping must be in (0, 1]")
        if not 0 <= self.restitution <= 1:
            raise ValueError(f"preset {self.name}: restitution must be in [0, 1]")
        if self.input_strength < 0 or self.initial_speed < 0:
            raise ValueError(f"preset {self.name}: input_strength and initial_speed must be non-negative")
        if self.mass_mode not in {"uniform", "bimodal"}:
            raise ValueError(f"preset {self.name}: mass mode must be 'uniform' or 'bimodal'")
        if self.mass_low <= 0 or self.mass_high < self.mass_low:
            raise ValueError(f"preset {self.name}: need 0 < mass low <= mass high")
        if not 0 <= self.heavy_fraction <= 1:
            raise ValueError(f"preset {self.name}: heavy_fraction must be in [0, 1]")

    def masses(self, world: WorldSpec) -> np.ndarray:
        """Per-ball masses; fixed per preset, identical across realizations."""
        count = world.ball_count
        if self.mass_mode == "bimodal":
            rng = np.random.default_rng([world.seed, 0x4D415353])
            heavy = rng.permutation(count) < round(self.heavy_fraction * count)
            return np.where(heavy, self.mass_high, self.mass_low).astype(np.float64)
        if self.mass_low == self.mass_high:
            return np.full(count, self.mass_low, dtype=np.float64)
        rng = np.random.default_rng([world.seed, 0x4D415353])
        return rng.uniform(self.mass_low, self.mass_high, count)


@dataclass(frozen=True, slots=True)
class Job:
    """One simulation inside a batch."""

    preset: int
    bits: tuple[int, ...]
    realization: int


def preset_from_payload(raw: dict[str, Any]) -> Preset:
    mass = raw.get("mass", {})
    return Preset(
        name=str(raw["name"]),
        damping=float(raw["damping"]),
        restitution=float(raw["restitution"]),
        input_strength=float(raw["input_strength"]),
        initial_speed=float(raw.get("initial_speed", 0.0)),
        mass_mode=str(mass.get("mode", "uniform")),
        mass_low=float(mass.get("low", 1.0)),
        mass_high=float(mass.get("high", mass.get("low", 1.0))),
        heavy_fraction=float(mass.get("heavy_fraction", 0.5)),
    )


def preset_to_payload(preset: Preset) -> dict[str, Any]:
    return {
        "name": preset.name,
        "damping": preset.damping,
        "restitution": preset.restitution,
        "input_strength": preset.input_strength,
        "initial_speed": preset.initial_speed,
        "mass": {
            "mode": preset.mass_mode,
            "low": preset.mass_low,
            "high": preset.mass_high,
            "heavy_fraction": preset.heavy_fraction,
        },
    }
