from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# Global physics parameters, each mapped from an unconstrained real u by a
# sigmoid into [low, high] (or into [log low, log high] for log scales).
SCALARS: tuple[tuple[str, float, float, bool], ...] = (
    ("damping", 0.80, 1.00, False),
    ("restitution", 0.00, 1.00, False),
    ("input_strength", 0.01, 2.00, True),
    ("initial_speed", 0.00, 0.15, False),
    ("mass_spread", 0.00, 1.50, False),
)


@dataclass(frozen=True, slots=True)
class Theta:
    """Physical parameters of the single trainable Pro physics system."""

    damping: float
    restitution: float
    input_strength: float
    initial_speed: float
    mass_spread: float  # masses = exp(mass_spread * z) / mean, z fixed per world
    stimulus_points: tuple[tuple[float, float, float], ...]  # one per input bit

    def to_payload(self) -> dict[str, Any]:
        return {
            "damping": self.damping,
            "restitution": self.restitution,
            "input_strength": self.input_strength,
            "initial_speed": self.initial_speed,
            "mass_spread": self.mass_spread,
            "stimulus_points": [list(point) for point in self.stimulus_points],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Theta":
        return cls(
            damping=float(payload["damping"]),
            restitution=float(payload["restitution"]),
            input_strength=float(payload["input_strength"]),
            initial_speed=float(payload["initial_speed"]),
            mass_spread=float(payload["mass_spread"]),
            stimulus_points=tuple(tuple(float(v) for v in point) for point in payload["stimulus_points"]),  # type: ignore[misc]
        )


class ThetaSpace:
    """Maps unconstrained vectors u in R^d to valid physics parameters.

    Stimulus points are kept inside the box with a margin, so every u is a
    valid physics system and the outer optimizer never needs constraints.
    """

    def __init__(self, space_size: tuple[float, float, float], inputs: int, margin: float = 0.25) -> None:
        self.low = np.full(3, margin)
        self.high = np.array(space_size, dtype=np.float64) - margin
        self.inputs = inputs

    @property
    def dim(self) -> int:
        return len(SCALARS) + 3 * self.inputs

    def decode(self, u: np.ndarray) -> Theta:
        u = np.asarray(u, dtype=np.float64)
        if u.shape != (self.dim,):
            raise ValueError(f"theta vector must have shape ({self.dim},)")
        values = {}
        for index, (name, low, high, log) in enumerate(SCALARS):
            s = _sigmoid(u[index])
            values[name] = float(np.exp(np.log(low) + s * (np.log(high) - np.log(low))) if log else low + s * (high - low))
        points = self.low + _sigmoid(u[len(SCALARS) :].reshape(self.inputs, 3)) * (self.high - self.low)
        return Theta(stimulus_points=tuple(tuple(float(v) for v in row) for row in points), **values)  # type: ignore[arg-type]

    def encode(self, theta: Theta) -> np.ndarray:
        u = []
        for name, low, high, log in SCALARS:
            value = getattr(theta, name)
            s = (np.log(value) - np.log(low)) / (np.log(high) - np.log(low)) if log else (value - low) / (high - low)
            u.append(_logit(s))
        points = (np.array(theta.stimulus_points, dtype=np.float64) - self.low) / (self.high - self.low)
        if points.shape != (self.inputs, 3):
            raise ValueError("theta stimulus point count must match the task inputs")
        return np.concatenate([np.array(u), _logit(points).ravel()])


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _logit(s: np.ndarray | float, eps: float = 1e-4) -> np.ndarray:
    s = np.clip(np.asarray(s, dtype=np.float64), eps, 1 - eps)
    return np.log(s / (1 - s))
