from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tele_neuron.endophalon_pro.theta import _logit, _sigmoid


# name, low, high, log-scale
SCALARS: tuple[tuple[str, float, float, bool], ...] = (
    ("damping", 0.80, 1.00, False),
    ("restitution", 0.00, 1.00, False),
    ("pulse_strength", 0.01, 2.00, True),
    ("mass_spread", 0.00, 1.50, False),
    ("anchor", 1e-5, 0.05, True),  # spring stiffness toward home positions
    ("bath", 1e-5, 0.02, True),  # heat-bath velocity kick per step (std)
    ("readout_width", 0.15, 1.50, True),  # soft readout zone radius
)


@dataclass(frozen=True, slots=True)
class Theta2:
    """Everything the Endophalon-2 outer loop may change."""

    damping: float
    restitution: float
    pulse_strength: float
    mass_spread: float
    anchor: float
    bath: float
    readout_width: float
    stimulus_points: tuple[tuple[float, float, float], ...]  # one per input channel
    readout_centers: tuple[tuple[float, float, float], ...]  # one per readout zone

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {name: getattr(self, name) for name, *_ in SCALARS}
        payload["stimulus_points"] = [list(point) for point in self.stimulus_points]
        payload["readout_centers"] = [list(point) for point in self.readout_centers]
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Theta2":
        return cls(
            **{name: float(payload[name]) for name, *_ in SCALARS},
            stimulus_points=_points(payload["stimulus_points"]),
            readout_centers=_points(payload["readout_centers"]),
        )


class Theta2Space:
    """Unconstrained vectors u in R^d <-> valid Theta2 (sigmoid-bounded)."""

    def __init__(self, space_size: tuple[float, float, float], inputs: int, zones: int, margin: float = 0.25) -> None:
        self.low = np.full(3, margin)
        self.high = np.array(space_size, dtype=np.float64) - margin
        self.inputs = inputs
        self.zones = zones

    @property
    def dim(self) -> int:
        return len(SCALARS) + 3 * (self.inputs + self.zones)

    def decode(self, u: np.ndarray) -> Theta2:
        u = np.asarray(u, dtype=np.float64)
        if u.shape != (self.dim,):
            raise ValueError(f"theta vector must have shape ({self.dim},)")
        values = {}
        for index, (name, low, high, log) in enumerate(SCALARS):
            s = float(_sigmoid(u[index]))
            values[name] = float(np.exp(np.log(low) + s * (np.log(high) - np.log(low)))) if log else low + s * (high - low)
        rest = u[len(SCALARS) :].reshape(self.inputs + self.zones, 3)
        points = self.low + _sigmoid(rest) * (self.high - self.low)
        return Theta2(
            **values,
            stimulus_points=_points(points[: self.inputs]),
            readout_centers=_points(points[self.inputs :]),
        )

    def encode(self, theta: Theta2) -> np.ndarray:
        u = []
        for name, low, high, log in SCALARS:
            value = getattr(theta, name)
            s = (np.log(value) - np.log(low)) / (np.log(high) - np.log(low)) if log else (value - low) / (high - low)
            u.append(float(_logit(s)))
        points = np.array(theta.stimulus_points + theta.readout_centers, dtype=np.float64)
        if points.shape != (self.inputs + self.zones, 3):
            raise ValueError("theta must have one stimulus point per input and one center per readout zone")
        return np.concatenate([np.array(u), _logit((points - self.low) / (self.high - self.low)).ravel()])


def _points(rows: Any) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(v) for v in row) for row in rows)  # type: ignore[misc]
