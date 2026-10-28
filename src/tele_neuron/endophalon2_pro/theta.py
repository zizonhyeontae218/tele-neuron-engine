from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tele_neuron.endophalon_pro.theta import _logit, _sigmoid


# name, low, high, log-scale
SCALARS: tuple[tuple[str, float, float, bool], ...] = (
    ("damping", 0.80, 1.00, False),  # per-step velocity retention
    ("k_linear", 0.005, 0.20, True),  # spring stiffness
    ("k_cubic", 1e-3, 2.0, True),  # cubic stiffening: force += k_cubic * extension^3
    ("mass_spread", 0.00, 1.50, False),
    ("anchor", 1e-5, 0.05, True),  # tether toward home (keeps the network from drifting away)
    ("bath", 1e-5, 0.02, True),  # heat-bath velocity kick per step
    ("pulse_strength", 0.01, 1.0, True),
    ("stimulus_width", 0.3, 2.0, True),  # radius of the Gaussian input push
    ("readout_width", 0.15, 1.5, True),  # radius of each soft readout zone
)


@dataclass(frozen=True, slots=True)
class SpringTheta:
    damping: float
    k_linear: float
    k_cubic: float
    mass_spread: float
    anchor: float
    bath: float
    pulse_strength: float
    stimulus_width: float
    readout_width: float
    stimulus_points: tuple[tuple[float, float, float], ...]
    readout_centers: tuple[tuple[float, float, float], ...]

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {name: getattr(self, name) for name, *_ in SCALARS}
        payload["stimulus_points"] = [list(point) for point in self.stimulus_points]
        payload["readout_centers"] = [list(point) for point in self.readout_centers]
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SpringTheta":
        return cls(
            **{name: float(payload[name]) for name, *_ in SCALARS},
            stimulus_points=_points(payload["stimulus_points"]),
            readout_centers=_points(payload["readout_centers"]),
        )


class SpringThetaSpace:
    """Unconstrained u in R^d <-> valid SpringTheta; points stay inside the box."""

    def __init__(self, space_size: tuple[float, float, float], inputs: int, zones: int, margin: float = 0.25) -> None:
        self.low = np.full(3, margin)
        self.high = np.array(space_size, dtype=np.float64) - margin
        self.inputs = inputs
        self.zones = zones

    @property
    def dim(self) -> int:
        return len(SCALARS) + 3 * (self.inputs + self.zones)

    def decode(self, u: np.ndarray) -> SpringTheta:
        u = np.asarray(u, dtype=np.float64)
        if u.shape != (self.dim,):
            raise ValueError(f"theta vector must have shape ({self.dim},)")
        values = {}
        for index, (name, low, high, log) in enumerate(SCALARS):
            s = float(_sigmoid(u[index]))
            values[name] = float(np.exp(np.log(low) + s * (np.log(high) - np.log(low)))) if log else low + s * (high - low)
        points = self.low + _sigmoid(u[len(SCALARS) :].reshape(-1, 3)) * (self.high - self.low)
        return SpringTheta(
            **values, stimulus_points=_points(points[: self.inputs]), readout_centers=_points(points[self.inputs :])
        )

    def encode(self, theta: SpringTheta) -> np.ndarray:
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
