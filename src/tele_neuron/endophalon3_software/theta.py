"""Endophalon-3 theta: Endophalon-2-Pro spring physics plus a slow region.

The network is split into a fast region and a slow region (a fixed spatial
split from the config). The slow region gets its own damping and a stiffness
multiplier, giving the physics two timescales: fast nodes near the inputs do
the current computation, slow nodes keep history (the recall1/recall2
problem). With regions disabled in the config, the slow parameters are unused
and the physics equals Endophalon-2-Pro.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tele_neuron.endophalon2_pro.theta import SCALARS as PRO_SCALARS
from tele_neuron.endophalon_pro.theta import _logit, _sigmoid


SCALARS: tuple[tuple[str, float, float, bool], ...] = PRO_SCALARS + (
    ("damping_slow", 0.80, 1.00, False),  # per-step velocity retention in the slow region
    ("stiffness_slow", 0.1, 10.0, True),  # multiplies k_linear and k_cubic on slow springs
)
NAMES = tuple(name for name, *_ in SCALARS)


@dataclass(frozen=True, slots=True)
class Theta3:
    damping: float
    k_linear: float
    k_cubic: float
    mass_spread: float
    anchor: float
    bath: float
    pulse_strength: float
    stimulus_width: float
    readout_width: float
    damping_slow: float
    stiffness_slow: float
    stimulus_points: tuple[tuple[float, float, float], ...]  # one per input channel (see isa.channel_names)
    readout_centers: tuple[tuple[float, float, float], ...]

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {name: getattr(self, name) for name in NAMES}
        payload["stimulus_points"] = [list(p) for p in self.stimulus_points]
        payload["readout_centers"] = [list(p) for p in self.readout_centers]
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Theta3":
        return cls(
            **{name: float(payload[name]) for name in NAMES},
            stimulus_points=_points(payload["stimulus_points"]),
            readout_centers=_points(payload["readout_centers"]),
        )


class Theta3Space:
    """Unconstrained u in R^d <-> valid Theta3 (sigmoid-bounded, points inside the box)."""

    def __init__(self, space_size: tuple[float, float, float], channels: int, zones: int, margin: float = 0.25) -> None:
        self.low = np.full(3, margin)
        self.high = np.array(space_size, dtype=np.float64) - margin
        self.channels, self.zones = channels, zones

    @property
    def dim(self) -> int:
        return len(SCALARS) + 3 * (self.channels + self.zones)

    def decode(self, u: np.ndarray) -> Theta3:
        u = np.asarray(u, dtype=np.float64)
        if u.shape != (self.dim,):
            raise ValueError(f"theta vector must have shape ({self.dim},)")
        values = {}
        for index, (name, low, high, log) in enumerate(SCALARS):
            s = float(_sigmoid(u[index]))
            values[name] = float(np.exp(np.log(low) + s * (np.log(high) - np.log(low)))) if log else low + s * (high - low)
        points = self.low + _sigmoid(u[len(SCALARS) :].reshape(-1, 3)) * (self.high - self.low)
        return Theta3(**values, stimulus_points=_points(points[: self.channels]), readout_centers=_points(points[self.channels :]))

    def encode(self, theta: Theta3) -> np.ndarray:
        u = []
        for name, low, high, log in SCALARS:
            value = getattr(theta, name)
            s = (np.log(value) - np.log(low)) / (np.log(high) - np.log(low)) if log else (value - low) / (high - low)
            u.append(float(_logit(s)))
        points = np.array(theta.stimulus_points + theta.readout_centers, dtype=np.float64)
        if points.shape != (self.channels + self.zones, 3):
            raise ValueError("theta needs one stimulus point per channel and one center per readout zone")
        return np.concatenate([np.array(u), _logit((points - self.low) / (self.high - self.low)).ravel()])


def _points(rows: Any) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(v) for v in row) for row in rows)  # type: ignore[misc]
