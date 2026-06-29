from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class BallState:
    id: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    mass: float
    radius: float

    def __post_init__(self) -> None:
        if self.mass <= 0:
            raise ValueError("mass must be positive")
        if self.radius <= 0:
            raise ValueError("radius must be positive")

    @classmethod
    def from_vectors(
        cls,
        *,
        id: int,
        position: np.ndarray,
        velocity: np.ndarray,
        mass: float,
        radius: float,
    ) -> "BallState":
        if position.shape != (3,):
            raise ValueError("position must have shape (3,)")
        if velocity.shape != (3,):
            raise ValueError("velocity must have shape (3,)")

        return cls(
            id=id,
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
            vx=float(velocity[0]),
            vy=float(velocity[1]),
            vz=float(velocity[2]),
            mass=float(mass),
            radius=float(radius),
        )

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @position.setter
    def position(self, value: np.ndarray) -> None:
        if value.shape != (3,):
            raise ValueError("position must have shape (3,)")
        self.x = float(value[0])
        self.y = float(value[1])
        self.z = float(value[2])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz], dtype=np.float64)

    @velocity.setter
    def velocity(self, value: np.ndarray) -> None:
        if value.shape != (3,):
            raise ValueError("velocity must have shape (3,)")
        self.vx = float(value[0])
        self.vy = float(value[1])
        self.vz = float(value[2])

    def copy(self) -> "BallState":
        return BallState(
            id=self.id,
            x=self.x,
            y=self.y,
            z=self.z,
            vx=self.vx,
            vy=self.vy,
            vz=self.vz,
            mass=self.mass,
            radius=self.radius,
        )
