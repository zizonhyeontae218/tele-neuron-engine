from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tele_neuron.endophalon.readout import RidgeReadout
from tele_neuron.endophalon_pro.config import ProConfig
from tele_neuron.endophalon_pro.physics import Readings
from tele_neuron.plus.observe import MultiScaleObserver


@dataclass(frozen=True, slots=True)
class Fitness:
    """J(theta) and its parts, measured with the inner-optimal readout W*(theta)."""

    j: float
    l_val: float  # mean squared error of W* scores on validation realizations
    r_noise: float  # spread of W* scores across realizations of the same input
    r_stability: float  # share of balls pinned against walls at readout time
    r_complexity: float  # mean squared readout weight (standardized features)
    val_accuracy: float  # every output bit right


class CaseFeatures:
    """Features of one theta, laid out as (realization, case, feature)."""

    def __init__(self, features: np.ndarray, wall_fraction: np.ndarray, targets: np.ndarray) -> None:
        self.features = features
        self.wall_fraction = wall_fraction
        self.targets = targets

    def split(self, realizations: range) -> tuple[np.ndarray, np.ndarray]:
        x = self.features[realizations.start : realizations.stop]
        reps, cases, width = x.shape
        return x.reshape(reps * cases, width), np.tile(self.targets, (reps, 1))


def observe(config: ProConfig, readings: Readings, grid: tuple[int, int, int], realizations: int) -> np.ndarray:
    """Readings for ``realizations x cases`` jobs of one theta -> (realization, case, feature)."""
    observer = MultiScaleObserver(config.world.space_size, (grid,), speed=False)
    features = observer.observe(readings.positions, readings.velocities)
    return features.reshape(realizations, len(config.cases), -1)


def fitness(config: ProConfig, data: CaseFeatures, train: range, val: range) -> tuple[Fitness, RidgeReadout]:
    """Inner loop: solve the readout exactly on train, then score theta on val."""
    x, y = data.split(train)
    readout = RidgeReadout(alpha=config.readout.ridge_alpha).fit(x, y)
    vx, vy = data.split(val)
    scores = readout.scores(vx)
    l_val = float(np.mean((scores - vy) ** 2))
    per_case = scores.reshape(len(val), len(config.cases), -1)
    r_noise = float(per_case.std(axis=0).mean())
    r_stability = float(data.wall_fraction[train.start : val.stop].mean())
    assert readout.weights is not None
    r_complexity = float(np.mean(readout.weights[:-1] ** 2))
    spec = config.objective
    j = l_val + spec.lambda_noise * r_noise + spec.lambda_stability * r_stability + spec.lambda_complexity * r_complexity
    accuracy = float(np.all((scores >= 0.5) == vy, axis=1).mean())
    return Fitness(j, l_val, r_noise, r_stability, r_complexity, accuracy), readout


def accuracy(readout: RidgeReadout, data: CaseFeatures, realizations: range) -> float:
    x, y = data.split(realizations)
    return float(np.all(readout.predict(x) == y, axis=1).mean())
