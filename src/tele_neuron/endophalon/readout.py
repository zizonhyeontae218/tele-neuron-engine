from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class RidgeReadout:
    """Closed-form ridge regression from reservoir features to output bits."""

    alpha: float
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    weights: np.ndarray | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "RidgeReadout":
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        self.mean = features.mean(axis=0)
        std = features.std(axis=0)
        self.scale = np.where(std > 1e-12, std, 1.0)
        design = self._design(features)
        penalty = self.alpha * np.eye(design.shape[1])
        penalty[-1, -1] = 0.0  # never shrink the bias
        self.weights = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
        return self

    def scores(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("readout is not fitted")
        return self._design(np.asarray(features, dtype=np.float64)) @ self.weights

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.scores(features) >= 0.5).astype(np.int64)

    def to_payload(self) -> dict[str, Any]:
        if self.weights is None or self.mean is None or self.scale is None:
            raise RuntimeError("readout is not fitted")
        return {
            "alpha": self.alpha,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RidgeReadout":
        return cls(
            alpha=float(payload["alpha"]),
            mean=np.array(payload["mean"], dtype=np.float64),
            scale=np.array(payload["scale"], dtype=np.float64),
            weights=np.array(payload["weights"], dtype=np.float64),
        )

    def _design(self, features: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.scale is not None
        normalized = (features - self.mean) / self.scale
        return np.hstack([normalized, np.ones((len(features), 1))])
