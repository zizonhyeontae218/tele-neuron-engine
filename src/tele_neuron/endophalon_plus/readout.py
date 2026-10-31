from __future__ import annotations

import numpy as np

from tele_neuron.endophalon.readout import RidgeReadout


class WideRidgeReadout(RidgeReadout):
    """Endophalon's ridge readout with a dual-form solve for wide feature sets.

    Plus produces thousands of features per sample, more than there are
    samples, so the n x n dual system is much cheaper than the primal one.
    """

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "WideRidgeReadout":
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        if not (self.alpha > 0 and features.shape[1] > features.shape[0]):
            super().fit(features, targets)
            return self
        self.mean = features.mean(axis=0)
        std = features.std(axis=0)
        self.scale = np.where(std > 1e-12, std, 1.0)
        # Standardized features are centered, so the unpenalized bias is
        # exactly the target mean and the rest is the same ridge solution.
        normalized = (features - self.mean) / self.scale
        offset = targets.mean(axis=0)
        gram = normalized @ normalized.T + self.alpha * np.eye(len(features))
        coefficients = normalized.T @ np.linalg.solve(gram, targets - offset)
        self.weights = np.vstack([coefficients, offset])
        return self
