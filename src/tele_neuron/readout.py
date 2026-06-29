from __future__ import annotations

import numpy as np

from tele_neuron.config import ExperimentConfig, ReadoutZoneConfig


def count_in_zone(positions: np.ndarray, zone: ReadoutZoneConfig) -> int:
    center = np.array(zone.center, dtype=np.float64)
    half_extents = np.array(zone.half_extents, dtype=np.float64)
    inside = np.all(np.abs(positions - center) <= half_extents, axis=1)
    return int(np.count_nonzero(inside))


def readout_bits(positions: np.ndarray, config: ExperimentConfig) -> tuple[int, ...]:
    return tuple(
        1 if count_in_zone(positions, zone) >= zone.threshold else 0
        for zone in config.readout
    )


def readout_counts(positions: np.ndarray, config: ExperimentConfig) -> tuple[int, ...]:
    return tuple(count_in_zone(positions, zone) for zone in config.readout)
