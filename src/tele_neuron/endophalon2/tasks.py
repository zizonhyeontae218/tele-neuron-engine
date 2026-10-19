"""Streaming tasks over two input channels a_k, b_k (one bit pair per pulse).

Each task maps the pulse history to a target per pulse. Pulses before the task
has enough history (k < lag) are masked out.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


TaskFn = Callable[[np.ndarray], np.ndarray]  # (P, 2) bits -> (P,) targets


def _shift(x: np.ndarray, lag: int) -> np.ndarray:
    out = np.zeros_like(x)
    if lag == 0:
        return x.copy()
    out[lag:] = x[:-lag]
    return out


STREAM_TASKS: dict[str, tuple[int, TaskFn, str]] = {
    # name: (lag, fn, question)
    "and": (0, lambda s: s[:, 0] & s[:, 1], "Q1 sanity: linear-separable, current pulse"),
    "xor": (0, lambda s: s[:, 0] ^ s[:, 1], "Q1: nonlinear on the current pulse"),
    "recall1": (1, lambda s: _shift(s[:, 0], 1), "Q2: memory of the previous pulse"),
    "recall2": (2, lambda s: _shift(s[:, 0], 2), "Q2: memory two pulses back"),
    "recall3": (3, lambda s: _shift(s[:, 0], 3), "Q2: memory three pulses back"),
    "xor_mem": (1, lambda s: s[:, 0] ^ _shift(s[:, 0], 1), "Q3: memory + nonlinear"),
    "parity3w": (2, lambda s: s[:, 0] ^ _shift(s[:, 0], 1) ^ _shift(s[:, 0], 2), "Q3: 3-pulse parity"),
}


def stream_bits(seed: int, stream: int, pulses: int, inputs: int = 2) -> np.ndarray:
    return np.random.default_rng([seed, 0x5354, stream]).integers(0, 2, size=(pulses, inputs))


def task_targets(name: str, bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(P, 2) bits -> (P,) targets and a (P,) mask of pulses that have enough history."""
    if name not in STREAM_TASKS:
        raise ValueError(f"unknown stream task {name!r}; known: {', '.join(STREAM_TASKS)}")
    lag, fn, _ = STREAM_TASKS[name]
    valid = np.arange(len(bits)) >= lag
    return fn(bits.astype(np.int64)), valid


def history_features(bits: np.ndarray, depth: int = 3) -> np.ndarray:
    """Physics-free baseline features: the last ``depth`` bit pairs, (P, 2 * depth)."""
    return np.hstack([_shift(bits, lag) for lag in range(depth)]).astype(np.float64)
