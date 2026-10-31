"""Opcode streams: per event, data bits (a, b) plus one of three operations."""

from __future__ import annotations

import numpy as np


OPS: tuple[str, ...] = ("xor", "recall1", "xor_mem")
CHANNELS = 2 + len(OPS)  # a, b, then one stimulus channel per opcode


def op_stream(seed: int, stream: int, events: int) -> tuple[np.ndarray, np.ndarray]:
    """(events, 2) data bits and (events,) opcode indices, both uniform at random."""
    rng = np.random.default_rng([seed, 0x4F50, stream])
    return rng.integers(0, 2, size=(events, 2)), rng.integers(0, len(OPS), size=events)


def channels(bits: np.ndarray, ops: np.ndarray) -> np.ndarray:
    """Physical input per event: [a, b, one-hot opcode] -> (events, 5)."""
    return np.hstack([bits, np.eye(len(OPS), dtype=np.int64)[ops]])


def op_targets(bits: np.ndarray, ops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Target of each event under its own opcode, and a mask of events with enough history."""
    a, b = bits[:, 0].astype(np.int64), bits[:, 1].astype(np.int64)
    previous = np.concatenate([[0], a[:-1]])
    table = np.stack([a ^ b, previous, a ^ previous])  # (ops, events)
    valid = np.arange(len(a)) >= 1
    return table[ops, np.arange(len(a))], valid


def fixed_op_targets(bits: np.ndarray, op: int) -> np.ndarray:
    """Targets if every event used operation ``op`` (for the separate-readout reference)."""
    return op_targets(bits, np.full(len(bits), op))[0]
