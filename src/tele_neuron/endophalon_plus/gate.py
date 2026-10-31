from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class LinearGate:
    """Picks a preset with ``argmax(W @ [bits, 1])``.

    A linear gate can only cut the input cube with flat boundaries, so it cannot
    compute parity on its own; it can route, not answer.
    """

    weights: np.ndarray  # (presets, inputs + 1)

    @classmethod
    def constant(cls, preset: int, presets: int, inputs: int) -> "LinearGate":
        weights = np.zeros((presets, inputs + 1))
        weights[preset, -1] = 1.0
        return cls(weights)

    def choose(self, bits: tuple[int, ...]) -> int:
        scores = self.weights @ np.append(np.array(bits, dtype=np.float64), 1.0)
        return int(np.argmax(scores))

    def assignment(self, cases: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]) -> tuple[int, ...]:
        return tuple(self.choose(bits) for bits, _ in cases)

    def to_payload(self) -> dict[str, Any]:
        return {"weights": self.weights.tolist()}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "LinearGate":
        return cls(np.array(payload["weights"], dtype=np.float64))
