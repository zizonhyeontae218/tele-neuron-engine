"""Endophalon-3 instruction set: level -> operation -> data, all carried by the input.

Every event is one instruction, written like ``"A XOR 1 1"``:

    level  A or B   which instruction table is active
    op     name     which operation inside that level
    a, b   bits     the data

An instruction is encoded into stimulus channels; each channel that is 1
pushes the physics at its own learned stimulus point. Two encodings exist:

    binary  [L, o2, o1, o0, a, b]       6 channels: level bit, 3-bit opcode, data
    onehot  [L, op0 .. op7, a, b]       11 channels: level bit, one-hot opcode, data

The opcode index of an operation is the same in both levels, so "A XOR" and
"B XOR" differ only in the level bit. Level A ops are indices 0-3; level B
adds 4-7. With ``binary``, "A XOR 0 0" pushes nothing at all (every channel
is 0); the physics must read "no push" as an instruction too. That is a
known trade-off, discussed in docs/endophalon3_handoff.md.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np


OPS: tuple[str, ...] = ("xor", "recall1", "recall2", "xor_mem", "and", "or", "parity3w", "and_mem")
LEVELS: dict[str, tuple[str, ...]] = {"A": OPS[:4], "B": OPS}
ENCODINGS = ("binary", "onehot")
MAX_LAG = 2  # recall2 / parity3w look two events back


def channel_names(encoding: str) -> tuple[str, ...]:
    if encoding == "binary":
        return ("level", "op_bit2", "op_bit1", "op_bit0", "a", "b")
    if encoding == "onehot":
        return ("level", *(f"op_{name}" for name in OPS), "a", "b")
    raise ValueError(f"encoding must be one of {ENCODINGS}")


@dataclass(frozen=True, slots=True)
class Instruction:
    level: str
    op: str
    a: int
    b: int

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"unknown level {self.level!r}; levels are {tuple(LEVELS)}")
        if self.op not in LEVELS[self.level]:
            raise ValueError(f"op {self.op!r} is not in level {self.level} ({', '.join(LEVELS[self.level])})")
        if self.a not in (0, 1) or self.b not in (0, 1):
            raise ValueError("data bits must be 0 or 1")

    @classmethod
    def parse(cls, text: str) -> "Instruction":
        """Parse ``"A XOR 1 1"``, ``"B and_mem 0 1"`` or ``"A:xor(1,1)"``."""
        tokens = [t for t in re.split(r"[\s:(),]+", text.strip()) if t]
        if len(tokens) != 4:
            raise ValueError(f"expected 'LEVEL OP a b', got {text!r}")
        level, op, a, b = tokens
        return cls(level.upper(), op.lower(), int(a), int(b))

    def encode(self, encoding: str = "binary") -> np.ndarray:
        index = OPS.index(self.op)
        level = int(self.level == "B")
        if encoding == "binary":
            return np.array([level, (index >> 2) & 1, (index >> 1) & 1, index & 1, self.a, self.b], dtype=np.int64)
        if encoding == "onehot":
            return np.array([level, *np.eye(len(OPS), dtype=np.int64)[index], self.a, self.b], dtype=np.int64)
        raise ValueError(f"encoding must be one of {ENCODINGS}")


def evaluate(ops: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ground truth of a stream: each event's own op applied to the a-history.

    ``ops`` are opcode indices into OPS. Returns targets and a mask of events
    with enough history (k >= MAX_LAG).
    """
    a, b = a.astype(np.int64), b.astype(np.int64)
    a1 = np.concatenate([[0], a[:-1]])
    a2 = np.concatenate([[0, 0], a[:-2]])
    table = np.stack([a ^ b, a1, a2, a ^ a1, a & b, a | b, a ^ a1 ^ a2, a & a1])  # rows follow OPS
    return table[ops, np.arange(len(a))], np.arange(len(a)) >= MAX_LAG


def program(lines: list[str], encoding: str = "binary") -> tuple[np.ndarray, np.ndarray]:
    """A hand-written program -> (events, channels) input and its targets."""
    instructions = [Instruction.parse(line) for line in lines]
    x = np.stack([ins.encode(encoding) for ins in instructions])
    ops = np.array([OPS.index(ins.op) for ins in instructions])
    target, _ = evaluate(ops, np.array([i.a for i in instructions]), np.array([i.b for i in instructions]))
    return x, target


@dataclass(frozen=True, slots=True)
class Stream:
    levels: np.ndarray  # (events,) 0 = A, 1 = B
    ops: np.ndarray  # (events,) opcode indices into OPS
    a: np.ndarray
    b: np.ndarray

    def channels(self, encoding: str) -> np.ndarray:
        index = self.ops
        if encoding == "binary":
            code = np.stack([(index >> 2) & 1, (index >> 1) & 1, index & 1], axis=1)
        elif encoding == "onehot":
            code = np.eye(len(OPS), dtype=np.int64)[index]
        else:
            raise ValueError(f"encoding must be one of {ENCODINGS}")
        return np.hstack([self.levels[:, None], code, self.a[:, None], self.b[:, None]]).astype(np.int64)

    def targets(self) -> tuple[np.ndarray, np.ndarray]:
        return evaluate(self.ops, self.a, self.b)


def random_stream(level: str, seed: int, stream: int, events: int) -> Stream:
    """Random instructions. ``level`` "A" or "B" fixes the level for the whole
    stream (separate A / B training). "AB" draws the level per event (joint)."""
    rng = np.random.default_rng([seed, 0x4533, stream])
    if level == "AB":
        levels = rng.integers(0, 2, events)
    elif level in LEVELS:
        levels = np.full(events, int(level == "B"))
    else:
        raise ValueError("level must be 'A', 'B' or 'AB'")
    sizes = np.where(levels == 1, len(LEVELS["B"]), len(LEVELS["A"]))
    ops = (rng.random(events) * sizes).astype(np.int64)
    return Stream(levels.astype(np.int64), ops, rng.integers(0, 2, events), rng.integers(0, 2, events))
