from __future__ import annotations

from itertools import product
from typing import Callable


Case = tuple[tuple[int, ...], tuple[int, ...]]


def _pair(high: int, low: int) -> int:
    return high * 2 + low


def _bits(value: int, width: int) -> tuple[int, ...]:
    """Most significant bit first."""
    return tuple((value >> shift) & 1 for shift in range(width - 1, -1, -1))


def _table(arity: int, fn: Callable[[tuple[int, ...]], tuple[int, ...]]) -> tuple[Case, ...]:
    return tuple((bits, fn(bits)) for bits in product((0, 1), repeat=arity))


TASKS: dict[str, tuple[int, Callable[[tuple[int, ...]], tuple[int, ...]]]] = {
    "and": (2, lambda b: (b[0] & b[1],)),
    "or": (2, lambda b: (b[0] | b[1],)),
    "xor": (2, lambda b: (b[0] ^ b[1],)),
    "half_adder": (2, lambda b: (b[0] ^ b[1], b[0] & b[1])),
    "majority3": (3, lambda b: (int(sum(b) >= 2),)),
    "parity3": (3, lambda b: (b[0] ^ b[1] ^ b[2],)),
    "full_adder": (3, lambda b: (b[0] ^ b[1] ^ b[2], int(sum(b) >= 2))),
    "parity4": (4, lambda b: (b[0] ^ b[1] ^ b[2] ^ b[3],)),
    "adder2": (4, lambda b: _bits(_pair(b[0], b[1]) + _pair(b[2], b[3]), 3)),
    "mult2": (4, lambda b: _bits(_pair(b[0], b[1]) * _pair(b[2], b[3]), 4)),
}

OUTPUT_NAMES: dict[str, tuple[str, ...]] = {
    "half_adder": ("sum", "carry"),
    "full_adder": ("sum", "carry"),
    "adder2": ("s2", "s1", "s0"),
    "mult2": ("p3", "p2", "p1", "p0"),
}


def task_cases(name: str) -> tuple[Case, ...]:
    key = name.lower()
    if key not in TASKS:
        known = ", ".join(sorted(TASKS))
        raise ValueError(f"unknown task {name!r}; known tasks: {known}")
    arity, fn = TASKS[key]
    return _table(arity, fn)


def output_names(name: str, width: int) -> tuple[str, ...]:
    return OUTPUT_NAMES.get(name.lower(), tuple(f"out{index}" for index in range(width)))
