from __future__ import annotations

from tele_neuron.endophalon import tasks as base


def _pair(high: int, low: int) -> int:
    return high * 2 + low


def _bits(value: int, width: int) -> tuple[int, ...]:
    """Most significant bit first."""
    return tuple((value >> shift) & 1 for shift in range(width - 1, -1, -1))


# Endophalon's tasks plus the 4-input ones used by the Plus experiments.
TASKS = {
    **base.TASKS,
    "parity4": (4, lambda b: (b[0] ^ b[1] ^ b[2] ^ b[3],)),
    "adder2": (4, lambda b: _bits(_pair(b[0], b[1]) + _pair(b[2], b[3]), 3)),
    "mult2": (4, lambda b: _bits(_pair(b[0], b[1]) * _pair(b[2], b[3]), 4)),
}

OUTPUT_NAMES = {
    **base.OUTPUT_NAMES,
    "adder2": ("s2", "s1", "s0"),
    "mult2": ("p3", "p2", "p1", "p0"),
}


def task_cases(name: str) -> tuple[base.Case, ...]:
    key = name.lower()
    if key not in TASKS:
        raise ValueError(f"unknown task {name!r}; known tasks: {', '.join(sorted(TASKS))}")
    arity, fn = TASKS[key]
    return base._table(arity, fn)


def output_names(name: str, width: int) -> tuple[str, ...]:
    return OUTPUT_NAMES.get(name.lower(), tuple(f"out{index}" for index in range(width)))
