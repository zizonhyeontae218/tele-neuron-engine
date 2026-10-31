"""Teleneuron-Endophalon-3: an instruction-driven physical computer (base, untrained).

One spring network (Endophalon-2-Pro physics, 144 nodes, 2 substeps) and ONE
linear readout. Every input event is an instruction ``LEVEL OP a b``: the
level bit picks the instruction table (A: 4 ops, B: 8 ops), the opcode picks
the operation inside it, and a, b are the data. The physics gains a slow
region (its own damping and stiffness) for history, aimed at recall1/recall2.

Status: base only. Nothing has been trained. See docs/endophalon3.md (design)
and docs/endophalon3_handoff.md (training method, know-how, run order).
"""

FAMILY = "Teleneuron-Endophalon-3"


def variant_name(level: str) -> str:
    """Level A and level B are trained separately: Endophalon-3-A, Endophalon-3-B (joint: -AB)."""
    return f"{FAMILY}-{level}"
