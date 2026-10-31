"""Teleneuron-Endophalon-2-ProMax: one physics, one readout, the operation chosen by input.

Endophalon-2-Pro gave each task its own readout. In ProMax, every event
carries data bits (a, b) and an opcode that picks the operation for that
event: xor (a_k XOR b_k), recall1 (a_{k-1}) or xor_mem (a_k XOR a_{k-1}).
The opcode can change on every event. It pushes the spring network at its
own stimulus point, and **one** linear readout must produce the right answer
for whichever operation was requested. The routing has to happen in the
physics.
"""

FAMILY = "Teleneuron-Endophalon-2-ProMax"


def variant_name(zones: int) -> str:
    """Models are named by readout size: ProMax8 has 8 readout zones, ProMax16 has 16."""
    return f"{FAMILY}{zones}"
