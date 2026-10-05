"""Plus physics base: a batched, backend-abstracted ball physics engine.

The base separates three ideas:

- ``WorldSpec``: what stays fixed (space, particle count and size, input points,
  time stepping, seed).
- ``Preset``: a set of physical laws played out in that world (damping,
  restitution, input strength, mass distribution).
- ``Batch``: many independent simulations, each with its own preset, input bits
  and noise realization, advanced together as one ``(B, N, 3)`` array.
"""

from tele_neuron.plus.world import Job, Preset, WorldSpec

__all__ = ["Job", "Preset", "WorldSpec"]
