"""Teleneuron-Endophalon: a fixed physical reservoir read by a trained linear readout.

Endophalon is a separate model family from TeleNeuron-001, not its successor.
001 searches the physics itself (seed, masses) so that one readout zone answers
the task. Endophalon keeps the physics fixed, observes how particles are spread
across voxels at several moments, and fits a closed-form ridge readout on top.
"""

FAMILY = "Teleneuron-Endophalon"
