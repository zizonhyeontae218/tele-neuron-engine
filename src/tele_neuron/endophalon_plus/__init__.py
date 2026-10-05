"""Teleneuron-Endophalon-Plus: preset-routed reservoir on the Plus physics base.

Endophalon keeps one physics fixed and learns only a linear readout. Plus adds
half-learning: three fixed physics presets, and a learned linear gate that picks
which preset each input runs on. The presets themselves are never trained.
"""

FAMILY = "Teleneuron-Endophalon-Plus"
