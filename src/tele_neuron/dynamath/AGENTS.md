# DynaMath development

For feature/dynamath-001a use docs/dynamath/experiment-001a/GOAL.md.
One-tick events are the new default; io_ticks=5 is explicit legacy compatibility.
Korean implementation is in korean_data.py and korean.py. Only law coefficients
learn; SILENCE is a fixed mapped output address, never a mode mask.

The user explicitly authorizes this separate feature/dynamath-1 branch, overriding the historical root prohibition only for this work. Preserve every pre-existing model/config/source, branch and tag. Never merge into master or endophalon3.

Canonical goal: docs/dynamath/GOAL.md; original brief: docs/dynamath/handoff.md.
Run PYTHONPATH=src python -m tele_neuron.dynamath --help.
Only formula coefficients may learn; input/output maps and event policy are fixed. One formula per tick, five in cyclic order, zero initial state, additive input, no output reset.
