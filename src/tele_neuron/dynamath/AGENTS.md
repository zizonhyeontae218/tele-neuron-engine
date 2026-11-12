# DynaMath development

The user explicitly authorizes this separate feature/dynamath-1 branch, overriding the historical root prohibition only for this work. Preserve every pre-existing model/config/source, branch and tag. Never merge into master or endophalon3.

Canonical goal: docs/dynamath/GOAL.md; original brief: docs/dynamath/handoff.md.
Run PYTHONPATH=src python -m tele_neuron.dynamath --help.
Only formula coefficients may learn; input/output maps and event policy are fixed. One formula per tick, five in cyclic order, zero initial state, additive input, no output reset.
