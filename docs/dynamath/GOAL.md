# DynaMath 1 — experiment 001

Implement five independently parameterized nonlinear laws on one persistent information-address state. Every tick adds sparse input, applies exactly one cyclic law, then checks output addresses. No physics, learned initial state, embeddings, output head or external LM.

Acceptance fixed before training: invariant tests pass; XOR and delayed recall held-out accuracy >=90%; character held-out cross entropy improves over initialization; real address-event generation produces at least one token. These gates do not imply useful conversation or GPT-2 quality.

Budget: one five-law logic run (600 Adam updates), one single-law logic reference (600), one character run (800 TBPTT updates); a debugging rerun is permitted only for a concrete defect. CPU, one thread. Keep failures. No Endophalon training, modifications or merging.

Implementation map: `src/tele_neuron/dynamath/{core,runtime,training,__main__}.py`; `tests/test_dynamath.py`; `data/dynamath/`; `configs/dynamath/`; `models/dynamath/`; `docs/dynamath/`. Add only isolated DynaMath paths and root orientation/index changes. Training and runtime snapshots carry schema, phase, state, RNG and optimizer as appropriate.
