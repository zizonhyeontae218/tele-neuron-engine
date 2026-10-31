# Endophalon 3 routing diagnosis — design

## Hypotheses
The large separate/single readout gap suggests missing conditional routing.
Per-op full-node readouts will distinguish retained physical information from
loss in the 16-zone projection. Operation decoding tests whether the opcode is
recoverable, while an explicitly gated diagnostic tests the effect of supplying
multiplicative routing outside the physics. None alone identifies a causal
mechanism or establishes a capacity ceiling.

## Procedure
1. Recollect each frozen card on train+validation only, without changing theta.
2. Fit on train only: shared/separate zones, shared/separate node displacements,
   opcode-gated zones (diagnostic), and opcode classifiers on zones/nodes.
3. Measure per-op accuracy/MSE and train→validation gaps. Preserve initial and
   trained states and all card hashes. Do not select a change using test data.
4. If physical information and opcode remain recoverable but shared losses
   sacrifice an operation, evaluate one loss-balancing correction; freeze its
   definition and coefficient before training. If evidence differs, document
   the actual bottleneck before selecting a different single correction.
5. Run A seeds 1/2, same warm start and 40×24; record failures and evaluate all
   original controls plus untouched confirmation streams 10000–10015.

## File responsibilities
| Path | Responsibility |
| --- | --- |
| scripts/diagnose_endophalon3_routing.py | Saved-state controls, validation evidence |
| docs/endophalon3_routing_runs/ | New diagnostics, CLI logs, confirmation data |
| docs/endophalon3_routing_report.ko.md | Decision, measured result and limits |
| configs/endophalon3/level_a_routing.json | New configuration; originals immutable |
| src/tele_neuron/endophalon3/core.py | At most the selected learning correction |
| tests/test_endophalon3_routing.py | Correction behavior and compatibility checks |

The selected correction is L_validation = 0.5 * pooled MSE + 0.5 * max_operation MSE, with the original noise/drift/complexity terms unchanged. The coefficient is fixed before training; no sweep is permitted. The default weight 0 preserves original selection exactly. No new readout features are deployed. The physics integrator and
instruction/data protocol are outside this change. All validation fits stay
within the existing whole-stream splits. There is no hyperparameter sweep.
