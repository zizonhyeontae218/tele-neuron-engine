# Endophalon 3 — independent A/B models connected in software

Owner direction: run A and B separately and connect their inputs/outputs in software. Preserve every 3T/shared-state-alpha model, implementation and experiment. No further training is part of this implementation slice.

Each model owns positions, velocities, local clock, event count and RNG state. The dispatcher advances only the selected model; the other is paused unchanged. Memory tasks use that model's local input history. Explicit software registers pass actual predicted output bits into later commands (including across A/B). Do not copy physical state, fake recall in software or supply oracle targets to the runtime.

Default existing-card preset: historical A binary seed 2 (88.33% on its original test) and B binary seed 1 (74.48%). These are observed historical deployment choices, not unbiased model comparisons. There is no saved B-eight-op model at 80–90% in the baseline cards. ProMax's 91–97% figures cover only three operations. Keep all source card/config bytes and seeds intact.

First acceptance: software router, explicit result registers, build/run/checkpoint CLI; interleaved execution must equal each model's standalone local input sequence; inactive state and checkpoints unchanged; native held-out accuracy replay for both imported heads; a fresh independent local-stream check without training. Correctness of wiring does not guarantee chained model accuracy.

Design and evidence: docs/endophalon3_software.md. New code goes in endophalon3_software; the prior endophalon3_alpha shared-state runtime remains unchanged for reproduction. Keep 144 nodes / 2 substeps / 16 zones for each active model; storage now contains two reservoirs rather than one.

## Verified outcome
Implemented independent-session router, real-output registers, CLI and checkpoints.
Native card accuracy exactly preserved: A 88.33%, B 74.48%. New local-stream check:
A 87.71%, B 74.74%. Tests 120 passed; smoke 11 combinations; saved bundle audit
passed. No new training. Per-op memory weaknesses and composed-program accuracy
remain limitations; do not claim a saved B-eight-op model already reached 80–90%.
