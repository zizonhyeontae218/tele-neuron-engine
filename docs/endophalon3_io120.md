# Endophalon 3 A/B I/O-only training

Start each independent module at its current published theta. Keep the spring dynamics, masses, damping, readout width, timing, 144 nodes, two substeps, and 16 zones fixed. Search the locations of active stimulus channels, shared stimulus strength and width, and 16 readout center positions. Refit one ordinary Ridge readout per candidate using the training streams only.

Each model gets 120 generations with 24 candidates per generation and one initial candidate. Rank candidates by the original validation objective plus 0.5 times the mean validation MSE of A recall2 or B recall2, xor_mem and parity3w. Use the existing train and validation streams. Evaluate the saved production module and the selected candidate on the same 64 fresh confirmation streams per level after selection. Report per-operation accuracy, the A90/B80 overall targets, and the strict overall90/all-ops85 gate.

The GitHub Actions matrix stores the models, optimizer checkpoints, logs and paired observations as artifacts. Active model cards are promoted only through a separate reviewed change.
