# Endophalon 3 — experiment closeout and integration

Preserve the fixed-budget Endophalon 3 A/B study and its negative routing follow-up alongside the 2026-11-01 backup consolidation. See `docs/endophalon3_training_report.ko.md`, `docs/endophalon3_routing_report.ko.md`, and `docs/integration/2026-11-01.md`.

- Retain 144 nodes, 2 substeps, 16 zones, single linear production readout, original splits and seeds. Do not overwrite published models, logs, configs or prior artifacts.
- Nine planned training cards and the equal-budget random baseline were completed and audited. Optional A one-hot seed 2 was not run. A achieved neither the overall/per-operation accuracy gate nor consistent slow-region recall2 improvement.
- Routing A trials on both search seeds failed the predeclared gate; B expansion stopped. The 0.5 loss remains opt-in for reproducibility, not an accepted improvement. No extra seed search or expanded budget is authorized by this closeout.
- Preserve interrupted attempts and their limits; missing final state from the interrupted cloud runs cannot be recovered. Keep existing backup consolidation and `ipad_lab.html` intact.
- Integrate the audited work into master and endophalon3 with history preserved; never force push or rewrite history. Other archive, Claude and iPad branches are outside scope.
