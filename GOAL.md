# Endophalon 4 — cloud-native resumable physical memory coupling

Continue frozen Endophalon 3 A/B (144 nodes, 2 substeps, 16 local zones,
interval16/readout12). Train two independent unidirectional coupling searches,
40 generations ×24 plus generation0=961 candidates each. Preserve all old models
and evidence. CPU/NumPy only in this Cloud workspace, 5-generation segments with
atomic generation checkpoints. No extra seeds, generations, GPU, Actions or servers.

Acceptance: each direction pooled lag1/4/8/16 confirmation accuracy >=75%,
>=10 percentage points over synchronized uncoupled; frozen existing-operation
accuracy drop <=3 points overall; finite physics; deterministic saved-model replay.
Report every lag separately. Fixed budget completion is distinct from scientific
success. Confirmation used once only after both searches and model selection.

Artifacts and exact protocol: docs/endophalon4/design.md,
configs/endophalon4/experiment.json. Resume/status/verify scripts required.
Final audit/tests before one commit and direct nonforce push to endophalon3 only;
no PR/new branch or changes to master. Previous goal preserved in this version's docs.

## Measured completion
Both 40-generation/961-candidate budgets and all eight segments completed.
Confirmation A→B 56.88% vs uncoupled49.27%; B→A59.64% vs50.20%.
Primary scientific gates failed: pooled75%, +10pp and original-operation
preservation. Long-horizon finite and saved physical/head replay passed.
Audit, pytest94, smoke8 and historical-file preservation passed. See
docs/endophalon4/report.md for per-lag results, uncertainty and limitations.
No extra search or next-version phase was started.
