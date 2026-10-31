# Endophalon repository consolidation — 2026-10-31

Preserve the six independent model families and their history, configs, cards,
experiments and tests; consolidate generation archives and update master through
ProMax8/16 without changing scientific implementations.

- master: active integrated development; archive/endophalon-v1: generation 1;
  archive/endophalon-v2: generation 2 plus shared and generation-1 dependencies.
- Keep Claude Volta and iPad Lab work, with synchronized agent guidance.
- Preserve every original head remotely before moving existing refs.
- Validate full pytest, all model entry points and both ProMax variants.
- Delete only the six superseded model branches, only after verified backup tags.
- No force push, history rewriting, retraining overwrite, or architecture redesign.
- Evidence and remaining blockers: docs/integration/2026-10-31.md.

## Current experiment — Endophalon 3 training (2026-10-31)

On `experiment/endophalon3-training`, evaluate the existing master implementation
without architecture changes: 144 nodes, 2 substeps, 16 zones, 40 generations ×
24 population. Preserve all prior cards, configs, source and logs. Follow
`docs/endophalon3_handoff.md`: Phase 0 tests/smoke/four probes; A binary seeds
1/2 and seed-1 random search; homogeneous seeds 1/2; A one-hot seed 1; B binary
and one-hot, ideally two seeds. Audit stream splits, readout controls, regularizers,
stability and ProMax replay. A passes only with test >=90% overall and >=85%
every op; assess recall2 ablation separately. Report every seed and incomplete
run. Commit evidence and open a PR; never merge master automatically.

New artifacts: `models/endophalon3/20261031_*.json` (trained CLI cards),
`docs/endophalon3_runs/20261031-training/` (raw logs and validation evidence),
`scripts/run_endophalon3_training.py` (bounded, non-overwriting CLI runner),
and a final research report linked from `docs/endophalon3.md`.
