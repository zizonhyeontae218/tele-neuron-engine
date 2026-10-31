# Endophalon 3 — routing bottleneck experiment

## Objective
Separate information loss from operation-selection failure, then evaluate one
training correction on level-A binary seeds 1/2 at the original 40×24 budget.

## Frozen baseline
- `experiment/endophalon3-training@24ebc6bb4133634380d7122070465e8445de014e`.
- Nine audited cards; primary A accuracy gates 0/3; maximum A test 88.33%.
- Model/log/audit files remain immutable. Baseline index closeout recovered the
  original partial cloud logs through generations 28/29; FileTree lint passed.

## Scope and invariants
- First diagnose saved states using train and validation streams only.
- Diagnostic per-op/gated/node readouts are controls, not replacement models.
- Keep 144 nodes, 2 substeps, 16 zones, one linear production readout,
  encoding=original binary, delay=12, original stream splits and physical seed.
- Select and document exactly one correction before new training. No budget or
  seed search; both seeds 1/2 must finish and be reported.
- Compare against the frozen original binary seeds 1/2. J values with different
  definitions are not directly ranked as model performance.
- The original test has been viewed. Reserve new streams 10000–10015 as a
  confirmation set; never use them to select the correction or theta.
- Expand to B only if both A seeds pass >=90% overall and >=85% every operation
  on both the original test and the confirmation streams.
- No automatic merge into master or the baseline branch.

## Deliverables and acceptance
- [x] Saved-state information/routing diagnosis with original card hashes.
- [x] One correction frozen: equal mixture of pooled/worst-op validation MSE.
- [ ] Corrected CLI, focused checks, two complete A model cards and raw logs.
- [ ] Split metrics, controls, stability and unchanged baseline evidence.
- [ ] Final report, synchronized FileTree, separate PR targeting the frozen
      baseline experiment branch (master remains untouched).
