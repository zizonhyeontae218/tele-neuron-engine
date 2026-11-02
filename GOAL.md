# Endophalon 3 — separate A/B modules, one continuous physical state

The owner renames the previous Endophalon 3 family (including separate A/B and single-head joint trials) **3T**. The new command-selected AB alpha is **Endophalon 3**. Historical source paths, cards and evidence remain intact; naming aliases identify them as 3T.

Implement two independently configured A/B modules sharing positions, velocities, clock and RNG. An instruction chooses the input/physics/observation profile and its own linear head. Keep 144 nodes, 2 substeps, delay 12, regular gap 16 and 16 active zones per event. No reset or hidden per-bank state; global input history spans mode switches. Parameter switching holds positions/velocities fixed and records the resulting mechanical-energy change, without claiming energy conservation.

Acceptance: executable persistent session and CLI, explicit level selection, checkpoint/restart equivalence, fixed-level equivalence with the legacy engine, shared-memory/reset controls and balanced level/operation metrics. Preserve originals, report negative results, fit A/B heads on train only, and freeze independent evaluation streams. No long outer search or seed selection in the first implementation slice.

Design: docs/endophalon3_ab_alpha_design.ko.md. Planned files: new endophalon3_alpha session/model/CLI, endophalon3t naming alias, focused tests, bounded evaluation script, separate model/evidence directories, README and command/name documentation. Further physics training follows measured MVP results; accuracy goals remain >=90% per level and >=85% per operation, not assumed achievements.

## MVP outcome
Implemented session, dispatch, two-bank bundle, CLI and checkpointing. Fixed-level
compatibility and continuity checks pass. One frozen-physics/two-head-fit
experiment reaches 67.19% fresh pooled accuracy (A 64.57%, B 69.86%); targets
remain unmet. See docs/endophalon3_alpha.md for controls, evidence and limits.
Next milestone is transition-aware bank-physics adaptation under a frozen budget;
no additional search has been run in this implementation slice.

## Owner-authorized 120-generation run
Train both bank theta vectors with separate train-only Ridge heads over shared
transition state. Exact scope, source/start fingerprints, 2,881-pair budget and
fresh confirmation IDs are frozen in docs/endophalon3_alpha_120_plan.md.
Previous models, original MVP observations and 3T evidence stay immutable.

## 120-generation outcome
Completed 120 generations / 2,881 pairs; saved-model audit passed. Confirmation
improved 65.44% to 75.36% (A 77.77%, B 73.07%); accuracy gates failed. The
automatic closeout path error was repaired without retraining or artifact changes.
See docs/endophalon3_alpha_120_report.md and docs/endophalon3_alpha_120_closeout.md.
No extra search or new seed is authorized by this closeout.
