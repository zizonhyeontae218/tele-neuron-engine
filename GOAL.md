# DynaMath series — dedicated branch

Make feature/dynamath-001a the DynaMath series home and retain only DynaMath 001a
and files needed for its reproduction and future work. This is repository cleanup,
not a new training phase. Original experiment evidence remains historical.

Keep: one independent src/dynamath package, current Korean CLI/training/data tools,
core/runtime tests, 001a configuration, source pins/license/split/tokenizer,
four 001a checkpoints and raw metrics/samples, shared architecture, development
instructions, focused CI and generated file index. Keep 5-tick/single-law controls.

Remove: unrelated model families/apps, training archives, workflows and tests;
superseded Experiment 001 toy English training assets/code/checkpoints; old
preservation inventories and duplicate logs that refer to the removed tree.
Prior history remains reachable through the parent commit and other existing refs.

Plan: remove dependencies on the old package initializer; move the DynaMath package
to src/dynamath; retain only shared training helpers used by 001a; route
python -m dynamath and the dynamath console command to the Korean CLI; update
packaging, imports, docs and CI. Replace the old whole-tree audit with a focused
001a checkpoint/source/evidence integrity check. Keep architecture and checkpoint
schema/coefficients unchanged. No new remote branch, merge or history rewrite.

Acceptance: DynaMath-only README/branch tree; clean install without removed model
dependencies; full surviving tests pass; all 64 recorded sample replays match;
001a checkpoints/frozen data/raw experiment results byte-identical to d5127ab;
CLI generation/chat/tick restore works; index lint passes; other refs unchanged.

Current experiment goal/report: docs/dynamath/experiment-001a/{GOAL,report}.md.
Scientific limitation remains semantic evaluation 0/6 and no demonstrated recall.

## Current authorized phase: 002
Implement docs/dynamath/experiment-002/CODEX_HANDOFF.md with additional Model 9,
Model 3 local/cross-read, Model 1 controls. See experiment-002/GOAL.md and pinned
config for acceptance/budgets; previous cleanup goal above is historical.
