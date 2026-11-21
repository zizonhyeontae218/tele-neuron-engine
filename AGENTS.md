# DynaMath series development

This branch is the DynaMath series home. The owner explicitly authorized removing
unrelated models and superseded experiments from this branch on 2026-11-13.
This supersedes historical keep-all instructions in archived experiment briefs.
Do not modify other branches, tags or the repository default branch; never force
push or rewrite history. Historical branch at the time: feature/dynamath-001a.

Read GOAL.md, FileTree.md, docs/dynamath/architecture.md and the versioned goal/report.
Current baseline is DynaMath 001a, not a useful conversational model. Preserve its
checkpoint bytes, frozen split/tokenizer, raw training/evaluation results and credits.
No new training or quality-driven reruns without a separately budgeted task.

One persistent numeric information state, additive input, exactly one cyclic law
per tick and immediate output inspection. Only law coefficients learn. No trained
encoder/head, answer lookup, output reset or private state per law. SILENCE is an
ordinary fixed output address. Keep 5-tick and single-law controls reproducible.

Code: src/dynamath; tests: tests/test_dynamath*.py and tests/test_cli.py; assets remain
versioned under */dynamath/001a or configs/dynamath. Future versions get their own
goal, configuration and evidence directory. The root README indexes the series.

Install: pip install torch --index-url https://download.pytorch.org/whl/cpu;
pip install -e '.[dev]'. Commands: python -m dynamath --help; python -m pytest -q;
python scripts/verify_dynamath_001a.py --output <scratch.json>;
python scripts/audit_dynamath_001a.py; python scripts/filetree_index.py update;
python scripts/filetree_index.py lint. The repository-local index tool is the
available Filetree fallback. Never hand-edit generated indexes.

## Authorized 002 phase
This historical phase targeted feature/dynamath-002. The user's issue #12 handoff
and extra scalar-sharing controls supersede the old no-training restriction for
this bounded phase. Preserve all legacy source/assets; new implementation lives
in src/dynamath/v002. Canonical contract: docs/dynamath/experiment-002/CODEX_HANDOFF.md;
plan: docs/dynamath/experiment-002/GOAL.md; budget: configs/dynamath/experiment-002.json.

## Authorized 003b phase
This historical phase used feature/dynamath-003b, based on feature/dynamath-003 at ca8141c.
The explicit user request authorizes word-level Model1 implementation, 50,000
updates, full-split evaluation, bounded Actions continuation, and commits here.
Canonical goal: docs/dynamath/experiment-003b/GOAL.md. Historical no-training
restrictions do not apply to this fixed budget. Preserve previous experiment
bytes and other refs. New code is isolated in v003b; Config accepts legacy
strings and ordered token lists. See v003b/AGENTS.md and the versioned README.
Generated index files use union merging for concurrent evidence/source additions;
always regenerate and lint their actual hashes after a merge. This does not
change merge behavior for source, configurations, experiment results or models.

## Consolidated DynaMath research branch

The existing DynaMath branch is the base for subsequent DynaMath work. Its
history reaches each of the earlier DynaMath branch tips. The 003aa and 003ba
results have versioned evidence, checkpoint archives and reproduction notes in
docs/dynamath and models/dynamath. Use the new experiment's user-approved
protocol when starting any further training.
