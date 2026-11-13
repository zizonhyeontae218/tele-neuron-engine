# DynaMath series development

This branch is the DynaMath series home. The owner explicitly authorized removing
unrelated models and superseded experiments from this branch on 2026-11-13.
This supersedes historical keep-all instructions in archived experiment briefs.
Do not modify other branches, tags or the repository default branch; never force
push or rewrite history. Current branch: feature/dynamath-001a.

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
