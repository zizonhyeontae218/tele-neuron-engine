# Contributing to DynaMath

Read AGENTS.md, GOAL.md and FileTree.md, then the current versioned experiment
report. DynaMath 001a is the reproducible baseline. Preserve its frozen evidence,
source attribution and saved weights; add future versions in new versioned paths.

Install CPU PyTorch and pip install -e '.[dev]'. Run python -m pytest -q,
python scripts/audit_dynamath_001a.py, and
python scripts/verify_dynamath_001a.py --output /tmp/dynamath-replay.json.
Update and lint the file index with scripts/filetree_index.py before delivery.

For learning changes, declare goal, evaluation protocol and compute cap before
training. Save failures as well as successes; never tune on the frozen test split.
Do not merge, rewrite history, or edit other branches as part of this branch's work.
