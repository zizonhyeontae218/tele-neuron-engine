# Tele-Neuron Engine agent instructions

Read GOAL.md, FileTree.md, docs/model_commands.md and the current model's docs.

## Branch roles
- master is active integrated development: six independent model families, with
  ProMax8 and ProMax16 configured separately.
- archive/endophalon-v1 preserves Endophalon, Plus and Pro at the generation-1
  boundary. Keep all three importable and executable.
- archive/endophalon-v2 preserves Endophalon 2, 2 Pro and 2 ProMax8/16. Keep v1
  dependencies: v2 imports utilities and physics from those packages.
- claude/eloquent-volta-ylkb47 is a working branch, not a canonical model branch.
- agent/add-ipad-lab is a working branch; retain its standalone ipad_lab.html.
- backup/2026-10-31/* are safety references to ORIGINAL heads. Do not move them.
  Old six model branches remain until backup tags are created and verified.

## Commands (repository root, Python >=3.11)
- Install: python -m pip install -e . -r requirements.txt
- Full tests: python -m pytest -q
- CLI/config smoke: python scripts/smoke_models.py
- Run/training commands: docs/model_commands.md (can take much longer than smoke).

## Invariants
Preserve source, configs, model cards, logs, task definitions, dataset splits and
seeds. Do not replace older models or overwrite published experiment outputs.
No force pushes, unrelated branch deletion or history rewriting. Respect branch
protection and inspect the remote head before updating. Archive changes must be
limited to compatibility fixes and documentation; record the evidence.
Keep README.md and README.ko.md branch links accurate. Update FileTree.md and,
when the Filetree generator is available, FileTree.md / FILETREE.hash.json.
Record actual commands, results and limitations; never infer benchmark parity
from smoke tests. No multi-agent delegation is required for this maintenance.
