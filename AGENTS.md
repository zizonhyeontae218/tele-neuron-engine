# Tele-Neuron Engine agent instructions

Read GOAL.md, FileTree.md, docs/model_commands.md and the current model's docs.
Endophalon-3 (base, untrained) starts at docs/endophalon3_handoff.md.

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
- Archive preview (read-only): python scripts/archive_endophalon.py
- Finalize backups/deletions: manual Archive Endophalon branches workflow on master.
  It creates and verifies tags before deleting only the six audited model refs.
  Never bypass its changed-ref checks or branch protection.
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

## Routing experiment branch
Read the current GOAL.md and docs/endophalon3_routing_design.md. The original
nine cards and all prior logs are immutable. New evidence belongs under
`docs/endophalon3_routing_runs/`; one corrected A config belongs alongside the
original configs. Diagnostic gated/separate heads must never be presented as
the production model. Keep 144/2/16 and the original CLI budget. Use validation
only to choose one correction; reserve streams 10000–10015 for confirmation.
Run the targeted correction checks and one full pytest suite after code changes.
Do not expand to B unless both A seeds satisfy the documented gates.
