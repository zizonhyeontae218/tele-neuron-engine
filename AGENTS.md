# Tele-Neuron Engine agent instructions

Read GOAL.md, FileTree.md, docs/model_commands.md and the current model's docs.
Endophalon-3 (base, untrained) starts at docs/endophalon3_handoff.md.

## Branch roles
- master is active integrated development: the six v1/v2 model families plus Endophalon-3, with
  ProMax8 and ProMax16 configured separately.
- archive/endophalon-v1 preserves Endophalon, Plus and Pro at the generation-1
  boundary. Keep all three importable and executable.
- archive/endophalon-v2 preserves Endophalon 2, 2 Pro and 2 ProMax8/16. Keep v1
  dependencies: v2 imports utilities and physics from those packages.
- claude/eloquent-volta-ylkb47 is a working branch, not a canonical model branch.
- agent/add-ipad-lab is a working branch; retain its standalone ipad_lab.html.
- The nine backup/2026-10-31/* branches are retired after consolidation.
  Their original heads remain as immutable tags with the same names.
- Keep claude/eloquent-volta-ylkb47 and endophalon3, and all experiment/preserve
  branches unchanged. The 2026-11-01 owner decision authorizes only the nine
  audited backup-branch deletions; it does not authorize experiment PR merges.
- Historical design documents/README versions live in the verified Drive archive
  linked from docs/integration/2026-11-01.md; reproduction files remain on master.

## Commands (repository root, Python >=3.11)
- Install: python -m pip install -e . -r requirements.txt
- Archive preview (read-only): python scripts/archive_endophalon.py
- The manual Archive Endophalon branches workflow is legacy and idempotent.
  It checks preservation tags after safety-branch retirement, and only ever
  deletes the six original model refs in its fixed allowlist.
  Never bypass its changed-ref checks or branch protection.
- Full tests: python -m pytest -q
- Retire the nine consolidated backups: python scripts/retire_backup_refs.py
  previews; --apply is authorized only for this fixed audited 2026-11-01 scope.
- Consolidation integrity: python scripts/verify_backup_consolidation.py
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
