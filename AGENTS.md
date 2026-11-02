# Tele-Neuron Engine agent instructions

Read GOAL.md, FileTree.md, docs/model_commands.md and the current model's docs.
Current names: docs/model_names.md. New 3: docs/endophalon3_alpha.md.
Historical 3T: docs/endophalon3.md and the preserved experiment reports.

## Branch roles
- master is active integrated development: the six v1/v2 model families plus 3T and new Endophalon-3, with
  ProMax8 and ProMax16 configured separately.
- archive/endophalon-v1 preserves Endophalon, Plus and Pro at the generation-1
  boundary. Keep all three importable and executable.
- archive/endophalon-v2 preserves Endophalon 2, 2 Pro and 2 ProMax8/16. Keep v1
  dependencies: v2 imports utilities and physics from those packages.
- claude/eloquent-volta-ylkb47 is a working branch, not a canonical model branch.
- agent/add-ipad-lab is a working branch; retain its standalone ipad_lab.html.
- The nine backup/2026-10-31/* branches are retired after consolidation.
  Their original heads remain as immutable tags with the same names.
- Keep master and endophalon3 aligned; retain Claude, archive and iPad branches.
  The owner authorized integration and audited cleanup of the three Endophalon 3
  experiment/preserve branches, without force pushes or history rewriting.
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

## Routing experiment record
Read GOAL.md and docs/endophalon3_routing_report.ko.md. The original
nine cards and all prior logs are immutable. New evidence belongs under
`docs/endophalon3_routing_runs/`; one corrected A config belongs alongside the
original configs. Diagnostic gated/separate heads must never be presented as
the production model. Keep 144/2/16 and the original CLI budget. Use validation
only to choose one correction; reserve streams 10000–10015 for confirmation.
Run the targeted correction checks and one full pytest suite after code changes.
Do not expand to B unless both A seeds satisfy the documented gates.

## Joint AB follow-up
The owner authorized joint A/B training and additional budget on 2026-11-01.
See docs/endophalon3_joint_plan.md for the frozen trials and stopping rules.
The historical 40-generation and separate-level scope does not govern this follow-up.
Preserve all previous cards and reports. No test-driven repeated search.

Joint trials are now complete, 0/3 accuracy gates passed. Read
docs/endophalon3_joint_report.md before any follow-up. The old and joint
model/evidence files remain immutable.

## Endophalon 3 naming and current implementation
The owner renamed the previous 3 models to 3T; keep their historical files and
family strings byte-identical. `tele_neuron.endophalon3t` is the explicit legacy
CLI alias. `tele_neuron.endophalon3_alpha` is the new Endophalon 3 (AB alpha).
Read GOAL.md and docs/endophalon3_ab_alpha_design.ko.md. Two A/B parameter/head
banks share one physical state. Preserve position, velocity, clock and RNG on
switches; record switching energy. Keep 144/2/16 active zones. No hidden input
history or per-bank state; only the evaluation oracle sees target history.
Implement/verify the bounded MVP before outer physics training.

Local index fallback (external skill generator unavailable):
`python scripts/filetree_index.py update` then `python scripts/filetree_index.py lint`.
