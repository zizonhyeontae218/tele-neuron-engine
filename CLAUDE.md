# Claude working context

Follow [AGENTS.md](AGENTS.md) for branch roles, preservation rules and commands.
`claude/eloquent-volta-ylkb47` tracks integrated work; it is not the ProMax archive.
The canonical development branch is `master`; generation snapshots live at
`archive/endophalon-v1` and `archive/endophalon-v2`.
Run `python -m pytest -q` and `python scripts/smoke_models.py` before proposing
model or configuration changes. Preserve all published cards and experiment logs.

Backup consolidation: see docs/integration/2026-11-01.md. Keep Claude and
endophalon3 branches. Historical documents are in Drive; all backup model and
reproduction files are preserved on master with immutable original-head tags.

Endophalon 3 training and the negative routing result are integrated; see
`docs/endophalon3.md` and the two experiment reports. The B routing gate failed.
The opt-in routing loss is retained for reproduction, not as an improvement.

Joint AB closeout: three fixed trials completed 120 generations each and failed
accuracy gates. See docs/endophalon3_joint_report.md. Preserve the joint cards,
raw logs and optimizer checkpoints; high-accuracy A/B remains unachieved.

Owner naming decision (2026-11-02): old Endophalon 3 experiments are now 3T;
the new AB alpha is 3. See docs/model_names.md, GOAL.md, and
docs/endophalon3_alpha.md. Preserve old raw family strings/files. New runtime
is tele_neuron.endophalon3_alpha; explicit legacy CLI is tele_neuron.endophalon3t.

Current training: owner-authorized 120 generations of new-3 paired-bank physics,
not 3T. Follow docs/endophalon3_alpha_120_plan.md. Frozen seed/budget/splits;
train-only separate heads, no held-out selection; preserve all prior artifacts.
