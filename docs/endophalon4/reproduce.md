# Run and inspect Endophalon 4

This published experiment is complete. On a fresh clone, first extract BOTH
checkpoint archives using the commands below before status/resume/audit. The
trainer reads expanded JSON checkpoints; it does not automatically extract tar
archives. Never start a new search for this already completed experiment.

Use the repository root in the same Codex Cloud computer. No Actions, GPU,
external training server or new Cloud task is needed. Python3.12/NumPy2.3.5,
BLAS/OMP/MKL threads1. The training script sets these before NumPy imports.
CPU quota2, memory cap8GiB, worker count2. Never change the frozen config to
extend a completed search. Existing v1/v2/v3 model files must stay unchanged.

```bash
python -m pip install -e . -r requirements.txt numpy==2.3.5
python scripts/train_endophalon4.py --segment-generations 5 --resume auto
python scripts/train_endophalon4.py --status
python scripts/train_endophalon4.py --verify-checkpoint
```

One training invocation runs one segment for each direction; call it eight times
for a new experiment. It resumes each direction separately, verifies every
atomic file, never restarts completed generations, and refuses corrupt committed
checkpoints rather than silently repeating them. Temporary incomplete writes are
ignored. A source/model/config/NumPy/environment mismatch also refuses resume.
A completed experiment produces no extra candidates when called again.

During the actual run, the measured A→B preflight result was retained as
its generation0 checkpoint without a duplicate initial candidate evaluation.
That original result is preserved in models/endophalon4/A_to_B/preflight_generation0.json.
Optimizer-only tests use synthetic scores and do not consume physical candidates.
Audits reconstruct transitions from already recorded scores without launching
new candidate physics or selecting models again.

After BOTH searches complete, exactly once:

```bash
python scripts/evaluate_endophalon4.py
python scripts/audit_endophalon4.py
python -m pytest -q
python scripts/smoke_models.py
python scripts/verify_backup_consolidation.py
python scripts/filetree_index.py update
python scripts/filetree_index.py lint
```

The evaluation script creates an exclusive confirmation-use marker after test
and long-horizon checks, before reading confirmation streams. A repeated call
refuses. Audits only rescore saved observations. Physical replay inside the
single evaluation invocation uses the same confirmation inputs solely to check
determinism; it does not introduce new confirmation data or influence selection.

## Checkpoint publication

All generation checkpoints remain local during training. For publication,
compress each direction's complete checkpoints directory into
models/endophalon4/<direction>/checkpoints.tar.gz and ignore the expanded copies
in Git. This keeps every generation, optimizer RNG and history without duplicating
large JSON files in the uploaded tree. The final model.json contains the selected
21 coupling parameters, complete fitted four-output local head and provenance.

On a fresh clone, extract the archives before status, resume, or audit:

```bash
for direction in A_to_B B_to_A; do
  tar -xzf models/endophalon4/$direction/checkpoints.tar.gz -C models/endophalon4/$direction
 done
python scripts/train_endophalon4.py --verify-checkpoint
python scripts/audit_endophalon4.py
```

Environment fingerprints are intentionally strict for resuming optimizer state.
A different Cloud CPU/platform may refuse training resume even when it can rescore
saved observations. Never edit checkpoint hashes or model weights to force a pass.
Read evaluation.json and the compressed observations for environment-independent
inspection; bitwise physical reproducibility is established on the recorded host.
