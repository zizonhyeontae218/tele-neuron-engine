# Model entry points

Run from the repository root after `python -m pip install -e . -r requirements.txt`.
Each model has its own module, configs, tests, docs and model-card directory.

| Generation | Model | Python module | Example config |
|---|---|---|---|
| 1 | Endophalon | tele_neuron.endophalon | configs/endophalon/xor.json |
| 1 | Plus | tele_neuron.endophalon_plus | configs/endophalon_plus/full_adder.json |
| 1 | Pro | tele_neuron.endophalon_pro | configs/endophalon_pro/parity3.json |
| 2 | Endophalon 2 | tele_neuron.endophalon2 | configs/endophalon2/base.json |
| 2 | 2 Pro | tele_neuron.endophalon2_pro | configs/endophalon2_pro/regular.json |
| 2 | 2 ProMax8 | tele_neuron.endophalon2_promax | configs/endophalon2_promax/promax8.json |
| 2 | 2 ProMax16 | tele_neuron.endophalon2_promax | configs/endophalon2_promax/promax16.json |


v1/v2 rows run with `python -m <module> --config <config> --output <new-card.json>`.
Default training is a full experiment; `--help` lists actual options. Smoke uses
temporary reduced configs. Original ProMax saved-card replay: `python scripts/replay_promax.py`.

## Current Endophalon 3

Saved-run reproduction was verified with Python 3.12 and NumPy 2.3.5:
`python -m pip install numpy==2.3.5`. Keep BLAS/OMP threads at 1. Reconstructed
theta allows only 1e-14 rounding; physical replay remains bitwise and may reject
a different CPU/math implementation. See the development history for the original
CI failure and recovery; never change model weights to make an audit pass.

```bash
python -m tele_neuron.endophalon3_software --model models/endophalon3_software/current.json --program configs/endophalon3_software/example_program.txt --checkpoint-out /tmp/new-3-state.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/audit_endophalon3_independent.py docs/endophalon3_software_runs/20261102/independent120/A
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/audit_endophalon3_independent.py docs/endophalon3_software_runs/20261102/independent120/B
```

Output paths must be new. `level A/B` changes the router default, no physics step.
`OP bit bit -> register` stores actual predictions; `@register` supplies later input.
Completed training is not repeated. Training source is scripts/train_endophalon3_independent.py
with current A/B configs, fixed seeds and checkpoints; no extra search is authorized.
Old 3 variants were removed; only experiment history remains in endophalon3_development.md.
