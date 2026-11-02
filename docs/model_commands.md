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
| 3T | 3T-A (historical) | tele_neuron.endophalon3t | configs/endophalon3/level_a.json |
| 3T | 3T-B (historical) | tele_neuron.endophalon3t | configs/endophalon3/level_b.json |

Train/evaluate with `python -m <module> --config <config>` using any row above.
Add `--output <new-path.json>` to save a new card. Do not overwrite original cards.
Training defaults are full experiments, not smoke workloads. `--help` lists each
module's actual options. ProMax8 and ProMax16 are configurations of the same
module, not separate Python packages.

`python -m pytest -q` runs all tests present in the current branch.
`python scripts/smoke_models.py` imports each available family and executes all
nine CLI/config combinations on reduced temporary configurations (three on v1).
This proves execution and serialization, not historical benchmark reproduction.

On master and archive/endophalon-v2, replay the original four ProMax cards with
`python scripts/replay_promax.py`. This uses saved theta/readout and the original
held-out streams, without retraining or overwriting any card.

Endophalon-3 training results: [endophalon3.md](endophalon3.md). Add `--probe` for the unchanged untrained
capacity probe. Training method, run order and know-how:
[endophalon3_handoff.md](endophalon3_handoff.md).

The routing loss in `configs/endophalon3/level_a_routing.json` is an opt-in negative experiment, not an improved default; see [routing report](endophalon3_routing_report.ko.md). Saved-card audits and reproduction procedures are documented in the [training report](endophalon3_training_report.ko.md).

Joint A/B cards and optimizer checkpoints are preserved separately. Read-only verification: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/audit_endophalon3_joint.py`. Training commands in the [frozen joint plan](endophalon3_joint_plan.md) reproduce the bounded trials; completed output paths are rejected. See [actual results](endophalon3_joint_report.md) before launching any new run.

## Endophalon 3 and 3T

See [name mapping](model_names.md). The preceding Endophalon-3 experiment descriptions refer to the currently named 3T models and preserve their historical paths.

New 3 (separate A/B, one continuous session):

```bash
python -m tele_neuron.endophalon3_alpha --config configs/endophalon3_alpha/from_3t_seed1.json --build /tmp/new-e3-banks.json
python -m tele_neuron.endophalon3_alpha --model /tmp/new-e3-banks.json --program configs/endophalon3_alpha/example_program.txt --output /tmp/new-e3-output.json --checkpoint-out /tmp/new-e3-session.json
# Continue with a NEW program; do not repeat the earlier program on the restored state.
python -m tele_neuron.endophalon3_alpha --model /tmp/new-e3-banks.json --program /path/to/next_program.txt --checkpoint-in /tmp/new-e3-session.json --checkpoint-out /tmp/new-e3-session-next.json
```

`level A` / `level B` selects the module for subsequent `OP a b` commands without advancing the physical clock. Full `A OP a b` / `B OP a b` syntax is also supported. Every data event advances 16 steps, observing 16 zones at delay 12; the initial 32-step warmup runs once. Output is the head score/thresholded bit, not ground truth. The first two events lack full recall2 history. Runtime never stores an oracle bit history. The bank-parameter switch energy is reported in each event.

The bounded initial readout fit/evaluation command is in [the alpha experiment plan](endophalon3_alpha_experiment_plan.md); it rejects existing output directories. This is not a long outer physics training run.

## New 3: paired-bank physics training, 120 generations

[Fixed plan](endophalon3_alpha_120_plan.md). After installation:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/train_endophalon3_alpha.py --output-dir docs/endophalon3_alpha_runs/20261102/physics120 --workers 4
```

This optimizes both bank theta vectors together, fits two separate heads on train only and uses one shared physical state. The budget is 120×24+1=2,881 candidate **pairs**. A partial matching checkpoint resumes deterministically; completed results are never overwritten. The paired-bank workflow runs this bounded workload and audits, reports and publishes the outputs with changed-head guards. A report link is added after completion.

The 2026-11-02 physics120 run is complete. Use
`models/endophalon3_alpha/20261102_physics120_seed1_binary.json` with the new-3 CLI
for the trained bundle. Do not rerun the completed budget. See
[endophalon3_alpha_120_closeout.md](endophalon3_alpha_120_closeout.md) for the
original successful audit and later closeout error recovery.

## Current independent A/B software connection

```bash
python -m tele_neuron.endophalon3_software --config configs/endophalon3_software/existing_best_observed.json --build /tmp/new-e3-software.json
python -m tele_neuron.endophalon3_software --model /tmp/new-e3-software.json --program configs/endophalon3_software/example_program.txt --output /tmp/new-e3-software-output.json --checkpoint-out /tmp/new-e3-software-state.json
# Resume with new commands, preserving both local sessions and output registers.
python -m tele_neuron.endophalon3_software --model /tmp/new-e3-software.json --program /path/to/next.txt --checkpoint-in /tmp/new-e3-software-state.json --checkpoint-out /tmp/new-e3-software-next.json
```

Use new output paths. Syntax: `[A|B] OP 0|1|@register 0|1|@register [-> register]`; `level A/B` changes the default without advancing physics. Inactive models freeze; each readout's recall uses its own local input history. Software registers are output values, not an oracle. See [actual model scores and limitations](endophalon3_software.md). No shared-state or 3T model file is overwritten.
