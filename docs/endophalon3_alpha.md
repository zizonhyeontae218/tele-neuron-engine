# Endophalon 3 (AB alpha) — implementation and first measurements

2026-11-02: the owner named the previous Endophalon 3 experiments **3T** and the new separate-bank AB alpha **3**. [Name mapping](model_names.md) preserves historical paths and raw card family strings. The new format and CLI are separate from the previous single-head family.

## Implemented behavior

Two complete independently configured A/B banks share one continuous positions/velocities/clock/RNG state. The command dispatcher selects the bank's input points, physical parameters, active 16-zone observation and its own train-fitted linear head. There is no per-operation head, per-bank saved physical state, input-history buffer or ground-truth memory supplied to the readout. Global data-event history spans mode switches; `level A/B` alone changes no physical state, time, RNG or event count. Initial jitter/warmup run only once; gap 16, delay 12, 144 nodes and 2 substeps remain.

Parameter switching holds positions/velocities fixed. Stored energy may jump because mass, stiffness or anchor settings change; that external-control energy change is recorded per event. No energy/momentum conservation claim is made. Each bank stores 16 observation centers and a head, so there are two stored observation/readout parameter sets, with only 16 zones evaluated per event.

Commands accept full `A OP a b`/`B OP a b` or `level A/B` followed by `OP a b`. Invalid commands are rejected before physical mutation. A saved session carries state, clock, active/selected modules and complete RNG states, tied to an exact model fingerprint. Restoring with another bundle is rejected. Non-finite integration fails explicitly. Output and checkpoint paths never overwrite existing files.

## Frozen first experiment

This is a bounded MVP, with **no outer physics search**. Exactly one original binary seed-1 A/B pair is used. Existing physics is frozen; A/B heads are separately refitted once on that level's events within the same training streams. Original heads and a pooled-head control are measured on the same physical observations. This pooled-head control is not the old joint-theta 3T model. [Plan](endophalon3_alpha_experiment_plan.md).

New AB stream IDs: train 30000–30023, validation 30024–30031, test 30032–30063, confirmation 30064–30127. Physical/data seed 1; 64 events, washout 4, split by whole stream. No seed/configuration/head-coefficient selection from test or confirmation.

| Fresh confirmation variant | Overall | A | B |
| --- | --- | --- | --- |
| original_heads | 65.60% | 60.80% | 70.49% |
| two_fitted_heads | 67.19% | 64.57% | 69.86% |
| pooled_head_control | 62.81% | 61.63% | 64.02% |
| same_heads_reset_at_switch | 63.72% | 58.48% | 69.07% |

The fitted two-head model remains below >=90% per level / >=85% per operation. Simply switching previously trained complete profiles does not preserve their fixed-level accuracy. Per-level/op and first/second-after-switch metrics, counts and negative outcomes remain in `docs/endophalon3_alpha_runs/20261102/seed1_binary/evaluation.json`. This implementation establishes a runnable separate-bank system; the high-accuracy model remains an unmet research goal.

## Shared-memory control

The reset control changes only positions/velocities to the original jitter state at each module transition. It keeps the same commands, fitted heads, elapsed clock and RNG progression; no extra warmup. Confirmation point estimates:

| Operation | Continuous state | Reset at switch | Difference |
| --- | --- | --- | --- |
| A recall1 | 80.53% | 66.80% | +13.73 pp |
| A recall2 | 62.37% | 57.42% | +4.95 pp |
| B recall1 | 64.03% | 56.92% | +7.11 pp |
| B recall2 | 58.16% | 50.63% | +7.53 pp |

These point estimates support a useful state-history contribution in this setting. They are not confidence intervals or proof that all profile switches preserve useful memory. Other operations still fail the target. This one-pair/one-data-seed experiment does not establish performance rankings or generalization. Observation is limited to 64-event streams; no infinite-horizon stability claim. The largest measured parameter-switch mechanical-energy jump was 0.40250055 in simulation units. Warmup/noise and switches are controlled simulation operations, not a closed conservative system.

## Verification and artifacts

- Fixed-level A and B observations and pre-event kinetic energies exactly match the preserved engine for tested streams; head scores match to 1e-13 and thresholded predictions exactly. Initial bitwise score checks exposed a maximum 4.44e-16 difference from one-row versus batched dot products; physical observations remain exact.
- State/time/RNG unchanged by selection-only or invalid commands; checkpoint continuation, model mismatch guards, explicit head selection and cross-level dependence tests pass.
- Full suite: **107 passed**, including CLI build/program/restore, explicit 3T serialization and mixed-bank batch-versus-individual equivalence.
- CLI smoke: 10 model/config combinations, including new 3 and the explicit 3T alias.
- Saved bundle audit: artifact hashes, disjoint splits, train-only head mean/scale/weights, all split metrics and exact physical replay of three fresh streams pass.
- Historical scientific source/configs/cards/logs remain byte-identical; current naming/documentation is updated without rewriting the originals.

Model: `models/endophalon3_alpha/20261102_seed1_binary.json`, an exact copy of the experiment bundle. Original observations, reset observations, model, evaluation and SHA-256 manifest: `docs/endophalon3_alpha_runs/20261102/seed1_binary/`.

Read-only audit after installation:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/audit_endophalon3_alpha.py docs/endophalon3_alpha_runs/20261102/seed1_binary
```

For an interactive program, use `python -m tele_neuron.endophalon3_alpha --model models/endophalon3_alpha/20261102_seed1_binary.json --program configs/endophalon3_alpha/example_program.txt`. [More commands](model_commands.md). Observed confirmation IDs are now used data and cannot be relabeled as untouched for an adaptive follow-up.

The next research step is adapting the bank physics to transitions while retaining separate heads, after inspecting validation transition/op failures and freezing another finite training budget. No further physical retraining or successful accuracy gate is implied by the current bundle. The old 3T results remain available as baselines.

Repository file index: the external skill generator was unavailable; `python scripts/filetree_index.py update` and `lint` provide a reproducible repository-local fallback using real tracked/unignored paths and actual SHA-256 values. This is separate from the unavailable external skill CLI.
