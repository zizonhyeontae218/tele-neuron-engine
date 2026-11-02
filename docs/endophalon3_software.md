# Endophalon 3 — independent A/B models with software connections

Implemented and verified, 2026-11-02 owner follow-up. A and B run as independent saved models. Software selects a model and explicitly connects predicted outputs to later inputs. Previous 3T and shared-state alpha experiments, including the completed 120-generation run, remain intact.

## Contracts

- Two independent physical states, local clocks, RNGs and histories. Only the selected model advances; the other pauses, without reset or idle physics steps. Each retains its own 144 nodes, 2 substeps and 16-zone readout. This stores two reservoirs, not one shared reservoir.
- `A OP a b` / `B OP a b` dispatch a data event. `level A/B` selects the default for `OP a b` without advancing either model.
- `A xor 1 0 -> r0` stores the actual thresholded model prediction as register r0. `B and @r0 1 -> r1` resolves that stored bit before input encoding. Registers hold outputs only; no software task oracle or hidden physical-history transfer.
- Recall1/recall2 refer to the selected model's own previous data inputs. Cross-model memory is explicit register dataflow. Merely switching A/B does not make one model read the other's input history.
- Invalid commands or unknown registers fail before advancing either session or changing registers. Register identifiers are explicit. Partial physics failures remain failed sessions; they are not turned into fabricated output bits.
- Checkpoints bind both local states, selected level and output registers to the exact model bundle. New model/result/checkpoint paths never overwrite originals.

## Existing models and accuracy

The default preset imports A binary seed 2 (`20261031_a_binary_s2.json`) and B binary seed 1 (`20261031_b_binary_s1.json`), unchanged theta/readout/config/seed. Historical held-out scores: A 88.33%, B 74.48%. These are the highest observed separate-level baseline scores, chosen for deployment from already viewed results. They are not an unbiased architecture ranking. A has 4 ops; B has 8. ProMax's 91–97% figures describe a different 3-op task and cannot stand in for B's eight-op accuracy. Wiring in software does not create a new 90/80 success result.

## Implementation/verification plan

New package `src/tele_neuron/endophalon3_software/` contains a thin router around two production Sessions. The existing engine remains unchanged. Bundle schema `endophalon3-software/v1` identifies independent sessions; checkpoints use their own schema and preserve each model fingerprint. Example commands/preset belong in `configs/endophalon3_software/`.

Tests cover identical interleaved-versus-standalone results, paused state/time/RNG, actual-output register transfer (including wrong predictions), malformed commands, and checkpoint restoration. Saved-card checks replay each original native test independently. Fresh local streams 50000–50031 per model are frozen here before evaluation; no theta/readout refit, seed search, or test-driven training occurs. They evaluate each imported model's own local task, not a previously shared global-history task. Chained accuracy depends on upstream prediction errors and input distribution; do not multiply marginal accuracies as if errors were independent.

## Measured implementation outcome

No new physics or head fitting was performed. Both native saved-card tests replay **exactly**, overall and per operation: A 88.33%, B 74.48%. These are not a newly trained A>=90% / B>=80% pair. Fresh model-local streams 50000–50031, with calls alternating A[k],B[k], gave **A 87.71% and B 74.74%**. Each task uses local input history, so this is not the previous global-history shared-state task and cannot be used as a direct improvement claim over its 75.36% result.

The remaining fresh recall2 scores were A 62.36%, B 62.93%; B xor_mem was 57.69% and parity3w 57.32%. High current-task scores coexist with these limits. No post-hoc retraining, alternate seed or operation-set substitution was used. Choosing cards from previously observed native results is a deployment decision; the new check is conditional on that choice.

Tests: **120 passed**. CLI smoke: **11** model/config combinations. Tests verify that interleaving equals each standalone local input stream, inactive positions/velocity/time/RNG never change, actual predicted bits (including deliberately wrong predictions) are wired to the other model, invalid commands do not mutate state, and checkpoints resume both contexts/registers. An initial test caught only stale B selected-level metadata; dispatch now updates that model's selection after successful execution. Physical outputs/state were unaffected by that correction.

Serialized model audit verifies exact imported theta/readouts against the original cards, bundle/observation hashes and native/fresh per-op accuracies. Canonical bundle: `models/endophalon3_software/20261102_existing_cards.json`. Evidence: `docs/endophalon3_software_runs/20261102/existing_cards/` (observations, evaluation, model, hashes, verification). Prior 3T and alpha source/cards/evidence are preserved.

```bash
python -m tele_neuron.endophalon3_software --model models/endophalon3_software/20261102_existing_cards.json --program configs/endophalon3_software/example_program.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/audit_endophalon3_software.py docs/endophalon3_software_runs/20261102/existing_cards
```

The short example demonstrates wiring, not a score benchmark; each model initially has insufficient recall2 history and its own warmup. End-to-end chain accuracy must be measured on a defined program distribution before claiming 90/80 performance for composed programs.
