# DynaMath 002a — intensive continuation on the EXISTING 002 branch

**Branch remains `feature/dynamath-002`.** 002a is a VERSION/EXPERIMENT LABEL and a set of new folders, not a new branch. This supersedes any earlier branch creation plan. Do not merge to master or overwrite any 002/001a/Endophalon evidence.

## Selection
- Continue `model1` (one dynamically writable scalar globally shared across all 9 laws).
- Continue `model3-cross` (three block-owned writable scalars, all blocks allowed to read across).
- Continue `model9-frozen` as the non-writing 9-law control; train its *fixed coefficients* under the same schedule.
- Exclude `model3-local` and `model9` from 002a training. Do not delete their historical checkpoints.
- All retained variants have **9 independently parameterized laws**, one common information state, 3-tick block-start inputs, two readout checks per block. Do not confuse model1 with a one-law model.

## Heavy continuation with no architecture changes
- Start from existing `models/dynamath/002/<variant>.pt` checkpoint, including Adam moments, learned coefficients, state S, scalars m, absolute tick, train-stream position and RNG. Resume training; do not restart training from random initialization.
- Freeze original 002 mathematical law, scalar gates, tokenizer, fixed Song Ys dataset and deterministic training stream. Do not silently rewrite formula structure, loss or input/output protocol as part of 002a. The project currently learns *parameters within a chosen family of formulas*, not symbolic mathematical formulas; report the distinction accurately.
- Total target: 18,000 parameter updates per variant (initial 1,200 plus 16,800 added) = 15x total 002 steps, ~9.1 passes of the existing truncated per-lane stream. Maintain the original 002 split. Segment ends: 3,600, 7,200, 12,000, 18,000 total updates.
- Preserve the original 002 stream semantics for continuation compatibility, but **report explicitly** that the original `streams()` truncates unequal lanes to the shortest and therefore excludes some training examples. Do not claim all 9,460 pairs were visited unless measured and verified.
- Per-segment wall cap 2,100 seconds; stop and report incompleteness if cap prevents reaching an end-step. CPU, threads=2, batch=12, TBPTT 24 input slots=72 state ticks, Adam settings from checkpoint.
- At each segment end: save new `models/dynamath/002a/<variant>-stepXXXXX.pt` (not old file) and validation metrics. Test data MUST NOT be used for selection/tuning: evaluate entire pinned held-out test once per finished final model, report actual evaluated positions if current evaluate() truncates tails. Generate frozen prompt samples with actual closed-loop feedback, preserve raw malformed samples (no postprocessing).
- Validate resume and model/optimizer/state continuity at least on small deterministic fixture; calculate checkpoint hashes, report train and eval environment/runtime, and provide final comparative table for 1/3-cross/9-frozen.
- Keep 002 initial checkpoint and raw results immutable, and compare against 002 pre-extension metrics with a caveat about prior 96-pair test. Use unchanged 002 implementation, any forward behavior difference from checkpoint must be flagged.
- Existing 002 recall models are a separate experiment. No assumption that additional language training makes m carry useful episodic memory. Optional fixed probes on final language models may isolate m reset vs S reset; no new memory-claim unless trained and supported.

## Execution
The 002a GitHub Actions workflow (`.github/workflows/dynamath-002a-train.yml`) runs `scripts/train_dynamath_002a.py` on `feature/dynamath-002`; it is triggered only by the workflow/script/GOAL changes, not by output commits. On success it commits checkpoints, logs and report to the SAME branch. On failure it uploads partial artifacts. Do not create new branches.

This is an experiment goal, not a claim of successful additional training until the Action has actually completed and committed evidence.
