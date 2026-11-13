# DynaMath 002b — final fixed-formula continuation

**VERSION / EXPERIMENT LABEL:** DynaMath **002b**. **Branch remains `feature/dynamath-002`**; do not create a new branch. Preserve 002, 002a, 001a and older experiments, and do not merge into master.

## Scope
- Train ONLY `model1` and `model3-cross`. **No** additional `model9`, `model3-local`, or `model9-frozen` training in 002b. Their original 002/002a artifacts remain untouched.
- Continue **from existing 002a 18,000-update checkpoints**, including optimizer moments, learned coefficients, recurrent shared state, writable scalar state, tick phase, RNG, and exact stream position. The 002a progress JSON SHA-256 validates each starting checkpoint.
- Preserve the existing **9 independent laws**, single shared state, 3-law blocks, admission at block-start and two output checks at the second/third law, and model-specific shared scalars. Absolutely **no change to the mathematical formula, tokenizer, loss, data split, training cadence, or optimizer hyperparameters**. This trains fixed-formula coefficients; it does **not** evolve or discover symbolic equations.
- Cumulative targets: **30,000 / 45,000 / 60,000 / 72,000 optimizer updates**. Total extension is 54,000 updates per model after 002a. At 72k, the original 24-input-slot/update stream implies ~36.38 nominal stream passes; the original unequal-lane truncation and any excluded pairs must remain documented.
- Per-stage wall clock cap 2,100 seconds. If an interrupted stage cannot reach its target, report the actual steps and incomplete status. Stop on nonfinite loss or severe instability rather than claiming success.
- At every target: immutable checkpoint in `models/dynamath/002b/` + full pinned validation evaluation and exact source lineage. Use validation only for diagnosing convergence; evaluate pinned full test **once** for each final completed model. Never tune based on test.
- Raw closed-loop generation with actual FIFO feedback, no forced EOS/answer lookup/cleanup, same frozen prompts as 002a. Report EOS, repetitive outputs, output quality, response/false emission and missed output. End-of-project selection must account for genuine coherent Korean, not simply lower CE.
- Do not conflate 002a's 18k test PPL (Model1 5.2849, Model3-cross 5.2447) with guaranteed gains; report both and mark any deterioration. 002a and 002b full test evaluation have matching denominators and objectives.
- Preserve checkpoints from all stages. Summarize validation trajectories, compute/size, dataset coverage, remaining limitations, which model is better, and whether further conventional fixed-formula training appears worthwhile.
- **End 002 after this final experiment**; no automatic follow-on training. Do not start 003 here.

## Execution
Workflow: `.github/workflows/dynamath-002b-train.yml`.
Trainer: `scripts/train_dynamath_002b.py`.
Results: `docs/dynamath/experiment-002b/` and `models/dynamath/002b/`.
GitHub Action commits results back to the same existing 002 branch. A successful workflow definition commit is NOT equivalent to completed training.

## 003 is deferred
The design concept of learning/changing the *structure of functions themselves*, instead of only coefficients within fixed functions, is recorded separately in `docs/dynamath/experiment-003/PLAN.md`. That is a **future research plan only**, not authorization to implement or run it during 002b.
