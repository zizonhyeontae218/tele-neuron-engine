# DynaMath 003 — Codex Cloud full implementation and initial experiment handoff

**Repository:** `zizonhyeontae218/tele-neuron-engine`
**Coding branch/workspace:** `feature/dynamath-003` (already created from **DynaMath**)
**Canonical completed history:** `DynaMath` branch preserves all DynaMath 001, 001a, 002, 002a, 002b evidence and checkpoints.
**Status:** assigned-to-Codex handoff, NOT an implementation or run completion assertion.

This task is **a new mathematical architecture experiment**, not another continuation of fixed formula weights. Build functioning code, test, conduct a bounded and reproducible experiment, critically evaluate outcomes, and commit everything to feature/dynamath-003. Do not create further feature branches, force-push, merge into DynaMath/master, or rewrite older artifacts.

## 1. Research aim: truly learn function structures

Existing v002 defines 9 learnable low-rank nonlinear transformations; optimization updates their numerical coefficients but the operators and composition graph remain fixed. 003 must **actually propose, evaluate and select structural changes to expressions**, e.g. replace `tanh` with another bounded operator, insert/remove an elementwise term, change a typed operator/connection, or restructure a law expression graph. The evolved formulas must be inspectable and human-readable, and logged as a machine-readable AST/DSL with reproducible edit lineage.

**Not sufficient:** calling a larger MLP an evolved formula, learning a mixture of permanently fixed operators via ordinary gradients with no discrete structural selection, merely adjusting coefficients/scalars, or claiming that an arbitrary pretrained language model has discovered a new law.

Use a **two-timescale optimizer**: coefficient optimization / session adaptation inside; explicit structural mutation, search and selection outside. Autonomously decide whether to use strongly typed genetic programming, constrained AST search, or discrete operator search combined with backprop; explain trade-offs, deterministic grammar, structural identity and mutation count.

## 2. DynaMath identity constraints

Preserve the defining behavior:
1. One continuous persistent shared numerical state `S` (not an MLP pipeline or independent per-token slots).
2. Exactly one of 9 independently parameterized laws F1..F9 operates per absolute tick, repeating every 9 ticks, no resets at law/block or response boundaries.
3. Three groups of 3, **no different semantic role assigned to each group**. Add at most one queued input token to fixed input addresses at the **first** law of each group (`t%3==0`), and perform both real output checks after group laws two and three (`t%3 in (1,2)`). If output checks produce nothing that's legitimate; if both emit, preserve and measure FIFO behavior.
4. Maintain disjoint fixed input/output addresses, additive injection, a non-mutating output inspection event clock, explicit SILENCE/EOS, real closed-loop FIFO feedback, persistent runtime scalar(s).
5. Initially prefer **Model1: 9 laws with one globally shared writable scalar**, and treat Model3-cross (9 laws, 3 mutually readable but block-owned scalars) as optional later candidate. A scalar is runtime state, **NOT** just a trainable weight; initial 0, 1, or Euler's e as appropriate. Do not add an external memory bank.
6. Do not change the core event protocol to artificially get a good score. Protocol improvements may be investigated only as a separately named control and cannot be conflated with structural-law search.

## 3. First prototype scope, reproducible budget and baselines

Start with the archived evidence:
- `docs/dynamath/experiment-003/PLAN.md`: initial research ideas, subsidiary to this handoff.
- `src/dynamath/v002/__init__.py` and `__main__.py`: frozen v002 math and runtime semantics.
- `models/dynamath/002a/model1-step18000.pt`: initially better-behaved free-generation baseline, original test PPL ~5.285.
- `models/dynamath/002b/model1-step72000.pt`: 72k lower teacher-forced PPL ~4.519 but **severe free-running text degradation**. Treat both as comparison, NOT a clean victory.
- `models/dynamath/002a/model3-cross-step18000.pt` and `models/dynamath/002b/model3-cross-step72000.pt` for optional future controls.
- Pinned Song Ys ChatbotData source/splits/tokenizer in `data/dynamath/001a/`; do not contaminate test by searching sample answers or picking on test prompts.

Proposed **bounded first run** (Codex may adjust before running if justified and freeze in `configs/dynamath/experiment-003.json`):
- Typed candidate expression grammar with safe ops: identity, signed affine/low-rank, add, elementwise multiply, tanh, bounded sigmoid, residual blend; optional sin/clip. Enforce shape/type safety, numerical stability, and bounded expression size and latent cost.
- Mutate **one law at a time initially**, not all nine at once. Start from a comparable baseline expression, try up to 12-24 structural candidates, each with ~200-400 inner adaptation steps. Save parent, edited AST, coefficient initialization and validation score. Keep untouched-parent control under **matched optimizer updates, tokens, CPU/state transitions**, including a parameter-count and FLOP accounting.
- Prefer a function-preserving identity initialization for inserted nodes so the recurrent state distribution is not abruptly destroyed; otherwise document reset and compare fairly. All nine laws remain separate with one law/tick.
- After selecting by **validation only**, take the best real structural change through one bounded confirmatory continuation (e.g. <=2400 more updates) against the unmutated control and report total compute. Reserve test split for final once and freeze generation prompts.
- Compare teacher-forced response CE/PPL and actual unconstrained generated Korean/Unicode legality, question false-emission, EOS, repeated fragments, FIFO backlog, learning progress and runtime speed. Count actual changed formula AST operator/node/edge types and complexity; log concrete before/after mathematical expressions.
- Include simple deterministic synthetic tasks where known recurrence/dependency structure is helpful, so even if 003 language quality fails, mathematical-law discovery can be separately assessed. Avoid zero-shot memory claims. Explicitly demonstrate a selected structural edit has a causal effect, preferably via restore-parent/no-edit ablation.
- Terminate or scale down to keep within available Cloud CPU/time quotas. If constrained, deliver a running, self-contained structural evolution demonstrator plus truthful partial metrics rather than asserting full experiment completion.

Important: The old teacher-forced objective assigns output at the first readout and SILENCE at second and has failed to predict generative quality. For fair **structural search** control keep the original schedule and report its failure rather than invisibly replacing it. An altered loss/schedule is separate later investigation, not confounded with the central 003 hypothesis.

## 4. Expected files and interfaces

Create under this 003 branch:
- `src/dynamath/v003/`: typed AST/graph, interpretable law application, safe mutation engine, deterministic search, checkpoint and CLI, preserving the v002 single-state cyclic contract.
- `configs/dynamath/experiment-003.json`: frozen seed(s), operator grammar, candidate budget, evaluation split IDs, CPU cap/limits and fair control protocol.
- `tests/test_dynamath_003.py`: few focused unit/integration tests; verify readout clock, no leakage, scalar persistence and causality, AST serialization/parsing, operator structural change, finite ranges, reproducibility and data isolation.
- `scripts/run_dynamath_003.py` and `docs/dynamath/experiment-003/`: training/search runner, deterministic saved-checkpoint replay, exact logs, resulting formulas, control comparisons and limitations.
- `models/dynamath/003/`: checkpoint(s) including structural AST, learned coefficients, optimizer and runtime state. Checkpoint inference must not require dataset/answer lookup.
- A top-level 003 README and accurate setup/reproduce commands. Audit existing 002 assets remain unmodified.

The existing `src/dynamath/v003/README.md` and `models/dynamath/003/README.md` are workspace markers, not code implementation; replace/extend only within the new branch as needed.

## 5. Acceptance checklist

- Demonstrate >=1 truly structural law mutation: a recorded operator/node/connection change beyond parameter values, with before/after readable expression and parsed serialized representations.
- 9-law shared-state 3x3 block event invariant remains true in baseline and mutated candidate.
- A saved AST + weights/state checkpoint can reproduce output deterministically with no corpus loaded; tests pass; old DynaMath 002 tests remain valid.
- Distinguish validation-based candidate selection from untouched held-out final test. No fabricated wins or semantic scores. If PPL improves but real generation worsens (as happened in 002b), explicitly declare that conflict.
- Record raw candidate outcomes, computational budget, failures, rejected mutations and limitations. Stop at a coherent prototype if compute/architecture constraints prevent more.

## 6. Deliver to user

Push new code and evidence on `feature/dynamath-003` only; include actual commit SHA, key file links, mathematical formulas (before/after), trained/generation outcomes, runtime/tests, critical assessment of whether 003 truly learned formula structures and what it did not learn. Do not prematurely rename 003 to 004. No automatic future tasks.
