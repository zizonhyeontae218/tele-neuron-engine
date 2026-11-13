# DynaMath 003 — defer: evolve mathematical laws rather than only their coefficients

**STATUS: RESEARCH PLAN ONLY.** Do not implement, launch, or train 003 during the 002b final continuation. This document merely preserves the proposed direction. There is NO new branch for this plan.

## Motivation

DynaMath 001/002/002a/002b call their recurrent transformations "laws" or "formulas", but the current training optimizes numeric parameters in a fixed template: nonlinear functions and the order of operations are predetermined. The true DynaMath long-term idea is that the program can also modify/learn **which mathematical operators, compositions and expressions constitute its 9 recurrent laws**.

## Preserve from 002 unless a later explicit decision changes it

- One continuously updated **shared persistent numerical state** S.
- Exactly **one law executes per absolute tick**, cycling F1..F9 with no reset.
- Three blocks of three laws, **no semantic role separation among blocks**.
- At the first law in each block, admit no more than one queued token additively into fixed input addresses; after the second and third law, check output addresses, permitting actual silence or emission.
- Keep learned numeric coefficients distinct from volatile runtime memory scalars. Preserve initial m values from {0,1,e}; consider both model1 and model3-cross as possible starting points based on the completed 002b evidence.
- Do NOT disguise a nine-layer feed-forward MLP, transformer, or independent-per-token pipeline as DynaMath.

## Candidate formula-evolution approach (not yet fixed)

Define a small safe, typed expression grammar for law internals:
- Scalar/vector unary: identity, tanh, bounded sigmoid, clip, abs; optional sin as separately tested operator.
- Binary: add, subtract, elementwise multiply, bounded affine/low-rank transform.
- Only finite, type-compatible, dimensionally valid operations; strict expression-size/depth limits, bounded norms and update magnitudes.
- Each law's expression is serialized as a human-readable AST/graph, along with constants/coefficients, random seed, provenance, and hash. A candidate must be inspectable as an *actual mathematical expression*, not only hidden weights.

Use a **two-timescale search**:
1. Inner loop: optimize numeric coefficients and existing scalar-write gates for a fixed candidate law expression, from matched initial conditions.
2. Outer loop: mutate/replace operators, add/remove/reconnect valid expression nodes, and score candidate formula structures after a limited adaptation budget.
3. Apply selection or acceptance with complexity penalties, stability/recurrent dynamics guards, token-level and genuine closed-loop quality tests.
4. Preserve elite baseline formula(s) and a reversible rollback point; ensure mutations of one law don't silently invalidate state continuity without fair comparison.

## Scientific failure modes and controls

- **Compute blowup** from symbolic/program search; require strictly bounded candidate counts, inner-loop steps and total CPU budget before running.
- **Catastrophic downstream distribution shift** when one recurrent law changes; measure recurrent divergence and preserve a parent control.
- **Gradient issues** for discrete operator edits; never imply backprop differentiates directly through arbitrary symbolic edits. Use explicit search/surrogate approaches with unbiased comparisons.
- **Degenerate complexity**: trivial identity/silence formulas, excessively deep expressions, numerical NaNs, learning answer lookup, overfit to short generated samples.
- **Token loss vs meaning**: lower teacher-forced PPL does not demonstrate coherent answers. Freeze held-out prompts and include no-lookup closed-loop generations and delayed-recall tests.
- **Meaningful runtime scalar**: demonstrate causal effect by selectively resetting m versus S, not just counting scalar writes.
- **Ablation**: best fixed-formula 002b final checkpoint vs formula-evolving candidate under matched train tokens/state transitions and hardware.

## 003 acceptance conditions for a later project

- The resulting code can print readable mathematical formulas before and after evolution, and show actual structural edits beyond numeric coefficient updates.
- Reproducible candidate lineage, measured costs, tests for state/tick invariants, checkpoints and honest comparison with frozen 002b.
- Explicitly state negative results if symbolic evolution fails to improve held-out performance.
- **Do not schedule, code, train, or create 003 branch until user separately requests 003.**
