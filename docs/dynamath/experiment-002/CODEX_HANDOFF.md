# DynaMath 002 — Codex implementation handoff

**Canonical target:** `feature/dynamath-002` in `zizonhyeontae218/tele-neuron-engine`, branched from `feature/dynamath-001a`. Deliver the runnable implementation, reproducible evidence, and critical report on this branch. Do not merge. Preserve DynaMath 001/001a, Endophalon, other branches, and their checkpoints. This document is the new project's design contract, not an assertion that implementation is complete.

## 0. Objective and non-negotiable architecture

DynaMath 002 is a **single-state, nine-law, three-block cyclic dynamical model**, NOT a depth-9 MLP, a nine-stage pipeline, or separate per-block state machines.

- There are **exactly 9 independent learnable laws**, named `F1..F9`. Divide them into three contiguous blocks `(F1,F2,F3)`, `(F4,F5,F6)`, `(F7,F8,F9)`. **Do not give blocks different semantic roles** (e.g., encode/retrieve/decode). All laws share the same parameterization class and one persistent information state.
- **One absolute tick executes exactly one law** on the **same shared state** `S`. Law is `F[(absolute_tick % 9)]` (zero-based internally). After F9, continue from F1. No reset at law, block, conversation or response boundaries.
- **Input admission**: at the FIRST law of each block only: F1, F4, F7, corresponding to `t % 3 == 0`. Inject **at most one queued input token per such slot**, additively into fixed input-address coordinates of `S`, before applying the selected law. At other ticks no token injection. At an empty admission slot apply the law with no input.
- **Output readout checks**: after applying the SECOND and THIRD law of every block only: F2/F3, F5/F6, F8/F9, i.e. `t % 3 in (1,2)`. Neither output attempt must emit; both must independently perform a real check. No outward emission on the first law. `SILENCE` competes in output-address space; don't externally force output/silence by the question/answer mode.
- Preserve fixed disjoint input/output addresses, additive injection, shared state across turns, deterministic event clock/polling, EOS, explicit SILENCE, and low-rank nonlinear-law spirit of DynaMath 001a.
- Retain one mutable scalar unknown `m_i` **per law**. Each initial value is deterministic and **must be one of exactly 0, 1, or e** (Euler's number, not a learnable initial parameter), e.g. `[0,1,e,0,1,e,0,1,e]`. The law **may replace its own value during execution**. Each `m_i` is a runtime state variable, NOT an optimizer-owned/static model weight. Other laws do not directly overwrite `m_i`. The current `m_i` must be read by its law so a write can influence future computation. Preserve all nine values through subsequent cycles, responses, idle ticks and checkpoints. Do not introduce more than one mutable scalar per law or a separate external memory bank in v002.

Rationale: 001a's five-law/one-tick test CE 1.9702 was worse than one-law/one-tick 1.7912, while five-law/five-tick CE 1.6283 consumed ~5x state transitions per token. These are teacher-forced next-token metrics, not conversational success. 002 tests whether a repeating **3-tick input cadence + 2 internal readout opportunities** and minimally writable law parameters improve behavior without converting the system into a feed-forward pipeline.

## 1. Precise clock and invariants

At each tick `t`, select `i=t % 9`, let `b=floor(i/3)`, `p=i%3`:
1. If `p==0`, dequeue at most one input symbol from a deterministic queue and add it to the fixed input-address region. Otherwise inject nothing.
2. Apply precisely `F_i` once, using the current `m_i`; optionally write a new `m_i` via that law's own learned state-dependent write rule. Write ordering (before/after state update) must be explicit and consistent for training/inference.
3. Increment the absolute tick.
4. If `p in (1,2)`, inspect eligible output addresses without mutating `S`, using a deterministic threshold, `SILENCE`, tie rule, and an event clock preventing a repeated `poll()` from duplicating an event.
5. For actual emitted symbols, record every event. Both eligible ticks are allowed to emit; no hidden forced 'one event per block' unless explicitly justified as an optional experiment, never as the default. Feedback of emitted tokens joins a deterministic FIFO queue, and can be admitted only at a subsequent block-start input slot; document any throughput/backpressure effects.
6. Do not overwrite the information state with the emitted token. All feedback follows the exact same additive input path as external tokens.

External question input versus self-feedback must have a documented, deterministic queue policy. No future token, target answer, or privileged response position may enter the state before its lawful input slot. Keep diagnostics: tick index, law index, block index, admitted token, selected output event/silence, output score, full FIFO depth, and per-law `m_i`.

**Temporal causal detail:** A block begins at t=0/3/6 etc.; readout occurs after t=1/2, 4/5, 7/8. The phase wraps after t=8 without resetting the state. An ordinary input token gets 1, 2, or more transitions depending on when it is read; do not pretend every input completes nine stages. No distinct per-token pipeline states.

## 2. Candidate mutable-unknown mathematics (Codex may refine with justification)

For each law, reuse the 001a low-rank transition structure, with a learned fixed influence of its *single* dynamic scalar:
```
z = S + additive_input_if_block_start
h = tanh(V_i z + b_i + q_i * m_i)
c = tanh(d_i ⊙ z + U_i h + c_i + r_i * m_i)
S_next = (1 - alpha_i) ⊙ z + alpha_i ⊙ c
```
All `U,V,d,b,c,q,r,alpha` are ordinary learned **fixed coefficients** after training. Each `m_i` is a persistent writable **single scalar** whose value affects later transitions. Design a small law-local bounded write candidate `v_i(S_next)` and gate `g_i(S_next)`:
```
m_i_next = m_i     (no-write)
        or clip(v_i(S_next), range)  (write)
```
A differentiable surrogate / straight-through gate is permitted for training, but inference must provide an actual retain-or-replace policy if writing is described as assignment. Prove/pin numeric bounds, preserve the exact initial constants, and log writes, gate activations, values and update counts. The laws must not be able to directly access future answers. The other eight `m_j` are unchanged at this tick. Avoid separate MLP stacks or a learned extra memory network disguised as a scalar.

**Distinguish** trainable coefficients from running variables. The `m_i` are session/checkpoint state, not `nn.Parameter`; gradients may flow through a differentiable runtime representation during training. Be explicit whether inference uses hard writes or continuous gated interpolation; no false claim of exact retention with continuous blending.

## 3. Training & experiment protocol

Re-use and preserve 001a's pinned Song Ys source, train/validation/test split manifest, Unicode tokenizer and source credit/license. Keep the 001a checkpoints untouched. Make a new `configs/dynamath/experiment-002.json`, dedicated package/runtime, tests, `models/dynamath/002/` outputs and `docs/dynamath/experiment-002/` evidence. Coexist with legacy v001/v001a; do not accidentally alter their scheduler defaults or their regression tests.

Avoid conflating token/input slots, readout events, and ticks:
- Each supervised external/teacher-forced input token is admitted only at block starts, so adjacent inputs are separated by three state transitions.
- Two output checks happen per block. Explicitly define a causal target schedule for each eligible check (including SILENCE, BOT, answer tokens and EOS), and prove the same schedule can be followed in teacher-forced training and genuine closed-loop generation without injecting ground truth at inference.
- Account for *both* readout slots in learning. A high teacher-forced PPL score is not sufficient; closed-loop output must be measured with its natural feedback delays, missed emissions, double emissions, and queue saturation.
- Separate question-time false emission, response omission, EOS, format/Unicode quality and semantic relevance in reporting. No hard-coded suppression of bad outputs in the final numbers.
- TBPTT length should be expressed in ticks AND input slots. A mean 001a pair spans ~58 NFD tokens; the 002 cadence implies ~174 ticks per pair. Investigate credit assignment across question/answer boundaries without claiming recurrent numeric persistence equals trained recall.
- Freeze planned metrics/IDs/seeds and bounded compute limits **before** full Song Ys training. Use at most one small implementation smoke/pilot, then a bounded, reproducible primary run; no tuning or cherry-picking on test.
- Include a **matched 9-law, 3-tick control with `m_i` frozen** to isolate the contribution of writable variables (similar total learned coefficients as practical), and report the legacy 001a baseline separately. If resources allow, an additional one-law/three-tick control can help isolate law multiplicity, but do not overclaim from single seeds, different compute or non-identical schedules.
- Train a small explicit delayed-recall task (e.g. bit/name with 0,5,50 idle ticks) with frozen held-out cases. Compare persisted `S` and `m_i` against controlled resets: reset only `S`, reset only `m`, reset both, preserve absolute tick in all arms. The previous 001a memory probe was **zero-shot**, so 0% was not evidence of unlearnability. For each learned task, verify whether `m_i` actually changes, stores target-conditioned information, and influences future output.
- Distinguish genuine model correctness from decoding postprocessing and answer lookup. No pretrained embedding, trainable token embedding or output head unless explicitly justified as a separate controlled research variant.

## 4. Minimum validation tests

Write focused tests rather than an unbounded test matrix:
- 9-law ordering wraps correctly; exactly one law/tick; all blocks use same `S` object/value flow; no law-private states.
- Inputs only at phases 0,3,6; queued tokens admitted FIFO at most one per slot; additive duplicate-address accumulation.
- Output checks at phases 1,2,4,5,7,8 only; real checks whether or not an event is emitted; repeated poll nonmutating and nonduplicating; ties, silence, threshold, EOS, double-event case.
- Numeric `m_i` initial values are restricted to {0,1,e}, exactly one variable per law; only active law writes; exact no-write persistence across cycles, idle and conversations; effect of a write on the next invocation of that law.
- Exact snapshot/restore of `S`, all 9 `m_i`, tick/phase, queue, and event clock. Resume training restores optimizer/RNG/data identity and reproduces uninterrupted training on a tiny fixture.
- Verify old DynaMath 001a/legacy smoke still passes without changes to checkpoints.
- Replay saved 002 model output with the dataset unavailable. Fail loudly rather than looking up training answers.

## 5. Deliverables and reporting

1. Runnable DynaMath 002 train/evaluate/generate/chat/tick CLI and a transparent numeric state snapshot export (JSON-safe representation for iPad/HTML inspection may be supplied as a separate **derived** export; never imply arbitrary torch weights are inherently JSON executable).
2. New configs, tests, source code and checkpoint(s); maintain versioned Song Ys attribution.
3. Raw logs and frozen sample outputs, both teacher-forced and free-running metrics, write statistics for all 9 unknowns, runtime memory/throughput and clearly stated hardware/method.
4. Report which improvements were observed and **which failed**. If language quality remains incoherent, say so plainly. Report parameter counts and CPU time; avoid unfair claims of efficiency against 001a.
5. Preserve old branches/artifacts and provide actual feature branch URL, commit SHA, important file links, test status, metrics and limitations.

## 6. Codex execution directive

Implement on the existing `feature/dynamath-002` branch. **Do not create an additional new main development branch**, do not merge to master, do not delete or overwrite old experiments. Treat this design as binding; for unspecified math/training details, use your judgment, explain choices before or in the report and maintain reproducibility. Finish with actual code, experiments and committed evidence, not a proposal-only response.
