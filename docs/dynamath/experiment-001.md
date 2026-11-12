# Experiment 001 — first DynaMath 1

Run date: 2026-11-12 (KST). Starting commit: `2d34a9af3bdc47d0c6ef0b2291f7a024e64f2f6a`.
Independent branch: `feature/dynamath-1`; existing master/endophalon3 remain untouched.

## Fixed acceptance criteria and outcome

Before learning, docs/dynamath/GOAL.md specified: invariant tests pass; XOR and delayed
recall >=90% on fresh episodes; held-out character cross entropy improves over
initialization; actual address-event generation emits text. All four were met.
This is an implementation/learning milestone, not a useful-chatbot or GPT-2 benchmark.

Exactly one five-law logic training run (600 updates), one single-law reference
(600) and one character run (800) were executed. No hyperparameter search, best-of
sampling, holdout-driven selection or attractive-output retraining occurred.
Small unit-test training runs use temporary fixtures to verify exact resume; legacy
smokes use temporary tiny fixtures, never the preserved production model files.

## Environment and data

- Python 3.12.14, torch 2.15.0+cpu, NumPy2.3.5, pytest9.2.0.
- Linux-6.18.57-x86_64-with-glibc2.41; INTEL(R) XEON(R) PLATINUM 8573C as reported by the virtual environment; CPU only.
- torch/BLAS/OMP threads=1. Runtime benchmark is batch1. Training batches differ.
- Logic optimizer seed41, training-generator seed42, held-out seed202 (512 episodes:
  240 XOR, 272 recall). Separate longer-gap probe seed303 (246 XOR,266 recall).
- Bits/operation are independent random binary values. Training and standard holdout
  gaps are 0..2 full idle cycles after each operand, amplitudes [0.9,1.1]. The query
  arrives after both operands. Recall asks for the first operand despite the second
  operand and idle ticks (15..35 ticks from first injection through query completion).
  The seed303 probe permits 0..3 gaps (15..45 ticks). Phase starts cycle through all5.
- Text seed43, 2784 training characters (50 authored dialogue pairs),
  608 validation characters (10 distinct dialogue pairs),
  27-symbol training-only alphabet. Files are in data/dynamath. No downloads.
  Pair split was fixed in advance. Common words and some answer phrases occur in
  both splits; this is a small same-domain holdout, not novel factual generalization.
- Full hyperparameters/defaults: configs/dynamath/experiment-001.json and architecture.md.
  Corpus/checkpoint hashes: models/dynamath/model-card.json and the .pt payloads.

## Learning results

| Model / metric | Before training | After training |
|---|---:|---:|
| Five laws, XOR (240 held-out cases) | 2.08% | 100.00% |
| Five laws, delayed recall (272 cases) | 6.25% | 100.00% |
| Single law, XOR | 2.92% | 100.00% |
| Single law, delayed recall | 0.00% | 100.00% |
| Character held-out cross entropy (nats) | 4.599113 | 1.224162 |
| Character held-out perplexity | 99.3962 | 3.4013 |
| Character held-out top1 accuracy | 0.66% | 68.53% |

Logic predicts over all8 vocabulary output addresses, not just a forced binary
head; this explains unusually low untrained accuracy (binary chance50% is not
the relevant initialization baseline). The finite truth-table combinations occur
in both training and evaluation. Holdout means new randomized episodes/amplitudes,
not unseen XOR truth values. Both models reach100% on the longer-gap probe as well.
The saved five-law checkpoint also emitted the correct real thresholded event for
all8 bit/query combinations at35 ticks, with output leaving state unchanged;
see logic-events.json.
Each of the five start-phase evaluations also gives100% XOR/recall. These tasks do
not establish a benefit from having five laws: the smaller single-law model ties.

Final train-text CE=0.533496, accuracy=84.33%;
validation CE=1.224162, accuracy=68.53%.
A uniform27-character baseline would have CE=ln27=3.295837. Text metrics are ordinary
teacher-forced next-character evaluation, not autonomous conversation accuracy.
Validation threshold eligibility coverage is 94.89%;
it is measured at character boundaries, with no forced fallback emission.
Training loss includes the threshold hinge; the CE table does not.

Training-loop wall times: five-law logic 5.743s, single-law logic
6.482s, characters 38.273s (exclude evaluation/save/install).
600 vs800 steps are different task budgets. The character run processes
800*24*16=307,200 teacher-forced next-character positions, 1,536,000 individual
state ticks represented by96,000 batched tensor updates. These training counts
are not a batch1 generation throughput claim. Compact history and per-phase
metrics are preserved in results.json/logic-results.json; original stdout is training-run.log.

## Actual greedy generations

Each row starts a new zero-state Session, conditions on the shown prompt, then
uses actual thresholded output addresses with additive predicted-token feedback.
The final runtime reproduced the original samples. `\n` is a newline.

| Prompt | Generated continuation, verbatim |
|---|---|
| `user: hi\nbot:` | ` hellogogo does the leargings.\n` |
| `user: what are you?\nbot:` | ` i lomate stter can and teas and rules.\n` |
| `user: hello\nbot:` | ` hello there.\n` |
| `user: what is dynamath?\nbot:` (held-out prompt) | ` a litt e addrres ise lit ticg anti ko a tick.\n` |

The misspellings and repetitive clauses are real. A correct familiar greeting is
compatible with memorization of this small corpus; it does not establish reasoning.
No response strings appear in inference code. Persisted multi-turn CLI and idle
snapshots were exercised separately in cli-verification.log.

## Final-runtime performance and stability

- N160, rank24, five laws; **40,920** learned coefficients total, **8,184** active/tick.
- Approximate arithmetic: **17,328 FLOPs/tick** plus184 tanh evaluations, parameter
  sigmoid/tanh transforms and runtime overhead. O(Nr); see architecture.md for counting.
- Three5000-tick batch1 runs including output checks: 18928, 18467, 17458
  ticks/s (median 18467); elapsed
  0.264165, 0.270755, 0.286395s. No token feedback in this microbenchmark.
- Separate full generation measurement: ten fresh sessions,80-character budget each,
  greedy feedback, no newline stop,1000-tick limit each. 800 characters in
  0.446985s = **1789.8 char/s**;
  7250 actual ticks including prompt conditioning and idle waits.
  Weights already loaded; model load, printing, I/O and network are excluded.
- FP32 coefficient tensors: 163,680 bytes (~159.84KiB);
  one persistent state: 640 bytes. Checkpoint file is515,765 bytes
  because it also includes Adam and training stream state.
- Final benchmark process peak RSS: 313,088KiB
  (~305.75MiB); this includes Python/PyTorch and other allocations,
  not only parameters. The original training-process peak was349,020KiB.
- 10,000 idle ticks: finite state; max absolute value at1000-tick samples ranged
  0.917012..0.950485.
  Different conditioned prompts ended at L2 distance 3.625790.
  Nonzero distance does not prove preserved semantic memory; no idle recall decoder was tested.
- Initial-runtime benchmark is retained separately in results.json. Factoring the
  output detector added function-call overhead; report the final-runtime figures above.
  No training rerun was needed because mathematical behavior did not change.
- No same-condition spring/Endophalon timing was performed. There is **no claimed
  speedup** over144 nodes/506 springs/2 substeps or a16-tick Endophalon event.
  A DynaMath tick, an input cycle and a complete generated character are different units.

## Verification and preservation

- `python -m pytest -q tests/test_dynamath.py`:7 passed. Covers additive/signed/duplicate
  input, ten-tick phase progression, idle updates, no output-state mutation,
  deterministic ties/refractory, real additive feedback, silence/budget handling,
  snapshot restoration with pending inputs at phase2, finite idle decay and
  exact interrupted-vs-uninterrupted TBPTT weights/state/offsets.
- Full `python -m pytest -q`:98 passed (16.55s). One warning comes from the
  environment's third-party anyio pytest plugin and is unrelated to model behavior.
- `python scripts/smoke_models.py`:8 existing model/config combinations passed,
  with temporary reduced fixtures; no saved production weights changed.
- `python scripts/verify_backup_consolidation.py`:166 byte-identical reproduction
  files,22 models,9 original heads reachable. Offline check does not re-fetch Drive.
- All136 original tracked source/model/config files compare byte-for-byte with the
  base commit. All original remote branches/tags match the pre-work snapshot.
  Only the new feature ref is added. Root changes are limited to agent exception,
  README pointer and generated indexes. Torch is declared in a separate DynaMath
  requirements file; the original pyproject.toml is preserved byte-for-byte.
- Preservation evidence: preservation.json; generated index lint and final committed
  protected-file checks are recorded in verification.log.

## Limits and verdict

Implementation invariants and the predeclared first-experiment gates pass. The
architecture learns nonlinear computation, delayed recall and next-character
patterns; genuine persistent-state generation works. The tiny English corpus,
27-character alphabet, overfitting gap and visibly broken free generations prevent
calling this a useful chatbot. Korean, broad knowledge, instruction following,
long-context memory, arbitrary-phase text quality and noise robustness are untested.

The sigmoid/tanh bound prevents numerical idle explosion, but does not guarantee
meaningful unattended activity, absence of attractors, unlimited memory or causal
interpretability. The fixed event detector can emit nonsense during idle periods;
training does not include a learned silence policy. Repeated/large additive input
can saturate state. One seed per model and a small synthetic task set are limited
evidence. There is no GPT-2 equivalence claim and no evidence yet that five phases
outperform one on these tasks.
