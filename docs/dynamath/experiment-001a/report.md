# DynaMath 001a — experiment report

Data credit: **Song Ys (`songys`)**, [original Chatbot_data](https://github.com/songys/Chatbot_data), MIT. Dataset authorship is separate from DynaMath model development.

Run date: 2026-11-13 (KST). Existing branch `feature/dynamath-001a` began as the
identical copy of `feature/dynamath-1` at `15d897b111ced59fd881c01201c5a530fa36e9c1`.
The supplied [handoff](handoff.md) is archived; the user's existing-branch decision
supersedes its branch-creation step. [GOAL.md](GOAL.md) freezes scope and budget.

## Outcome

**A — Stage A event logic verified.** New default checks output immediately after
every tick with refractory 1; tick 1 and all five phases can emit. It preserves the
single information state, additive inputs, one cyclic law/tick and readout purity.
Explicit 5-tick compatibility remains for the matched control and original weights.
The event/scheduling change does not change law coefficients or geometry.

**B — Korean training completed.** Four predeclared runs each completed 2,400
updates and covered all 9,460 train pairs. Saved models generate genuine decoded
Korean locally without dataset access. All frozen sample replays reproduce exactly.

**C — Conversational quality and semantic memory failed.** The primary model's
six semantic cases all fail qualitative relevance/coherence review. Zero-shot
memory recall is 0% with and without state reset. Generated text is often malformed.
This is an executable experiment, not a useful chatbot, GPT-2 equivalent, proof
of consciousness, or evidence of five-law superiority. The rank-matched single
law collapses to silence; this negative result is retained without a retry.

## Source, split and tokenizer

Original GitHub source commit `4cf20d13fc46f5037fd1c531cd566e2dd9f72974`;
CSV Git blob `4cc2bfa9a124991dc5ca80f5da1786f4162f46e1`;
CSV SHA-256 `287eb129695b577321c80ad397bb3c2279164d4ca577874d129fd3db5b30afe2`.
Acquisition UTC: `2026-11-13T00:32:41.152734+00:00`. Original MIT text is retained at
`data/dynamath/001a/songys-LICENSE.txt`. No Hugging Face mirror was used.
The original pinned CSV parses to **11,823** pairs, not the approximate 11,876 in
the handoff. No invalid/empty rows; one whitespace-padded label is stripped.
There are 73 extra exact Q/A duplicate rows and 161 extra exact Q rows, retained
within their question groups. All 11,823 valid pairs enter the split before streams.

Groups use NFC/casefold/alphanumeric-only question keys (11,600 groups), then
SHA-256(seed 43 : key) modulo 100 assigns 80/10/10 buckets. This groups spacing
and punctuation variants; semantic paraphrases can still cross splits. It is not
semantic deduplication or exact label stratification. No hyperparameters or
samples were selected using test results. The manifest and evaluation plan were
written before training. Full test metrics were evaluated once per final model.

| Split | Pairs | Labels 0 / 1 / 2 | Text tokens | UNK tokens | Roundtrip failures |
|---|---:|---|---:|---:|---:|
| train | 9,460 | 4199 / 2893 / 2368 | 549,767 | 0 | 0 |
| validation | 1,150 | 538 / 337 / 275 | 65,690 | 1 | 1 |
| test | 1,213 | 553 / 340 / 320 | 69,074 | 1 | 1 |

No exact normalized Q or Q/A pair overlaps across splits. Repeated distinct exact
answers overlap: train/validation 544, train/test 597, validation/test 109. Thus
familiar answer wording is common; the holdout is not independent factual knowledge.
Raw NFC normalization changes: 0. NFC ingest -> NFD codepoints; fixed train-only
vocabulary **112** including USER/BOT/EOS/SILENCE/UNK at IDs 0..4. All training
texts roundtrip exactly. Validation/test have one unknown codepoint each (see
source-report.json); UNK decodes as replacement character, never expands vocabulary.
Unknown rates are 0.001522% and 0.001448% of text tokens respectively. No learned
tokenizer, pretrained embedding or learned output head exists.

## Architecture and protocol

Unchanged five-law low-rank family from [Experiment 001 architecture](../architecture.md),
scaled to N352/rank24. Input addresses [0,112), output [112,224), latent [224,352).
Softmax and the event argmax exclude latent addresses. Gain6 and threshold0.1 are
fixed. Every tick adds pending input, applies exactly one F[t mod laws], then
checks output. Each law has its own U,V,inner bias,bias,diagonal,leak coefficients;
only these coefficients learn. One persistent numeric state per runtime instance;
training batch16 represents 16 independent instances, never five private states.
No trainable initial state, physics or output-state reset.

USER/question ticks target the ordinary SILENCE address. BOT injection's immediate
post-law read predicts the first answer token; each additive answer token on the
next tick predicts its successor, ending with EOS. EOS feedback targets SILENCE.
Question-time emissions are collected, not externally mode-masked. If response
readout is silent, inference genuinely idles rather than substituting truth.
Teacher forcing uses true feedback; free generation uses predicted feedback, so
exposure mismatch remains and is measured through raw outputs. Law phase continues
across all tokens/turns and TBPTT boundaries. See README.md for cap/queue behavior.

Stage A isolated the scheduler using identical weights: original 5-tick output
` hellogogo does the leargings.\n` at260 ticks becomes
`uuuuwp  rrtsweeetiaaatttttttt eh.i.\n` at48 ticks for the fixed greeting prompt.
This expected distribution shift is not evidence of failure to learn Korean.
Stage-A tests cover actual crossing, silence, ties, every phase, first tick,
repeated symbols, duplicate polls, nonmutation, queued inputs and exact restore.

## Frozen compute budget and training evidence

One seed43, CPU, torch/OMP/BLAS one thread, Adam lr.002, gradient clip1,
batch16/TBPTT32. No learning-rate search, best checkpoint or quality-driven rerun.
At most2,400 updates or600 training seconds per run; all hit the step cap.
Each sees **1,228,800 supervised input/output positions**, about2.125 stream passes.
All578,147 distinct training positions and all9,460 complete pairs were covered.
The 5-tick control has five times the state transitions per position, matching
training tokens rather than FLOPs/time. Single-law controls match tokens and
updates, with actual wall time and parameter differences reported below.
Small temporary resume/unit fixtures have at most4 total updates per fixture run;
these are correctness checks, not additional songys training or a pilot search.

| Variant | Laws / ticks per token | Rank | Parameters | Active coefficients/tick | Training seconds | State ticks |
|---|---|---:|---:|---:|---:|---:|
| five-1tick | 5 / 1 | 24 | 89,880 | 17,976 | 39.462 | 1,228,800 |
| five-5tick | 5 / 5 | 24 | 89,880 | 17,976 | 126.566 | 6,144,000 |
| one-1tick | 1 / 1 | 24 | 17,976 | 17,976 | 35.406 | 1,228,800 |
| one-matched-1tick | 1 / 1 | 126 | 89,886 | 89,886 | 45.199 | 1,228,800 |

The approximately parameter-matched single law differs by only6 coefficients.
Rank changes optimization/geometry, so its collapse does not isolate the effect
of law count. One seed and this bounded budget cannot establish comparative
architectural superiority. Initialization and before-validation metrics are
recorded per variant. Training history and stdout retain all100-update checkpoints
of loss observations; the saved model is the last update, never the best sample.
Adam, state, absolute ticks/phase, lane offsets, torch RNG, configuration and data
identity are stored. Exact interrupted-vs-uninterrupted numerical continuation is
tested. Starting lane offsets can be inside pairs; concatenation/wrap introduce
artificial transitions. Evaluation begins lanes at whole-pair boundaries.

## Held-out next-token results

Response CE/perplexity includes answer text plus EOS, excludes question/silence
positions, and uses the same112-token vocabulary in every row. All-tick CE includes
SILENCE and is not the same language metric. Character/byte bits include EOS in the
numerator, with NFC answer characters or UTF-8 bytes as explicitly labeled denominator.
They are normalized descriptive code lengths, not results on another tokenizer.

| Variant | Train response CE | Validation response CE | Test response CE | Test perplexity | Test response accuracy | Question false emissions |
|---|---:|---:|---:|---:|---:|---:|
| five-1tick | 1.9843 | 1.9756 | 1.9702 | 7.172 | 40.88% | 0.22% |
| five-5tick | 1.6371 | 1.6362 | 1.6283 | 5.095 | 49.26% | 1.27% |
| one-1tick | 1.7988 | 1.7962 | 1.7912 | 5.997 | 44.32% | 0.98% |
| one-matched-1tick | 3.1695 | 3.1462 | 3.1355 | 22.999 | 0.00% | 0.00% |

Primary validation response CE improves **5.7981
→ 1.9756**. Primary test bits/NFC
answer character=6.0236, bits/UTF-8
answer byte=2.4038.
The 5-tick control has lower CE than five-law immediate output, and the smaller
single law also beats five-law immediate output. These observations argue against
claiming that immediate timing or five laws improved language quality here.

## Actual free generation and semantic evaluation

Frozen sample IDs are the first8 train and first8 test rows, chosen before learning;
these early rows are daily-life examples, not a label-representative QA benchmark.
All16 raw samples/variant, token IDs, Unicode failures, limits and targets are
preserved in variant JSONs. Targets occur only in evaluation, never inference.

| Variant | EOS | Timeout | Mean decoded length | EOS + valid Unicode/controls format | Held-out exact (8 cases) |
|---|---:|---:|---:|---:|---:|
| five-1tick | 100.00% | 0.00% | 10.56 | 93.75% | 0.00% |
| five-5tick | 100.00% | 0.00% | 7.06 | 100.00% | 0.00% |
| one-1tick | 93.75% | 6.25% | 7.19 | 81.25% | 0.00% |
| one-matched-1tick | 0.00% | 100.00% | 0.00 | 0.00% | 0.00% |

“Format” above means EOS/no unexpected control/no unattached Jamo/no replacement
character; it does **not** mean semantically correct Q→A. One of16 primary outputs
contains unattached Jamo. No samples were repaired or replaced. The rank-matched
single-law model has16/16 no-output timeouts; zero false emissions in teacher
forcing is therefore not a quality success.

Primary semantic cases, verbatim:

| Question | Actual generated text | Coherent, relevant response |
|---|---|---|
| 안녕하세요 | 알 어예요. | fail |
| 고마워 | 안 안아은 갓있을 어일 거례요. | fail |
| 오늘 너무 슬퍼 | 안을 가란 할이 었어요. | fail |
| 시험 때문에 걱정돼 | 안아 아일 이있을 거예요. | fail |
| 잠이 안 와 | 아람은 같아요. | fail |
| 밥을 못 먹어서 배고파 | 안은 알어 이예요. | fail |

The six meanings and keyword rules were predeclared in evaluation-plan.json.
The frozen keyword proxy scores0/6 for the primary model. Assistant qualitative
review against the declared meanings also scores0/6; this is not independent human
annotation. The deterministic frequent-answer baseline always returns
**맛있게 드세요.**, appearing20 times in training. It is a
coherent reply to hunger (1/6 meanings), irrelevant to the other5. The frozen
keyword rule falsely rejects that hunger reply; the original proxy is retained,
and this false negative is documented in semantic-review.json without retuning.

The frequent-answer baseline's full-test exact answer rate is
0.0824%, full-test character-multiset F1=0.3094.
It has no neural perplexity. Character overlap is lexical, not semantic correctness.
Model exact-match is measured only on the8 frozen held-out generations above;
its denominator must not be confused with the baseline's full-test denominator.

Low teacher-forced question emission rates do not guarantee fresh-chat silence:
the primary saved model emits4 question-time events while conditioning on
`안녕하세요`. These are exposed as `question_events` in CLI JSON. Fresh-session
phase/context and free-running errors remain serious weaknesses.

## Memory and persistence ablation

For every variant, a two-turn probe first states a bit (0/1) or a Korean name
(민수/지영), optionally idles0/5/50 real ticks, then asks the same current question
across different target values. Compare retained state against zero-at-turn while
preserving the absolute clock/phase. Prior actual generated response and pending
feedback are part of the persistent condition, recorded in full. Idle emissions
are recorded and are not automatically self-fed by raw tick mode.

All variants score **0% target recall with persistence and0% with reset** on these
zero-shot probes. There was no special memory-task learning, so this is failure
to demonstrate semantic recall in these Korean checkpoints, not proof that the
law family cannot learn delayed recall. Numeric boundedness alone is not memory.
The old Experiment001 trained logic evidence is preserved, not relabeled as a
successful one-tick Korean memory experiment. No persistence benefit is established.

## Runtime, decoded Korean throughput and environment

| Variant | Batch1 ticks/s | Original complete NFC chars/s | Separate Hangul syllables/s | Approx FLOPs/tick | Peak process RSS KiB |
|---|---:|---:|---:|---:|---:|
| five-1tick | 21290 | 2585.8 | 1513.6 | 38,064 | 391,392 |
| five-5tick | 20544 | 979.6 | 737.7 | 38,064 | 398,908 |
| one-1tick | 22400 | 2935.5 | 1956.9 | 38,064 | 406,080 |
| one-matched-1tick | 18296 | 0.0 | 0.0 | 181,884 | 406,464 |

Original throughput uses3,000 raw ticks including output detection and5 fresh
complete greeting generations. The separate saved-checkpoint verification uses20
complete greeting generations and counts composed Hangul syllables explicitly;
its outputs/timing are preserved. These very short, repetitive greedy responses
are prompt-specific speed measurements, not useful-chat throughput. NFC characters,
Hangul syllables and decomposed tokens are different units. No-output baselines
receive0 decoded characters/s despite fast idle ticks.

Primary event-step detection mean wall latency=82.0µs;
this measures input injection + law + detector, not guaranteed time to an emitted
or meaningful answer. Approximate FLOPs=4Nr+12N+2r excludes nonlinear evaluations,
coefficient transforms, indexing, allocation, detector and Python overhead.
FP32 primary law coefficients=359,520 bytes; one state=1,408 bytes.
RSS is cumulative process peak (including Python, torch, Adam, previous models and
evaluation), not an isolated per-model memory allocation.

Hardware: AMD EPYC 9V74 80-Core Processor; source managed Codex Cloud, CPU only.
Python3.12.14, torch2.15.0+cpu, NumPy2.5.4,
pytest9.2.0; Linux-6.18.57-x86_64-with-glibc2.41; torch/OMP/BLAS threads1.
Training times exclude acquisition, installation, evaluation and serialization.
The old Endophalon comparison is not matched in task or hardware; no speedup claim.

## Verification and delivery

- `python -m pytest -q`: **112 passed**, including14 new invariant/protocol/resume
  cases (21 combined with the7 original DynaMath tests).
- `python scripts/smoke_models.py`:8 legacy model/config combinations pass.
- `python scripts/verify_backup_consolidation.py`:166 byte-identical reproduction
  files,22 models,9 original heads reachable. This is an offline archive check.
- `python scripts/audit_dynamath_001a.py`:535 pre-existing files byte-identical;
  all original checkpoints, Endophalon artifacts and Experiment001 evidence intact.
  The allowed original edits are explicitly listed. Other remote refs are checked
  against the initial snapshot; no merge or other branch update is performed.
- `python scripts/verify_dynamath_001a.py`:64/64 frozen sample replays identical
  from saved checkpoints, with dataset loading disabled; exact runtime snapshots
  at nonzero phase with pending inputs. No answer lookup is involved.
- Saved-model `generate`, persistent two-turn `chat`, `tick` and snapshot continuation
  exercised as subprocess CLIs; raw stdout and arguments in cli-verification.json.
- Repository-local Filetree fallback generator updates and lints actual hashes.
  Final index/preservation logs and raw metrics accompany this report.

Existing branch URL:
https://github.com/zizonhyeontae218/tele-neuron-engine/tree/feature/dynamath-001a
The delivered commit is supplied in the delivery message; protected-file checks
reference the fixed base commit above. No automatic merge.

## Limits and next smallest research question

This bounded experiment demonstrates executable one-tick dynamics, Korean token
learning, learned silence competition, reproducible checkpoints and actual Korean
emission. It does not achieve coherent QA or useful multi-turn memory. Single-seed
optimization, limited passes, teacher-forcing exposure, token decomposition,
phase/context effects, repeated answers, imperfect near-duplicate grouping and
small generation probes restrict inference. Keeping5 laws does not confer quality
or efficiency benefits in this evidence. The negative rank-matched result is
especially insufficient as an architectural conclusion.

The next useful milestone would be a separately budgeted controlled protocol/
optimization investigation, with learned silence and correct Korean response
transitions as explicit quality gates. No such extra training was started here.
