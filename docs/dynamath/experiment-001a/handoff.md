# DynaMath 001a

> **데이터 출처 및 크레딧 — Song Ys (`songys`)**  
> 이번 실험의 한국어 학습 데이터는 Song Ys가 공개한 [Chatbot_data](https://github.com/songys/Chatbot_data)의 `ChatbotData.csv`를 사용한다. 원본 데이터의 출처·저작권 및 MIT LICENSE를 명확히 기록하고, 데이터 제공자를 실험 보고서 첫머리에도 명시할 것. 외부 데이터의 출처와 연구용 모델의 개발 주체를 혼동하지 말 것.
>
> **실험명: DynaMath 001a** · 기존 DynaMath 1(최초 실험)의 직접적인 후속 실험 · 한국어 songys 데이터 및 매 1틱 출력 검사


## Role and execution

You are implementing and running the *next actual DynaMath experiment*, not merely drafting an architecture. Use the existing repository `zizonhyeontae218/tele-neuron-engine`, starting from `feature/dynamath-1`; inspect the branch's code and committed Experiment 001 evidence first. Create an isolated branch, provisionally `feature/dynamath-001a`, and preserve all legacy branches, checkpoints and Endophalon artifacts. Never merge to master automatically. If that ref exists, inspect before modifying; do not clobber others' changes.

This handoff records two user decisions:
1. **Check output eligibility immediately after *every single tick***. Remove `t % 5 == 0` as an output eligibility gate. The five independently learned laws still rotate once per tick, `k = t % 5`, and all five operate on one persistent information state.
2. **Replace the 50-pair toy English dialogue experiment with a serious Korean-data experiment**, using the original `songys/Chatbot_data` corpus (~11,876 question/answer pairs), not merely selected anecdotes.

The central scientific question: can a persistent mathematical state with five sequentially applied learned laws produce and learn Korean Q/A at larger scale under immediate per-tick address events? Do NOT claim GPT-2 performance, consciousness, useful chatbot quality or computational superiority without matching evidence. The code must run, learn, save/resume, generate Korean, and publish reproducible metrics and failures.

## Locked conceptual invariants

- One shared numeric information-address state `S`, retained across ticks and emissions.
- Each tick first adds pending input to addressed state values, then applies exactly one learned law, then checks output address values (regardless of phase).
- Exactly five independently parameterized laws rotate F0→F1→F2→F3→F4→F0, without separate hidden states, springs, collision physics, trained initial state, external LLM, or lookup answers.
- Only law coefficients may learn. Fixed input/output address mapping, deterministic tokenizer/decoder, and fixed event policy are allowed; an external trained token embedding/head must not silently replace address dynamics.
- If output is emitted, the observed state must remain unchanged by readout. Self-feedback adds emitted token to its input address at a **subsequent** tick; never force-assign an address value.
- Idle ticks are real law applications even without input; preserve absolute phase, pending queue, events and state in checkpoints.
- Separate the periodic law choice from event eligibility. Event detector checks every tick; refractory should be no more than one tick for the one-tick comparison. Prevent double emission when polling the same tick twice, but do not artificially throttle output to a five-tick boundary.

## Corpus and provenance

Primary canonical dataset: `https://github.com/songys/Chatbot_data`, original `ChatbotData.csv` (fields `Q,A,label`). Original GitHub repo contains an MIT LICENSE; Hugging Face mirror `songys/ChatbotData` declares CC-BY-SA-4.0 instead. Use **the original upstream GitHub artifact**; record exact source commit SHA, CSV blob SHA/hash, acquisition date, license text and attribution. GitHub blob SHA seen during planning: `4cc2bfa9a124991dc5ca80f5da1786f4162f46e1`; recheck it. Verify true parsed row count against the documented 11,876; record invalid/empty/duplicate rows. Do not silently mix the HF mirror or alternative versions. If in doubt, preserve attribution and avoid redistributing a transformed dataset.

Corpus is small compared with general-purpose language-model training but >200x the earlier 50 Q/A pairs. Source includes daily life/negative-breakup/positive-love labels; some answers repeat across near-duplicate questions. Train/val/test **must split at the Q/A-pair level before concatenation** and group near-identical normalized Qs to reduce leakage. Also report repeated exact A and Q-A pairs *across* splits and the distribution of labels. Keep a frozen held-out test set (e.g. 10% after deterministic grouping; 10% validation, 80% train, with label balance insofar as feasible). Do not select hyperparameters or outputs on the final test set. Prefix and separate turns explicitly, with a dedicated end-of-response marker.

## Korean tokenization — explicit pre-training decision

The original character model maps distinct characters to disjoint fixed input and output addresses and cannot ingest Korean out of the box. Implement an **untrained deterministic Unicode tokenizer** before training: baseline choice is NFC ingest, NFD Hangul decomposition to modern Jamo/codepoints, plus fixed control markers (`USER`, `BOT`, `EOS`, optionally `UNK`). Preserve reversible normalization/decoding for ordinary corpus text, including Korean syllables and punctuation; report exact training vocabulary size, normalization loss/unknown rate, token lengths, and round-trip tests. Build vocabulary from train only; valid/test unseen chars must use a documented policy (unknown token or excluded examples, with measured coverage). Do not learn a separate tokenizer or use a pretrained model. If a different fixed tokenizer (such as UTF-8 bytes) yields better coverage/compute tradeoffs, you may choose it **before** model training, but document the decision and preserve byte-for-byte reversible decoding tests for Korean. Make active input/output addresses equal the fixed vocabulary size, and allocate additional latent addresses without letting softmax select latent positions.

Because Korean may require 2–3 decomposed symbols per syllable, count tokens/s, normalized Korean characters/s, memory and actual generated decoded characters; don't confuse a token with a syllable.

## Two-stage DynaMath 001a

### Stage A — event-timing semantic isolation

Starting from DynaMath1, change `Session.poll()` and associated scheduling so every tick checks event eligibility immediately after the law applies. Test all five phases, condition crossing, ties, silence, duplicate polling, constant suprathreshold state, idle, input addition, repeated symbols, state checkpoint restoration, absolute phase, and output not mutating state. Write *before/after behavior tests*. Existing 5-tick-trained weights need not produce useful text at 1-tick timing: report incompatibility as an expected distribution shift, not training failure. Do not change law geometry/weights while isolating this behavior. Include explicit assertion: at tick 1 with a qualifying address and no refractory conflict, an event can occur.

### Stage B — larger Korean learning experiment

Train from scratch on the fixed songys split, with a **one-tick-per-token** training/inference protocol and explicit `BOT` control input. During user-question ticks, label 'silence'; after the `BOT` boundary, allow the first response token on the immediate next tick. A generated response token is fed by additive input in the next tick and predicts the following token. Train an actual silence-vs-emit policy using output-address thresholds or a *fixed mapped dedicated silence address* in the state; don't merely suppress external output by mode and then claim spontaneous appropriate silence. Supervise EOS emission and make consecutive repeated characters legal. Verify the same protocol in teacher-forced training, free generation, `chat` and `tick` CLI; avoid train/inference exposure mismatch as far as possible. Phase must continue across tokens and whole conversations, not reset at every character or turn.

Keep the five-law low-rank family as baseline; capacity may scale to the token vocabulary and task, but record N, rank, parameters, active coefficients/tick, FLOPs/tick, latency, training memory, and training tokens. No pre-trained neural language model. BPTT / TBPTT permitted; implement deterministic train checkpoint/resume, clip gradients, no hidden test leakage. Full data must be eligible for training, even if compute budgets limit epochs/updates. Set explicit compute cap before full run; make a small pilot for bugs only, then one bounded run. Do not keep rerunning until sample text looks attractive.

## Minimum controls and comparisons

- Baseline 1: five laws, original 5-tick IO schedule (where feasible) on exactly the same Korean split/tokenizer/budget, to isolate immediate emission effects; otherwise report why a matched control is not attainable.
- Baseline 2: one law, one-tick protocol, matching training tokens/time and **reporting** parameter differences. Prefer an additional approximately parameter-matched single-law control if the cost is modest. No conclusion on five-vs-one if controls are not comparable.
- Non-neural data baseline: deterministic frequent-answer or simple n-gram baseline for context on the scale of the task, clearly distinguishing retrieval-style scores from model perplexity.
- Ablation on persistent state: reset-at-turn vs persist-across-turn in an actually defined multi-turn probe. For meaningful memory tests, use controlled sequences where current question alone is insufficient, and distinguish carryover from nuisance influence.

## Evaluation, prior to observing test results

1. Source validation: rows, labels, duplicates, splitting/leakage reports, tokenizer and coverage. The first lines of README, experiment report and model card must credit Song Ys and link the original source.
2. One-tick runtime invariants from Stage A, especially first-tick output and no mutation.
3. Character/token next-step CE and perplexity (only comparable for identical tokenizer) on train/val/frozen test; report normalized bits per Korean character or per UTF-8 byte as secondary comparison.
4. Free generation for a fixed list of Korean prompts from both train and held-out; publish exact samples including repetitions, failed Unicode and no-output cases. Report EOS rate, silence false-positive rate, Q→A format success, output lengths and timeout rate. Never substitute targets for predictions.
5. At least one semantic Q/A evaluation with predeclared rubric and an appropriate simple baseline; mere teacher-forced 60–70% next-symbol accuracy is not Q/A quality.
6. Memory: delayed bit recall + 2+ turn Korean toy probe, and idle-time retention after gaps. Idle numeric boundedness alone is not semantic memory.
7. Compute: batch1 wall-clock tick throughput, input-event response latency, complete Korean **decoded characters/s**, FLOPs estimate with exclusions, CPU/RAM/hardware/threads, source environment. Match old Endophalon comparison only if truly same task and platform.
8. Multiple random seeds only if within declared budget. If one seed only, explicitly label inference limited.

## Repository delivery and stopping conditions

Create isolated branch from `feature/dynamath-1`, leave Experiment 001 checkpoints and old model families intact. Add all code, unit tests, corpus acquisition/preprocessing script, immutable split manifest, tokenizer metadata, training run script/config, output model/checkpoints (or GitHub Release if size prohibits regular Git), raw sample logs, metrics JSON, execution timestamps/hardware and descriptive report under new versioned `docs/dynamath/experiment-001a*` paths. Consistently use **DynaMath 001a** as the model/experiment designation; do not present it as “Experiment 002” or use a songys-branded model name. Include original songys license/attribution and source pin. If large dataset not committed, acquisition must be reproducible. Run existing tests and preservation audit; make no unrelated branch edits. Report exact commit and branch URL. State separately: (A) Stage A event logic verified, (B) Korean train ran/completed, (C) quality and limitations. Do NOT call DynaMath 001a complete until model can actually generate decoded Korean from the saved checkpoint with no answer lookup, and record what could not be finished or reproduced.

**The core idea is not a 5-layer Transformer:** it is a single perpetual shared information-space state governed by five cyclical learned absolute laws, with additive external perturbations and output-address events checked *every tick*. Preserve this throughout architecture choices.
