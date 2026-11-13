# DynaMath 001a

Data credit: **Song Ys (`songys`)**, [original Chatbot_data](https://github.com/songys/Chatbot_data),
MIT; license and source pin in `data/dynamath/001a/`. This is a DynaMath experiment,
not a model developed by the dataset author.

Current package/CLI: `dynamath` (`python -m dynamath` also routes to the Korean CLI).
Historical logs retain the original package name; the cleanup changes no model
coefficients or recorded experimental results. See [cleanup record](../branch-cleanup.md).

Frozen goal/budget: [GOAL.md](GOAL.md). Final results: [report.md](report.md).
Raw outputs, including broken text and silence errors, are in the variant JSON
files. No pretrained language model, learned tokenizer, answer lookup, embedding
or output head is used in DynaMath inference.

From the repository root (CPU only):

```bash
python -m venv .venv
. .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[dev]'
python -m dynamath.korean_data
python -m dynamath.korean generate --question '안녕하세요'
python -m dynamath.korean chat --state-out /tmp/001a-session.pt
python -m dynamath.korean tick --state-in /tmp/001a-session.pt --count 1 --state-out /tmp/001a-next.pt
python -m pytest -q
python scripts/audit_dynamath_001a.py
python scripts/verify_dynamath_001a.py --output /tmp/dynamath-replay.json
python scripts/filetree_index.py lint
```

The CLI prints actual decoded output and diagnostics. `question_events` records
inappropriate question-time emissions; they are never hidden from the results.
Chat preserves one numeric state and absolute tick/phase across turns. Saved
runtime snapshots restore the pending queue, event clock and scheduler. `tick`
without input performs real idle law applications; stopped processes do no work.
`tick --input` encodes text then adds all queued symbols at the next tick; chat
instead injects each token on its own tick.

To reproduce training, use a clean copy with `models/dynamath/001a/*.pt` and the
generated variant JSONs moved aside, then run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m dynamath.korean experiment
```

Existing experiment outputs refuse overwrite. Acquisition verifies the pinned
CSV blob and immutable split; it does not modify the upstream repository.
Only acquisition requires the network; saved-model generation is local and
does not read the dataset, evaluation targets or frequent-answer baseline.

Individual bounded training/resume is also supported, into new paths:

```bash
python -m dynamath.korean train --output /tmp/001a-partial.pt --steps 100
python -m dynamath.korean train --resume /tmp/001a-partial.pt --output /tmp/001a-resumed.pt --steps 2400
```

`--steps` is total, not additional updates. Resume validates data, vocabulary,
law geometry, schedule, optimizer settings and budget; restores all lane states,
absolute ticks/phase, stream offsets, Adam and torch RNG. Wall times differ across
runs; weights/state continuation is bitwise tested on this CPU environment.

Per tick: additive queued input, one original low-rank law at `absolute_tick % 5`,
then immediate output-address competition. No law-private state or output reset.
SILENCE has fixed input/output addresses like every other token. Argmax SILENCE
or a value below .1 produces no outward event; all other eligible winners can
emit on every tick, including question and idle ticks. Ties choose lowest address.
Readout only records the event clock. Feedback is additive on the next tick.

NFC ingest -> NFD codepoints, train-only vocabulary. USER/BOT/EOS/SILENCE/UNK
occupy fixed reserved positions 0..4. Decoding composes NFC but preserves invalid
unattached Jamo. No semantic spelling repair is performed. Unknown codepoints map
to UNK, decoded as `�`; rates and roundtrip failures are recorded per split.

Question protocol: USER and question tokens target SILENCE. Injecting BOT and
applying its law predicts the first answer token on that immediate tick. Each
answer token injected at the following tick predicts the next token or EOS.
EOS is fed on a subsequent tick targeting SILENCE. Teacher forcing, generation,
chat and raw tick all use the same `advance`/law transition. If inference emits
SILENCE unexpectedly during a response, it idles rather than feeding a truth token;
this exposure error remains a limitation. Output caps can leave one feedback token
pending, faithfully retained in snapshots.

Training lanes start at evenly spaced positions of the pair-concatenated stream;
some first positions are inside a pair. State persists through TBPTT, turns and
stream wrap; the wrap is an artificial conversation boundary. Evaluation lanes
begin at whole-pair boundaries and persist across their assigned conversations.
Independent splits/probes start from zero. One-law controls use modulo 1 by design.

The 5-tick control uses explicit `io_ticks=5` and original eligibility boundaries.
The 001a timing control retains explicit 5-tick behavior. Original Experiment 001
weights/code are available in Git history and feature/dynamath-1, outside this
dedicated current tree; use its historical instructions for reproduction.
The default `Session` now uses one-tick input/generation and refractory=1.
Old session snapshots without `io_ticks` restore the legacy schedule.
