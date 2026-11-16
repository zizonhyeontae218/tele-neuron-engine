# DynaMath 003b — reproduction and status

See `GOAL.md`, `architecture.md`, `tokenizer-report.json`, `training-report.md`
and `results.json`. Completion requires implementation, **50,000 actual updates**,
full evaluation, raw free generation and reproducible artifact preservation.
Partial checkpoint status is never equivalent to completion.

## Install and run

Python >=3.11; CPU or CUDA PyTorch >=2.5, NumPy, pytest, matplotlib.
The executed CPU versions are pinned in configs/dynamath/environment-003b.txt;
the Actions runner installs these constraints for repeatable segment resumes.

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/train_dynamath_003b.py --seconds 14400
python scripts/train_dynamath_003b.py --seconds 14400 --resume models/dynamath/003b/latest.pt
python scripts/evaluate_dynamath_003b.py --checkpoint models/dynamath/003b/latest.pt --mode validation
python scripts/evaluate_dynamath_003b.py --checkpoint models/dynamath/003b/latest.pt --mode generate
python scripts/evaluate_dynamath_003b.py --checkpoint models/dynamath/003b/latest.pt --mode memory
python scripts/filetree_index.py update
python scripts/filetree_index.py lint
```

Use an appropriate CUDA PyTorch install and `--device cuda` on a GPU host; auto
uses CUDA if available, otherwise CPU. GPU execution is supported in code but
only the actually tested device is claimed in results. No pretrained embedding
or alternate tokenizer is installed. `--seconds` caps each segment, never the
50,000-step target. Resume the same checkpoint until complete.

DailyDialog is fetched to ignored work/dailydialog/data.zip from the same pinned
Hugging Face revision and SHA-256 as 003a. Attribution/license (CC BY-NC-SA 4.0),
IDs, removed dialogues with reasons, before/after overlaps, OOV, hashes and full
train-only tokenizer frequencies/rules are saved independently. No raw corpus
archive is added to Git. tokenizer.json reproduces exact IDs and decode rules.

## Model usage

```python
from dynamath.v003b import Model, WordTokenizer
from dynamath.v003b.evaluation import generate
model, payload = Model.load('models/dynamath/003b/latest.pt')
tok = WordTokenizer.load('docs/dynamath/experiment-003b/tokenizer.json')
result, session = generate(model, tok, 'Hello, how are you?', response_slots=96)
print(result['text'])
next_turn, session = generate(model, tok, 'What did I just ask?', 96, session, turn=1)
print(next_turn['text'])
```

Only trusted checkpoints should be loaded. Original character checkpoints still
load with their versioned model classes; no character model is converted to
word dimensions. Free output may be repetitive, silent, malformed or incorrect;
raw output is retained without repair.

## Actions continuation

`DynaMath 003b word training` starts on the initial workflow push to this branch
or via workflow_dispatch. Four sequential reusable jobs each budget four hours
of training and reserve time for full validation, final test and artifact upload.
Later jobs skip when the previous segment reaches 50,000. Every new segment
downloads the immediately preceding checkpoint artifact; a missing checkpoint
fails explicitly rather than starting over. Segments upload checkpoints and
evidence with 90-day retention, and commit small reports/index to this branch.

For manual recovery, dispatch with previous `resume_run_id` and `resume_segment`
matching the artifact suffix, e.g. run ID and `0` restore
`dynamath-003b-checkpoints-0`. First recover/download the artifacts if a run failed.
Starting with both inputs blank intentionally begins from seed 43; do not do
this when continuation is intended. concurrency prevents overlapping main runs.
If four segments do not reach 50,000, results remain incomplete and another
explicit dispatch can resume from the last successful artifact.

```sh
gh workflow run dynamath-003b-train.yml --ref feature/dynamath-003b \
  -f resume_run_id=RUN_ID -f resume_segment=3
gh run download RUN_ID -n dynamath-003b-checkpoints-3 -D models/dynamath/003b
```

The final step was preselected. Full validation is performed at 5k/20k/35k/50k;
full test is only run after saving immutable final selection. The evaluator CLI
does not expose interim test scoring. The final runner preserves test JSON and
refuses to overwrite it with a different selected checkpoint.

`DynaMath 003b full-corpus final evaluation` waits for the recorded production
run and downloads its last cumulative checkpoint artifact. It adds actual FIFO
generation for **every** Q/A pair in the complete validation split at all four
milestones and every test pair for the frozen final model. This supplements the
22 fixed English prompt samples and the full teacher-forced metrics. This bulk
generation protocol uses a fresh Session per Q/A pair, so a runaway feedback
queue from one failed answer cannot consume later prompts' compute budget.
It measures independent-prompt behavior; teacher-forced dialogue context and
the explicit multi-turn/follow-up memory probes retain their session state. Corpus
results preserve IDs, text, controls, premature emissions, EOS failure and queues
in `closed-loop-<split>-<step>.json.gz`; only these corpus files omit the very
large per-tick trace. Fixed-prompt samples retain that trace. The test model is
never reselected or retrained. To recover this stage without any training:

```sh
gh workflow run dynamath-003b-finalize.yml --ref feature/dynamath-003b \
  -f training_run_id=RUN_ID
```

The waiting job has a five-hour bound plus evaluation time; if training takes
longer, dispatch that final-evaluation workflow after training completes.
After successful final evaluation, checkpoint assets are archived without an
expiry in the owner-visible release draft `dynamath-003b-checkpoints`. The draft
is not published automatically; the previous branches and public releases stay
untouched. Repeating finalization reuses cached corpus measurements and preserves
the fixed checkpoint/test selection.
