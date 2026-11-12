# DynaMath 1 — run and reproduce

A real but very small character model: five mathematical transition laws on one
persistent address state. It learns and generates; it is not yet a useful general
chatbot. See [architecture](architecture.md), [experiment and limits](experiment-001.md),
[decisions](decisions.md), and [original execution brief](handoff.md).

## Install

From this repository's `feature/dynamath-1` checkout, Python >=3.11:

```bash
python -m venv /tmp/dynamath-venv
source /tmp/dynamath-venv/bin/activate
python -m pip install -e . -r requirements.txt
python -m pip install -r docs/dynamath/requirements.txt
```

Exact measured runtime was Python3.12.14, NumPy2.3.5, torch2.15.0+cpu, pytest9.2.0,
Linux x86_64. For that CPU build, install NumPy/pytest and use PyTorch's CPU index:

```bash
python -m pip install numpy==2.3.5 pytest==9.2.0
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

The CLI also sets torch CPU threads=1 and deterministic algorithms. Existing
Endophalon modules do not import torch. The separate DynaMath dependency file keeps
the original pyproject.toml byte-identical for the legacy reproduction audit.

## Use saved weights

```bash
python -m tele_neuron.dynamath generate
python -m tele_neuron.dynamath generate --prompt 'user: hello
bot:'
python -m tele_neuron.dynamath chat
```

The second command contains a literal newline inside the shell-quoted prompt.
Chat accepts the small training alphabet (lowercase English and ` .:?` plus newline).
Unsupported characters are rejected. Input `hello`, `what are you?`, or other allowed
text; responses come from the current numeric state, with no response lookup.
Each new chat turn continues the existing state. `/quit`, Ctrl-C or EOF ends chat.

## Tick, save and resume state

```bash
python -m tele_neuron.dynamath generate --state-out /tmp/dynamath-state.pt
python -m tele_neuron.dynamath tick --state-in /tmp/dynamath-state.pt --count 7 --state-out /tmp/dynamath-idle.pt
python -m tele_neuron.dynamath generate --state-in /tmp/dynamath-idle.pt --prompt 'user: hello
bot:' --max-chars 80 --max-ticks 1500
python -m tele_neuron.dynamath tick --seconds 2 --hz 100
```

`tick --count` executes a batch without waiting for wall time. `--seconds` is an
optional bounded wall-clock loop. No process means no ticks. Both can emit raw
address events; only `generate`/`chat` provide additive token feedback.
Max-ticks includes prompt conditioning/alignment. A budget that is too small can
produce no output; there is no fallback string or forced output.

## Train

Reproduce all first-experiment runs (overwrites only DynaMath artifacts in --root;
copy this branch to a separate checkout if you want to retain the published run):

```bash
python -m tele_neuron.dynamath experiment --root .
```

This runs exactly 600 five-law logic steps, 600 single-law comparison steps and
800 character steps. `configs/dynamath/experiment-001.json` is a descriptive record
of these code defaults, not a dynamically loaded configuration. It writes checkpoints
under models/dynamath and metrics under docs/dynamath. No Endophalon entry point is used.

Train only the character model to a new checkpoint:

```bash
python -m tele_neuron.dynamath train --output /tmp/new-characters.pt --steps 800
python -m tele_neuron.dynamath train --resume /tmp/new-characters.pt --output /tmp/resumed-characters.pt --steps 900
```

`--steps 900` means total900 (100 more after800). Training state, offsets, RNG,
optimizer and phase are restored; corpus hashes must match. Validation is evaluated
and logged but never used for gradient updates or checkpoint selection. The logic
.pt files load as models via `DynaMath.load`; the character resume command deliberately
rejects them. Their experiment is reproducible from the fixed seed, with a finite budget.

## Verify and benchmark

```bash
python -m pytest -q tests/test_dynamath.py
python -m tele_neuron.dynamath benchmark
python scripts/filetree_index.py lint
python scripts/verify_backup_consolidation.py
```

`benchmark` measures batch1 ticks including output checks, plus a separate greedy
feedback generation run. It does not measure model load, useful dialogue latency,
networking, GPU acceleration or Endophalon spring speedups. Peak RSS is process-wide,
not weight-only memory. Published measurements and their scope are in experiment-001.md.

## Programmatic use

```python
from tele_neuron.dynamath import DynaMath, Session
model, metadata = DynaMath.load('models/dynamath/characters.pt')
session = Session(model)
session.consume('user: hi\nbot:')
state_before = session.state.clone()
events = session.tick(100)  # real idle transitions, no state reset
session.save('/tmp/live-session.pt')
resumed = Session.load('/tmp/live-session.pt')
text = resumed.generate('user: hello\nbot:')
```

`enqueue(text)` adds all queued characters at the next tick; `consume`/`generate`
pace text one character per five ticks. `model.inject(state, tokens, amplitude)`
provides signed additive numeric injection without replacing addresses.
