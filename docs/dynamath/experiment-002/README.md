# DynaMath 002

Contract: [CODEX_HANDOFF.md](CODEX_HANDOFF.md). Additional user-authorized Model 3
and Model 1 variants change scalar ownership only; all have nine independent laws.
Model 3 cross-read permits any law to read all three block scalars; local reads
only its block scalar. A law only replaces the active owner's scalar.

```bash
python -m pip install -e '.[dev]'
python -m dynamath.v002 train
python -m dynamath.v002 evaluate --checkpoint models/dynamath/002/model9.pt
python -m dynamath.v002 generate --question '안녕하세요'
python -m dynamath.v002 chat
python -m dynamath.v002 tick --ticks 18 --snapshot work/session.json
python -m dynamath.v002 tick --restore work/session.json --ticks 9
python -m pytest -q
python scripts/audit_dynamath_001a.py
python scripts/filetree_index.py update
python scripts/filetree_index.py lint
```

Train uses the pinned Song Ys CSV (downloaded if absent), legacy tokenizer/splits
unchanged, no learned embedding/head. Generation/chat/tick require only checkpoint;
no data path or answer lookup. Snapshot JSON is numeric runtime state, not an
executable export of model weights. Fixed PyTorch checkpoint is also required.
The train CLI accepts `--resume <checkpoint>` for matching topology, restoring
optimizer, torch/Python RNG, stream position, S, m, absolute tick and data hash.
The primary checkpoints precede the resume metadata extension: completed updates
are recovered exactly as `tick / 72`, and Python RNG is reconstructed from seed
43 (the primary loop never consumed Python random draws). Optimizer, torch RNG,
S/m/t/position and data hash were saved. New checkpoints save Python RNG and update
count directly. A completed-budget resume performs zero further updates.

## Mathematical and scheduling decisions

Read old scalar(s), perform the low-rank nonlinear state update, then evaluate
state-linear scalar write candidate/gate. Candidate `e*tanh(w*S+b)` is bounded
by [-e,e]. Hard gate is `sigmoid(g*S+b)>=.5`; training uses a straight-through
surrogate, forward values are still exact retain/replace. Gate coefficients are
ordinary learned coefficients. Scalars are tensors in session state, never optimizer
parameters. Initial values are float32 representations of repeating [0,1,e].
One scalar per owner, no extra latent memory network. State dimension is two
fixed disjoint vocabulary address regions plus 64 latent coordinates.

At block phase 1 target next-token/BOT-answer/EOS as supplied by legacy pairs;
at phase 2 target SILENCE. Both slots contribute equally to loss. There is no
runtime phase-1 emission guarantee or phase-2 suppression. A teacher-forced token
replaces an observed prior emitted token at next admission during training;
closed loop admits only actually emitted symbols and external inputs. Failure
to emit therefore stalls, and double emission can build FIFO backlog. BOT is an
external delimiter, never a privileged decoder mode. All real emissions, including
unexpected controls and EOS, are fed back by the same FIFO. No queue drops/caps.
External question burst and BOT append before running; later feedback joins tail.

Each generation sample starts a fresh independent session, as an evaluation case;
chat preserves state, scalars, tick and queue across turns. Runtime never resets
at response boundaries. Idle and snapshots preserve scalars. Training lanes persist
across pairs; lane tails are truncated to shortest lane (reported stream length).
TBPTT is 24 input slots / 72 ticks; this is shorter than a typical full pair and
limits long-range credit assignment. CPU wall cap may stop before one corpus pass.

Recall is a distinct trained synthetic task over fixed bit/query addresses, with
0/5/50 idle ticks and alignment to next legal query slot. It reads after query's
third law. Four arms reset immediately after encoding the bit, never reset tick.
Held-out seed changes bit ordering; there are only two classes, so this is a small
binary retention diagnostic, not generalization to novel facts or Korean memory.

### Reproduction scope

Primary runs used the initial runner prior to resume metadata extensions and a
numerically equivalent hard-assignment cleanup. `pilot-failure.log` records the
single implementation pilot that completed computation but failed to serialize
because torch initialized before NumPy was installed. It produced no checkpoint
or selected score; primary starts again from seed 43. In the final code the gate
is `hard + (probability - probability.detach())`, and replacement is
`(1-gate)*old + gate*candidate`, avoiding subtraction-rounding in the initial
`old + gate*(candidate-old)` form. Inference is exact retain-or-replace.
Saved-token replay checks any effect on published outputs rather than claiming
bit-identical coefficient retraining across these source revisions.

To reproduce the published primary arithmetic, run
`python scripts/reproduce_dynamath_002.py` in a separate checkout. This explicitly
archives the original floating-point expression order; it uses the same frozen
config, seed, training schedule and evaluation. Normal `python -m dynamath.v002
train` uses the final exact replacement implementation.
