# DynaMath series branch cleanup

User-authorized on 2026-11-13 (KST). Starting experiment commit:
`d5127ab7ee1a4a0e1001e534c6e5048c7ef3cd72`.

This branch now serves as the DynaMath series home, with DynaMath 001a as the
current baseline. The root README displays DynaMath, current results, run commands
and paths for future versioned experiments. No new training was run in this phase.

## Scope

Removed 531 unrelated or superseded tracked files: other model families, their
workflows/configurations/apps/tests/archives, and old toy English Experiment 001
training assets and checkpoints. Redundant inventory/backup logs were removed.
Earlier files remain recoverable at the starting commit and existing historical
branches/tags. No history rewrite, branch deletion, merge or default-branch change.

Retained the four 001a saved models, including the failed single-law control,
source/license/split/tokenizer, frozen training configuration, raw metrics,
sample/semantic/memory failures and original experiment goals. The focused audit
pins 24 immutable artifacts to SHA-256 values from the starting experiment commit.
The original report's old verification counts are explicitly marked historical.

## Independent package and workflow

- `src/dynamath` replaces the former nested package. Its initializer imports only
  DynaMath, so importing the model no longer depends on unrelated model code.
- `python -m dynamath` and the installed `dynamath` console command route to the
  current Korean generate/chat/tick/train/experiment CLI.
- The old English trainer is removed. Shared `setup`, `dump` and address loss
  helpers preserve their original mathematical behavior.
- Packaging is named `dynamath`, with PyTorch as the runtime requirement and pytest
  as the development extra. Removed unused plotting/physics dependencies and assets.
- CI checks DynaMath tests, all frozen checkpoint replays, artifact hashes,
  the generated index, and the current CLI. Replay verification writes to scratch;
  it never overwrites frozen experiment timing/results.
- The numeric law family, checkpoint format, data split and weights are unchanged.
  Explicit 5-tick and one-law controls remain reproducible.

## Verification

- 22 current tests pass: 20 retained DynaMath invariant/protocol/resume tests plus
  two installed-CLI tests outside the checkout, including persistent chat/tick.
- All 64 frozen sample generations match exactly from the independent package;
  runtime snapshots with pending feedback restore exactly for all four variants.
- All 24 pinned experiment artifacts remain byte-identical; other remote refs
  match the pre-cleanup snapshot.
- A built wheel is installed to an isolated target and exercised with Python's
  isolated mode. The import resolves to that wheel, the old namespace is absent,
  and the saved model generates the same Korean result without dataset access.
- Local Markdown links, whitespace, generated index and focused CI are checked.
  Current local evidence: cleanup-tests.log and cleanup-verification.json. Remote
  CI outcome and delivery commit are supplied in the delivery message.

The scientific result is unchanged: Korean emission works, meaningful Q/A and
multi-turn recall fail the current probes, and five-law superiority is unproven.
Future model changes need a separate goal, budget and versioned evidence.
