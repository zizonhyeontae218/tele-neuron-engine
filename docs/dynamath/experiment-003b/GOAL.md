# DynaMath 003b — word-level English training

Implement and train a fresh 003a Model1 with nine unchanged baseline tanh(z)
ASTs. Preserve historical code, checkpoints, source pins and other branches.
This document supersedes historical no-training rules for this phase only.

## Fixed contract

- NFC English words and separate punctuation; contractions stay together;
  controls USER/BOT/EOS/SILENCE/UNK retain IDs 0..4. Train-only frequency then
  lexical vocabulary, maximum 8192. No BPE, character fallback or pretrained data.
- Pinned 003a DailyDialog, full splits after whole-dialogue and exact Q/A
  leakage filtering with test > validation > train priority. No 512 cap.
- Seed 43; Adam; batch 8; TBPTT 16; exactly 50,000 target updates; peak LR
  2e-4, warmup 1000, cosine to 2e-5; clip 1; latent 64; rank 12.
- One S, one m; nine independent formulas, one/tick; one additive input on
  tick 1 of each three-tick slot; inspect ticks 2/3; actual FIFO feedback;
  retain state after output. No learned embedding or output head.
- Restart S/m/session clock independently at every dialogue boundary;
  keep them through all turns and TBPTT detach. No shortest-lane truncation.
- Response CE weight .75 and SILENCE CE weight .25 (different from 003a).
  Teacher-forced evaluation and closed-loop Session generation are distinct.

## Implementation plan and ownership

`src/dynamath/v003b/{tokenizer,data,model,training,evaluation}.py` own token
rules/audit, frozen data and lane streams, unchanged equations with independent
lane clocks, atomic resumable training, and metrics/FIFO memory evaluation.
`scripts/{train,evaluate}_dynamath_003b.py` are entry points. Configuration lives
in configs/dynamath; evidence in this directory; ignored checkpoint binaries in
models/dynamath/003b. Actions chains bounded jobs through uploaded artifacts.
Shared Config only gains ordered-list typing; old strings remain supported.

## Acceptance

- [x] Tokenizer metadata, frequencies, split IDs/removals/hashes and full OOV.
- [x] Focused and historical tests; exact same-environment restart test; CI.
- [ ] Actual 50,000 updates, resumable checkpoints at 5k/20k/35k/50k.
- [ ] Full validation metrics at milestones, table and graph; fixed last-step
      final selection, then full test once (never tune against test).
- [ ] >=20 fixed English FIFO prompts including multi-turn follow-up;
      raw outputs, failures and S-only/m-only/both reset memory comparisons.
- [ ] Parameter/state/cost/time/memory measurements and reproducible reports.

If bounded execution stops, preserve actual completed step and remaining work;
implementation alone does not satisfy completion. No automatic claim of language
understanding, semantic memory or formula-evolution superiority.
