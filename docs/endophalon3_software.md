# Endophalon 3 — current independent A/B

One self-contained runtime package, two independent 144-node/2-substep/16-zone models.
Software selects A/B, pauses the inactive session, and passes actual output bits through
explicit registers: `A xor 1 0 -> r0`, then `B and @r0 1`. Recall uses the selected
model's local history. No state transfer, hidden oracle or automatic correctness repair.

Models: models/endophalon3_software/A.json and B.json; combined current.json.
Each completed 120 generations / 2881 candidates, fixed A optimizer seed2/B seed1,
physical/data seed1. Training-only Ridge heads; validation-only theta selection.
Fresh confirmation and per-op outcomes are in the [development history](endophalon3_development.md)
and current independent120 evidence. Earlier versions/model artifacts were removed.

Commands: docs/model_commands.md. Current configs retain the initialization theta
needed to reproduce this training; they are not runnable old model/head archives.
Single-run scores do not establish chained-program accuracy or infinite-horizon stability.

Actual confirmation: A 87.01% baseline -> 89.17%, B 75.10% -> 76.61%. A90/B80 targets and strict per-op gates were missed. A recall2 is 72.23%; B parity3w 51.36%, xor_mem 55.22%, recall2 66.52%. The original workflow failed in postprocessing; verified original artifacts were recovered without any extra training. See development history for checksums, audit portability and limitations.
