# Endophalon 3 — independent A/B release

A/B each completed 120 generations and passed saved-model audits. Current models are models/endophalon3_software/A.json, B.json and current.json. Separate states/clocks/RNG and local histories, explicit predicted-bit software registers. Old 3 variants/artifacts removed from active tree; experiment results only in docs/endophalon3_development.md. No history rewrite or unrelated branch changes.

Confirmation: A 89.17% (gate False), B 76.61% (gate False). Budget completion is separate from accuracy success. No further training was started.

Both overall user targets (A >=90%, B >=80%) and the strict overall90/all-ops85 gates were missed. A recall2 and B parity3w/xor_mem remain weak. Original Actions postprocessing failed; unchanged trained artifacts were verified and recovered without retraining. Details: docs/endophalon3_development.md.
