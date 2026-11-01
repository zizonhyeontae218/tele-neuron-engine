# Endophalon 3 — joint A/B training closeout

All three predeclared joint trials completed the 120-generation limit (2,881 theta evaluations each; 8,643 total). **0/3 passed the A/B accuracy gates.** Training completion does not establish the requested performance. No further seed, coefficient or test-driven restart was performed.

Each saved card uses one physics system and one train-fitted linear readout for interleaved A/B instructions, including memory across level switches. The 144-node, 2-substep, 16-zone invariants, original objective, physical/data seed 1, regular timing and readout delay 12 were retained. Each trial warm-started the corresponding original B-card theta; it did not resume the old optimizer distribution. Binary search seeds 1/2 and one-hot search seed 1 were fixed before training. All old cards/configurations/logs remain immutable.

| Joint trial | Validation pooled | Original AB test pooled | Fresh pooled | Fresh A | Fresh B | Both-level gate |
| --- | --- | --- | --- | --- | --- | --- |
| ab_binary_s1 | 81.25% | 71.98% | 72.32% | 73.89% | 70.67% | FAIL |
| ab_binary_s2 | 81.67% | 73.85% | 73.02% | 73.59% | 72.43% | FAIL |
| ab_onehot_s1 | 86.04% | 75.42% | 76.77% | 80.87% | 72.48% | FAIL |

Thresholds are >=90% overall for **each** level and >=85% for **each** operation within each level. Full operation-level values are retained in each card's `metrics`; pooled metrics do not substitute for these gates. Fresh streams 20000–20063 were declared in the plan before training and evaluated once after theta selection. Original joint AB splits are train 0–23 / validation 24–31 / test 32–47. Selected mean validation accuracy is 81–86%, while fresh accuracy is 72–77%; this gap and the weak recall/parity results limit the claims. Confirmation has now been observed and cannot be treated as untouched for a later adaptive experiment.

No candidate passed the validation gates. All trials exhausted the finite 120-generation budget, without triggering the 40-generation plateau rule. This does not prove that more compute or a different method cannot succeed. It also does not demonstrate slow-region recall2 benefit or long-horizon stability. Seeds vary the optimizer only, not physical/data generation; no independent-data or broad generalization claim is made. The earlier negative routing follow-up and its closed B expansion gate remain historical results; this joint study was separately authorized by the owner.

## Evidence and reproduction

- Training source commit: `c70f262`; [Actions run 41287301037](https://github.com/zizonhyeontae218/tele-neuron-engine/actions/runs/41287301037).
- Original output, optimizer/RNG checkpoints and logs: `docs/endophalon3_joint_runs/20261101/ab_*/`.
- Canonical byte-identical model copies: `models/endophalon3_joint/ab_*.json`.
- SHA-256 manifest: `docs/endophalon3_joint_runs/20261101/artifact_manifest.json`.
- Frozen procedure and stop rules: [joint plan](endophalon3_joint_plan.md).
- Read-only replay: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/audit_endophalon3_joint.py` after installing the project. It checks serialized heads, training-only refits, validation fitness, exact level/operation accuracies, finite states, checkpoint consistency and artifact hashes. It never retrains or rewrites published output.

GitHub CLI artifact download encountered a redirected-storage 403. The GitHub connector downloaded all three artifacts successfully; their original model/checkpoint/log bytes were recovered. The local audit initially needed the project source path; the audit script now resolves it directly. Training outputs were not altered to repair the audit environment.

The first checkpoint audit compared `encode(theta)` with the raw optimizer vector and failed for binary seed 2 (one scalar, max difference 0.0443435). The existing encoder clips probabilities to [1e-4, 1-1e-4], so it is not an exact inverse near a bound. The correct audit decodes the checkpoint vector and requires exact equality with the saved theta. Models, checkpoint vectors, physics and training results were not changed.

## Closeout verification

Full pytest: **95 passed**. All three saved heads exactly reproduce train/validation/original-test/fresh accuracies by level and operation; train-only refits, validation fitness, finite states and checkpoint theta checks pass. All three downloaded ZIP SHA-256 values match GitHub artifact digests, and every extracted model/checkpoint/log matches the original ZIP bytes. Existing 166 backup reproduction files and nine reachable original heads still pass integrity verification. Detailed check record: `docs/endophalon3_joint_runs/20261101/closeout_verification.json`.

The external Filetree skill generator is unavailable in this runtime. A separate local generator updates the index and actual SHA-256 ledger; an independent tracked-path/hash check verifies it. This is not a claim that the unavailable skill CLI lint ran.
