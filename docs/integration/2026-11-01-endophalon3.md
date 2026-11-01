# Endophalon 3 integration closeout

The owner authorized integration of the existing Endophalon 3 work, superseding the experiment merge prohibition for this scope only. Earlier integration records are historical. The fixed constraints and preservation rules remain in force.

| Plan item | Status and evidence |
| --- | --- |
| Base, probes, full training A binary seeds 1/2, no-region seeds 1/2, A one-hot seed 1, B binary/one-hot seeds 1/2, equal-budget random baseline | Completed and previously audited; nine saved cards and raw evidence in `docs/endophalon3_runs/20261031-training/`. |
| A one-hot seed 2 | Optional in original handoff; not executed. |
| Training results and routing A seed 1/2, original and confirmation evaluations | Completed; see the training and routing reports. |
| Routing B follow-up | Stopped by the predeclared gate; both A seeds failed. No search or extra training is authorized by this closeout. |
| Integration and documentation | This closeout. Accuracy goal remains unmet; completion of the plan is not a performance claim. |

Original A binary test was 78.85% / 88.33%; A one-hot seed 1 was 87.71%. All three failed the >=90% overall and >=85% each-operation gate. Recall2 region-minus-homogeneous was +5.13 and -6.84 percentage points, so consistent improvement was not shown. B binary/one-hot test means were both 74.06%. The routing correction reduced A test accuracy to 75.31% / 81.77%; it is retained as opt-in evidence with the original objective as default.

Training `24ebc6b`, routing `2506a9d` and workspace `ff3df87` are preserved in ancestry. The four workspace-only interrupted run files are byte-identical to `cloud-attempts-recovered/` in the training tree. The workspace summarizer is older than the recovery-aware version, so its older content was not restored over the later one. Interrupted final theta/readout and complete logs were not recovered; the partial logs remain. Search seeds share one physical/data seed; finite results cover the measured 64-event streams only.

Master's earlier 166 reproduction files and iPad Lab remain intact. Source model cards, logs and configurations from the latest routing tree are preserved byte-for-byte. PR #10 and #11 were draft records with no issue comments, inline comments or reviews at inspection; the integration preserves their actual results and history.
