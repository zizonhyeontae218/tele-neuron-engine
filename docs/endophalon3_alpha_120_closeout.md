# Endophalon 3 physics120 — closeout recovery

The 120-generation training budget completed (2,881 candidate pairs), with zero failed candidates. The complete saved-model audit **passed in the original Actions job**, including train-only head refits, objective, all held-out metrics and exact six-stream physical replay. Original job 41435945048 is still recorded as **failure** because the subsequent manifest-writing step mixed a relative output path with an absolute repository root. Auto-publication was skipped. This is a closeout error, not incomplete training or a failed accuracy reproduction.

The closeout path is now normalized before repository-relative hashing. A regression test runs the actual closeout with workflow-style relative paths, verifies output hashes, and confirms that a completed model cannot be overwritten. No training code, settings, original starting bundle, selected theta or trained head was changed. No retraining, seed change or additional candidate search occurred.

## Recovery and preservation

All 132 original artifact files were recovered, including the originally generated report and standalone model; they remain byte-identical. Artifact ZIP: 1,945,980 bytes; SHA-256 `eb56d6165d46c319edfb67e196b38bf35aa3ceb24800ce716bc81eceeb6ffc8d`. Original artifact ID 13142542905. Per-file original hashes: `docs/endophalon3_alpha_runs/20261102/physics120/recovery_original_files.json`. The full original Actions job log, including the successful audit and later traceback, is preserved as `actions_job.log` in the same directory. The original report is [retained unchanged](endophalon3_alpha_120_report.md).

Recovered budget, 121 archived checkpoints, start/source hashes and serialized held-out metrics were independently checked. The original successful expensive physical/head-refit audit was not repeated. Original 114 tests passed before training; the new closeout regression passed separately. CLI smoke (skipped by the failed job) and preservation/index checks were completed during recovery. Prior model/config/source/experiment files remain unchanged.

## Measured outcome

On the same fresh confirmation streams 40000–40063, overall accuracy rose from **65.44% to 75.36% (+9.92 pp)**. A rose from **58.90% to 77.77% (+18.87 pp)**; B rose from **71.69% to 73.07% (+1.37 pp)**. Both level and per-operation acceptance gates remain unmet. Validation-selected objective fell from 0.3286299508 to 0.1820288241; validation accuracy was 80.00%, with A 84.06% / B 75.55%. The run took about 74.6 minutes of measured process time; this is one seed/pair and one physical/data seed, not a cross-machine efficiency comparison.

| Level | Confirmation operation | Accuracy | Events |
| --- | --- | --- | --- |
| A | xor | 91.16% | 475 |
| A | recall1 | 76.69% | 429 |
| A | recall2 | 60.24% | 508 |
| A | xor_mem | 84.27% | 464 |
| B | xor | 69.17% | 253 |
| B | recall1 | 68.67% | 233 |
| B | recall2 | 50.65% | 231 |
| B | xor_mem | 73.98% | 246 |
| B | and | 93.96% | 265 |
| B | or | 92.75% | 262 |
| B | parity3w | 51.34% | 261 |
| B | and_mem | 82.16% | 213 |

The main remaining weaknesses are A recall2, B recall2 and B parity3w. A xor and B and/or exceed 90%, but other operations prevent acceptance. Reset-at-switch control was 70.18% overall versus 75.36% with continuous state; this is the same-head state-ablation comparison, not evidence that all memory tasks meet the target. Only 64-event streams were measured. Fresh confirmation has now been observed and cannot be reused as untouched validation for a follow-up.

The original GitHub workflow is not rerun: a clean rerun could repeat the completed training budget. The recovered artifacts and focused code fix are integrated through ordinary commits, preserving the failure record. Master and endophalon3 are verified after normal fast-forward publication; unrelated branches are not changed. No new remote branch is created. A separate verification record and complete hash manifest record this recovery.
