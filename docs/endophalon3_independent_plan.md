# Endophalon 3 — independent A/B 120 generations and tree cleanup

Owner instruction: train the standalone A and B models for 120 generations each, remove the other 3-version implementations/models/artifacts from the active tree, and retain their experimental results only in `docs/endophalon3_development.md`. This explicitly supersedes prior 3T/alpha artifact preservation instructions for this cleanup. v1/v2, Claude, generation archives and iPad Lab are outside scope. Master/endophalon3 remain synchronized; no new work branch, force push, tags or history rewrite.

A starts from the previously selected A-binary seed-2 theta; B starts from B-binary seed-1 theta. Those initial theta values remain only as necessary initialization in the current A/B training configs. Optimizer states restart, using fixed A search seed 2 and B search seed 1; physical/data seed remains 1. Each independent 77-dimensional search is 120×24+1=2,881 candidates; total 5,762 candidates. Elite 6, initial sigma .5, floor .05. No early stop, seed replacement, extra coefficient search or budget extension. Each candidate fits one train-only Ridge readout and scores the original MSE/noise/drift/complexity objective. The models have separate sessions/histories/clocks/RNG and remain connected only through explicit predicted-output registers.

Fresh fixed per-level splits, 64 events / washout 4:

| Model | Train | Validation | Test | Confirmation |
| --- | --- | --- | --- | --- |
| A | 60000–60015 | 60016–60023 | 60024–60055 | 60056–60119 |
| B | 61000–61023 | 61024–61031 | 61032–61063 | 61064–61127 |

Held-out streams are first used after selecting theta on validation. Baseline is gen-0 theta with a train-only fitted head on these new training streams; published earlier 88.33/74.48% figures are historical scores on different streams. Keep 144 nodes, 2 substeps, gap16, delay12, 16 zones per active module. Production sessions are the simulator; no task oracle is supplied to inference. Record all failures and accuracy gates separately from budget completion.

Current package `endophalon3_software` is made self-contained before launch, reusing the necessary spring/ISA/config/theta primitives without importing retired 3 packages. A.json/B.json/current.json are the only current model assets after integration; current configs, training/evaluation code and this run's reproducibility evidence remain. A complete current trained head/physics bundle is not an archive of a discarded version.

Cleanup occurs only after BOTH runs finish and pass saved-model audits. A fixed SHA-256 allowlist of old files guards against concurrent new work. Consolidated development history records prior successes, failures, stopped gates, interrupted-run/recovery limits and source commits, without copying old models into backup folders. Remove old 3T/shared-state packages, old model/config/result directories, obsolete scripts/tests/workflows. Necessary shared primitives now live in the current package and are not removed. Git naturally retains old commits through normal history; this operation removes the active tree rather than rewriting the repository.

Workflow matrix trains A/B independently and uploads artifacts even on failure. The completion job audits both, writes current models, applies the checked cleanup, updates README/agent/goal/command docs and index, runs current tests/smoke/preservation checks, then atomically fast-forwards master/endophalon3 only if both still match the source commit. Concurrent work causes refusal; a follow-up agent must normally integrate while preserving that work. Failed/partial runs may resume only identical settings/source/RNG checkpoints within the original 120-generation budget. Completed runs must not be restarted.

User target reporting: A >=90%, B >=80% confirmation overall. Also report the inherited strict gate (overall >=90% and every operation >=85%) separately. Neither gate authorizes extra generations or replacement seeds.
