# Endophalon 3 — 120-generation physics training result

The owner-authorized new-3 paired-bank run completed **120 generations / 2,881 candidate pairs**. A/B parameters and heads remain separate; one production Session carries state through commands. This is not the historical 3T single-head experiment. The selection used training/validation only. [Frozen plan](endophalon3_alpha_120_plan.md).

| Fresh streams 40000–40063 | Overall | A | B |
| --- | --- | --- | --- |
| Frozen baseline | 65.44% | 58.90% | 71.69% |
| Trained paired banks | 75.36% | 77.77% | 73.07% |
| Trained heads, reset-at-switch | 70.18% | 71.00% | 69.40% |

Fresh confirmation gate (>=90% each level and >=85% every operation): **FAIL**. A completed budget is distinct from a performance success. All per-operation and first/second-after-switch results, regularizers, failed candidates, switching energy and timing are in `docs/endophalon3_alpha_runs/20261102/physics120/evaluation.json`.

Both 77-dimensional bank theta vectors were optimized together, while each candidate's two heads were separately fitted on only the corresponding training events. Search seed 1, physical/data seed 1; one binary bank pair, no seed/coefficient search. The balanced-level MSE and existing noise/drift/weight penalties were frozen before training. Drift covers train+validation; switching energy is an external-control change, not a conservation residual. Fresh confirmation was first used after theta selection. The old test was previously observed and is descriptive. Confirmation is now observed and must not be reused as untouched evidence in adaptive follow-ups.

Reset controls change positions/velocities at bank switches with no extra warmup, preserving time/RNG and the same heads. They are state ablations, not separately trained reset models. Finite observations cover 64-event streams only; no infinite-horizon or multi-data-seed generalization claim. This one run cannot establish that a different seed/method/budget could never improve the result.

Artifacts include 121 archived optimizer states, full latest mean/sigma/RNG/best state and candidate metrics, raw log, environment, settings/source/start hashes, selected two-head bundle and held-out observations. The original model/configs/MVP/3T evidence is preserved. The standalone trained bundle is `models/endophalon3_alpha/20261102_physics120_seed1_binary.json`.

Read-only audit: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/audit_endophalon3_alpha_training.py docs/endophalon3_alpha_runs/20261102/physics120`. It verifies the budget, checkpoints, source/start fingerprints, train-only head refits, objective and all held-out metrics, plus exact physical replay on six held-out streams. Tests and model smoke are separate execution checks, not accuracy claims.
