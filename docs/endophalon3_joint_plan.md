# Endophalon 3 joint A/B training — 2026-11-01

Owner follow-up authorizes further training toward one model that accepts A and B in the same stream. This supersedes the earlier no-extra-training/joint-model restriction for this experiment. Prior experiments remain immutable; their negative routing B gate is not relabeled as passed.

Three predeclared trials: AB binary search seeds 1/2 and AB one-hot search seed 1. Each warm starts its corresponding original B card theta (not optimizer continuation, because old CEM distribution/RNG were not saved). No favorable seed replacement. 144 nodes, 2 substeps, 16 zones, one train-fitted linear readout, original physical/data seed and objective remain. Level varies per event; history crosses level switches. No extra head or explicit instruction input to the readout.

Each run: maximum 120 new generations ×24 candidates plus the initial theta; stop on 40 consecutive generations without best-J improvement, or when a candidate passes validation gates. All selections use training/validation only. Original AB train 0–23, validation 24–31; original-test 32–47 is scored once after final selection. Unused joint AB confirmation IDs 20000–20063 are frozen here before training and scored once after selection. Gates: each level overall >=90%, and every operation within that level >=85%. This extends the original A accuracy threshold to B as an explicit experiment target, not a prior achievement. Slow-region causal improvement is not established by this experiment.

CEM mean, sigma, RNG state, best theta, metrics and full history are saved atomically each generation. A resume reuses the same distribution/RNG; final card reuse is rejected. Results, failures and plateau exits remain evidence even if all goals fail. No repeated test-driven restart or unlimited training is performed. Follow-up design depends on measured results.

Command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/train_endophalon3_joint.py --run ab_binary_s1 --workers 4` (same for the other two names). Raw logs/checkpoints/final cards are exported by the joint training Actions workflow. No old model/config/output is overwritten.
