# Endophalon-3 handoff: how to train it, and what we learned

For the next worker. The base is built and tested, but **no training has been
run**. Read [endophalon3.md](endophalon3.md) first for the design. This page
covers the decisions already made, the run order, the success criteria, and
the know-how from six earlier versions.

## Decisions already made (by the project owner)

- **Name:** `Teleneuron-Endophalon-3`. Level A and level B are **trained
  separately** and produce `Endophalon-3-A` / `Endophalon-3-B`.
- **Input:** `level → operation → data`, all carried by the input (ISA in
  `isa.py`). Level A has 4 ops, level B has 8 ops.
- **Physics:** keep **2 substeps** and **144 nodes**. Do not switch to 1
  substep or a smaller network, even though both looked cheaper in probes.
- **Accepted additions:**
  - the slow region (two timescales)
  - readout-only recording
  - binary vs one-hot opcode comparison
  - the ProMax control set
- **Success criteria (fixed):**
  1. Level A, single readout: **≥ 90% overall and ≥ 85% on every op** (held-out test streams).
  2. The slow region measurably raises recall2 over the homogeneous `--no-regions` ablation.
  3. Report both seeds, and report failures as plainly as successes.

  A third criterion, "≥ 4× less physics compute", was proposed with 1
  substep and 64 nodes. Those were overruled, so only the ~25%
  readout-only-recording saving applies. Report the compute honestly.

## Run order

Each training run is about 40 generations × 24 candidates = 961 θ.

| Configuration | Time on 4 cores |
|---|---|
| Level A (16 train streams) | ~17 min (23 s per generation, measured) |
| Level B (24 train streams) | ~18 min (25 s per generation, measured) |
| `--random-baseline` | doubles the run time |

Background jobs are cut at 2 h, so chain at most ~3 runs per job.

1. **Re-check the base.** Run `python -m pytest -q`, `python scripts/smoke_models.py`, and
   `--probe` on the configs you will train. The probe numbers in
   `docs/endophalon3_probes/` are the gen-0 reference.
2. **Level A, binary.** `level_a.json`, seeds 1 and 2, with `--random-baseline`
   on seed 1. Save cards under `models/endophalon3/` with **new names**.
3. **Level A, slow-region ablation.** Same as step 2 with `--no-regions`, seeds 1
   and 2. This is criterion 2: compare per-op recall2.
4. **Level A, one-hot.** `level_a_onehot.json`, seed 1 (2 if time allows).
5. **Level B, binary and one-hot.** Expect lower accuracy. parity3w was 72–79%
   even for a per-op readout of the full state, so it is the ceiling to watch.
6. **Write-up.** `docs/endophalon3.md` results section, README sections, run logs in
   `docs/endophalon3_runs/`, and a PR. Do not overwrite earlier cards or logs.

## Training method (what the code does, and why)

### The bilevel loop

This is the same loop as Endophalon-Pro, 2, 2-Pro and ProMax.

- **Outer loop:** a cross-entropy search over θ in an unconstrained space.
  Sigmoids map every vector to a valid physics, so there are no constraint
  checks. Each generation:
  - samples `population − 1` candidates around the current mean, plus the mean itself
  - scores them, then moves the mean to the log-weighted top `elite`
  - sets σ ← 0.7·spread + 0.3·σ, floored at `min_sigma`
- **Inner loop:** for each θ, the readout is solved exactly with ridge
  regression on the train streams. No gradients flow through the physics.
- **Fitness:** `J = L_val + 0.05·R_noise + 0.1·R_drift + 0.01·R_complexity`,
  computed on validation streams.
  - `L_val`: MSE of the single readout.
  - `R_noise`: within-class score spread ÷ between-class gap. It is scale-free
    on purpose.
  - `R_drift`: the **per-stream** median |energy trend|.
  - `R_complexity`: mean squared readout weight.
- **Splits:** by **whole stream** (train / val / test), never by event. The
  first `washout` events are dropped, and must be ≥ 2 because recall2 needs two
  events of history. Test streams are never seen by either loop.
- **Warm start:** `theta_init` is ProMax16 seed 1 (96.9% on 3 ops), plus new
  level/opcode stimulus points. The slow region starts equal to the fast one.
  `init_sigma` is 0.5 rather than 0.8 because the start is already good.

### Controls the code reports (and why each exists)

| Control | Question it answers |
|---|---|
| `single_readout_plus_instruction` | Does handing the readout the instruction help? If not, the routing is in the physics (ProMax: at most +0.9 points). |
| `separate_readouts` | The upper reference: one readout per op. The gap to it is the cost of routing. |
| `linear_nodes` | What the physics state holds, with a 432-feature readout. A diagnostic, not a model. |
| `history_baseline` | Physics-free: a linear readout on the instruction plus a_k, b_k, a_{k−1}, a_{k−2}. It cannot do XOR-type ops. |
| initial θ | Gen 0, untrained. |
| `--random-baseline` | Same budget, no learning. If training does not beat it, the search is not working. |

## Know-how from earlier versions (the expensive lessons)

1. **Check where the nonlinearity lives.** Gaussian readout zones are
   themselves nonlinear in positions. Endophalon-2-Pro proved the physics does
   the XOR by showing a **linear** readout of node displacements still gets
   100%. Keep that check (`linear_nodes`).
2. **Watch for metrics the search can game.**
   - Endophalon-2's pooled `R_drift` let heating and cooling streams cancel, so
     the search found θ with 0.01 on train+val but 0.5 on test.
   - The per-stream drift fixed it (ProMax: train 0.14, test 0.13).
   - Pro's first `R_noise` (raw score spread) rewarded a readout that shrank
     every score, and the regularized runs got *worse*.
   - Both times, **compare the train-time and test-time value** of every
     regularizer.
3. **Seeds disagree more than you expect.** ProMax8 vs ProMax16 differed by
   1.4 points on average, while their own seeds differed by about 5. Never
   rank configurations on one seed. Physics θ also varies a lot between seeds,
   e.g. pulse strength 0.28 vs 0.05 for the same task, so do not claim "the
   physics learns X" from one run.
4. **Recall is hard for one readout because it is the opposite of XOR.**
   recall1 must ignore the current pulse, which dominates the state at
   readout time (ProMax8 seed 1: recall1 80% vs xor 99%). The slow region is
   the structural answer to test. If it does not help, see "If stuck" below.
5. **Irregular timing costs real accuracy** (2-Pro: 90–95% vs ~100%), and
   that loss is in the physics too: even a full-state linear readout got 93%.
   Keep this base on regular timing until A/B work.
6. **Physics can blow up.** A θ that diverges is flagged non-finite and scored
   J = 10. The log line `failed=N` counts them per generation. If many fail,
   lower `init_sigma` or narrow the stiffness range in `theta.py`.
7. **Do not trust the first number; verify the claim.** Every earlier
   write-up was checked against the logs before publishing, and several claims
   were corrected (ranges, a "physics re-tunes" story that one seed
   contradicted). Do the same.
8. **Repository rules.** Read `AGENTS.md`:
   - never overwrite published cards or logs
   - no force pushes
   - keep README branch text accurate
   - smoke runs are not benchmarks

## Probe numbers to beat (gen 0, untrained, `docs/endophalon3_probes/`)

| Config | Single zone readout | Weakest ops |
|---|---|---|
| level A binary | 71.5% | recall2 53%, recall1 65% |
| level A onehot | 72.9% | recall2 56%, recall1 59% |
| level B binary | 66.5% | parity3w 40%, xor_mem 58% |
| level B onehot | 65.6% | parity3w 43%, xor_mem 50% |

## If stuck (ideas not yet built)

- **Readout delay per op.** Recall might read better at a different delay. The
  engine already records any set of `delays`. This would change the protocol,
  so ask the owner first.
- **More readout zones or a larger budget for level B** (θ has 77–92
  dimensions; ProMax16 had 72 and needed 40 generations).
- **Learned region split.** The slow region is now a fixed `y > 3.0`.
  `slow_mask` takes any axis and threshold, which could become part of θ.
- **Joint A+B model** (`"level": "AB"`), only after A and B work separately.
