# Teleneuron-Endophalon-2-ProMax8 / ProMax16

The sixth, separate version. There is **one physics system and one readout**,
and the operation is chosen by input. All earlier versions are unchanged.

Endophalon-2-Pro gave each task its own readout. In ProMax, every input event
carries two data bits and an **opcode**:

| Opcode | Output for event k |
|---|---|
| `xor` | a_k ⊕ b_k |
| `recall1` | a_{k−1} |
| `xor_mem` | a_k ⊕ a_{k−1} |

The opcode is drawn at random for every event, so it can change from one event
to the next. It enters the physics as a push at its own learned stimulus
point. A **single** linear readout, fixed for all operations, reads the soft
readout zones 12 steps later. To give the right answer for whichever
operation was requested, the physics has to route the computation.

| | ProMax8 | ProMax16 |
|---|---|---|
| Readout zones | 8 | 16 |
| θ dimensions | 48 | 72 |
| Physics | Endophalon-2-Pro spring network (144 nodes, 506 springs) | same |
| Input channels | a, b, opcode×3 (5 stimulus points) | same |
| Training | warm start from the best Endophalon-2-Pro θ | same |

- **Training settings:** 40 generations × 24 candidates (961 θ). Regular
  timing, every 16 steps. Splits: 16 train / 8 validation / 16 test streams
  of 64 events. About 20 minutes per run on 4 cores.
- **Objective:** `J = L_val + 0.05·R_noise + 0.1·R_drift + 0.01·R_complexity`,
  computed on the single readout.

## Results (held-out test streams, single readout)

| Model | xor | recall1 | xor_mem | **All ops** | First half → second half |
|---|---|---|---|---|---|
| **ProMax16, seed 1** | 98% | 97% | 96% | **96.9%** | 97 → 96 |
| ProMax16, seed 2 | 94% | 89% | 93% | 91.9% | 92 → 91 |
| **ProMax8, seed 2** | 99% | 93% | 93% | **95.0%** | 95 → 95 |
| ProMax8, seed 1 | 99% | 80% | 95% | 91.0% | 91 → 91 |

| Control (test streams) | 8 zones | 16 zones |
|---|---|---|
| Untrained physics (Endophalon-2-Pro θ), single readout | 67.0% | 71.0% |
| Random search, same 961-θ budget | 77.9% | 80.9% |
| Physics-free: bits + opcode, linear | 66.7% | 66.7% |
| Trained, single readout **also given the opcode** | 91.9% / 95.4% | 96.9% / 92.6% |
| Trained, **separate readout per operation** (upper reference) | 98.6% / 97.3% | 99.7% / 97.8% |
| Trained, linear readout on all node displacements (diagnostic) | 99.5% / 99.6% | 99.5% / 96.6% |

Where two numbers are given, they are seed 1 / seed 2. The physics-free
baseline solves recall1 (100%), but xor and xor_mem stay at chance (51% /
47%).

Model cards: `models/endophalon2_promax/teleneuron_endophalon2_promax{8,16}.json`
and `…_seed2.json`. Run logs: `docs/endophalon2_promax_runs/`.

## What this shows

- **One physics system with one readout runs three operations, picked
  by input.** After training, the single readout is right on 91–97% of
  events across all three operations, compared with 67–71% untrained. That is
  close to the separate-readouts reference (97–100%): input-driven routing
  costs about 2–8 points compared with giving every operation its own readout.
- **The routing is in the physics, not the readout.** Giving the opcode to the
  readout as well changes almost nothing (at most +0.9 points). A linear readout
  cannot switch functions with an extra input, so whatever selects between
  XOR and recall happens inside the physics, driven by the opcode push.
- **Training the physics is what does it.** Random search with the same budget
  reaches only 78–81%.
- **It keeps working without reset.** In every run, accuracy is the same in
  the first and second halves of each stream (within 1 point).
- **The per-stream drift fix holds.** `R_drift` on training streams now
  matches test streams (0.14 / 0.13, 0.14 / 0.14, 0.17 / 0.17, 0.12 / 0.10).
  The pooled version in Endophalon-2 gave 0.01 vs 0.5.
- **8 vs 16 zones: not settled.** Averages are 93.0% (ProMax8) and 94.4%
  (ProMax16), but the difference between seeds (about 5 points) is larger
  than the difference between the two sizes. The best single model is
  ProMax16 seed 1 (96.9%). The second best is ProMax8 seed 2 (95.0%).
- **recall1 is the hardest operation to route** (80–97%). It needs the
  readout to ignore the current input and report the previous one, which is
  the opposite of what xor needs.

## Caveats

- **Two seeds per model.** The ranking of 8 vs 16 zones can change with more
  seeds.
- **Regular timing only.** Irregular timing was harder in Endophalon-2-Pro and
  is not tested here.
- **Warm start.** Both models start from the best Endophalon-2-Pro θ, so the
  results include what that run already learned.
- **The node-displacement readout is a diagnostic.** It shows what the state
  holds, but it reads 432 numbers; the model reads 8 or 16 zones.

## Run

```bash
python -m tele_neuron.endophalon2_promax --config configs/endophalon2_promax/promax16.json \
  --random-baseline --output models/endophalon2_promax/promax16.json
python -m tele_neuron.endophalon2_promax --config configs/endophalon2_promax/promax8.json --seed 2
```

## Next

1. **More seeds,** to settle ProMax8 vs ProMax16.
2. **Irregular timing with opcodes.**
3. **More operations** (e.g. recall2, parity3w) on the same single readout,
   to see how many functions one physics system can host.
