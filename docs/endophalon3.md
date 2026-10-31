# Teleneuron-Endophalon-3 (training experiments)

**Status (2026-10-31):** 9/9 planned models trained and audited; A accuracy gates 0/3.
See the [research report](endophalon3_training_report.ko.md). The original
[handoff](endophalon3_handoff.md) remains the fixed experiment protocol.

Endophalon-3 builds on the Endophalon-2-Pro / ProMax spring architecture. It
is an instruction-driven physical computer: one physics system and **one**
linear readout, and every input event is an instruction that picks a level,
then an operation inside that level, then gives the data.

## Instruction set (`src/tele_neuron/endophalon3/isa.py`)

```python
from tele_neuron.endophalon3.isa import Instruction, program

Instruction.parse("A XOR 1 1")         # level A, op xor, a=1, b=1
Instruction.parse("B:and_mem(0,1)")    # alternate syntax
x, targets = program(["A XOR 1 0", "A recall1 0 0", "A recall2 1 1", "A xor_mem 0 1"])
# x: (4 events, 6 channels) input to the physics; targets: [1, 1, 1, 1]
```

In low-level terms, `"A XOR 1 1"` means:
1. `level("A")`: set the level bit.
2. `run("XOR")`: set the opcode bits.
3. `input(1, 1)`: set the two data bits.

All three are pushed into the running physics as **one** event, and the
readout reads the answer 12 steps later.

| Level | Ops (opcode index) | Output for event k |
|---|---|---|
| A | `xor` (0), `recall1` (1), `recall2` (2), `xor_mem` (3) | a⊕b, a_{k−1}, a_{k−2}, a_k⊕a_{k−1} |
| B | the 4 above + `and` (4), `or` (5), `parity3w` (6), `and_mem` (7) | a∧b, a∨b, a_k⊕a_{k−1}⊕a_{k−2}, a_k∧a_{k−1} |

The opcode index of an op is the same in both levels, so "A XOR" and "B XOR"
differ only in the level bit.

### Encodings

Each channel that is 1 pushes the physics at its own learned stimulus point.

| Encoding | Channels | Stimulus points |
|---|---|---|
| `binary` (default) | `[level, op_bit2, op_bit1, op_bit0, a, b]` | 6 |
| `onehot` | `[level, op_xor … op_and_mem, a, b]` | 11 |

`binary` scales: 3 opcode bits cover 8 ops. The physics must decode bit
combinations. `onehot` is the ProMax style.

### Levels and training

As agreed, **A and B are trained separately**:
- `configs/endophalon3/level_a*.json` produce **Endophalon-3-A**.
- `level_b*.json` produce **Endophalon-3-B**.

The level bit is still part of every input. A later joint model can use
`"level": "AB"`, where the level is drawn per event.

## Physics (`engine.py`, `theta.py`)

- **Base engine:** the Endophalon-2-Pro spring network. 144 nodes, 506 springs,
  **2 substeps**, Gaussian input pushes, anchor, heat bath, soft readout
  zones (16).
- **New: slow region.** Nodes with home `y > 3.0` (56 of 144) get their own
  `damping_slow`, and their springs are scaled by `stiffness_slow` (averaged
  over a spring's two ends). This gives the physics a second timescale for
  history: the recall1/recall2 problem. The configs **start the slow region
  equal to the fast one**, so gen 0 is the homogeneous physics and training
  decides whether to separate them. `--no-regions` removes the slow region.
- **New: readout-only recording.** Soft zones are computed only at the readout
  delay, not at all 16 delays (~25% less simulation time, per ProMax16
  profiling).
- **Equivalence (tested):** with no slow region and every delay recorded, the
  engine output equals `endophalon2_pro.engine.run_events` exactly.

θ has 11 scalars plus 3 × (channels + 16 zones):
- `binary`: 77 dimensions.
- `onehot`: 92 dimensions.

## Stage-1 capacity probes (untrained, `docs/endophalon3_probes/`)

The probes start from `theta_init`: ProMax16 seed-1 physics and readout
zones, plus new level/opcode points. They read held-out test streams with
ridge readouts.

| Config | Single readout (16 zones) | Single readout on all node displacements | Separate per-op readout on nodes (≈ what the state holds) |
|---|---|---|---|
| level A, binary | 71.5% (xor 96, recall1 65, recall2 53, xor_mem 72) | 87.9% | 100% for every op |
| level A, onehot | 72.9% | 89.9% | 100% for every op |
| level B, binary | 66.5% (parity3w 40) | 70.7% | 100% except parity3w 72% |
| level B, onehot | 65.6% | 72.4% | 100% except parity3w 79% |

- **The information is already there.** Per-op readouts reach about 100%,
  except parity3w.
- **The single readout routing it is not.** The problem is getting one shared
  readout to pick the right function per instruction. ProMax closed a similar
  gap (67–71% → 91–97%) by training the physics.

## Commands

```bash
python -m tele_neuron.endophalon3 --config configs/endophalon3/level_a.json --probe            # stage 1, no training
python -m tele_neuron.endophalon3 --config configs/endophalon3/level_a.json --random-baseline \
    --output models/endophalon3/<new-card>.json                                                # stage 2 training
python -m tele_neuron.endophalon3 --config configs/endophalon3/level_a.json --no-regions ...   # homogeneous ablation
```

## Fixed-budget results — 2026-10-31

Original scientific source/configs/models/logs are preserved. All training runs use 144 nodes, 2 substeps, 16 zones and 40 generations ×24 population (961 theta evaluations). A and B train separately; physics and dataset seed remain 1.

| Run | Overall | xor | recall1 | recall2 | xor_mem | and | or | parity3w | and_mem |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_binary_s1 | 78.85% | 73.22% | 81.50% | 73.50% | 87.12% | — | — | — | — |
| a_binary_s2 | 88.33% | 97.07% | 94.49% | 67.09% | 93.99% | — | — | — | — |
| a_no_regions_s1 | 80.73% | 91.63% | 77.56% | 68.38% | 85.41% | — | — | — | — |
| a_no_regions_s2 | 85.42% | 93.31% | 90.55% | 73.93% | 83.26% | — | — | — | — |
| a_onehot_s1 | 87.71% | 92.05% | 93.31% | 77.78% | 87.12% | — | — | — | — |
| b_binary_s1 | 74.48% | 80.87% | 80.17% | 61.19% | 62.18% | 89.92% | 94.59% | 55.00% | 75.21% |
| b_binary_s2 | 73.65% | 83.48% | 83.47% | 59.70% | 60.50% | 87.39% | 89.19% | 50.83% | 77.69% |
| b_onehot_s1 | 74.17% | 63.48% | 88.43% | 58.96% | 57.98% | 97.48% | 99.10% | 52.50% | 78.51% |
| b_onehot_s2 | 73.96% | 72.17% | 85.95% | 56.72% | 57.98% | 80.67% | 96.40% | 57.50% | 87.60% |

A accuracy gates (>=90% overall and >=85% each op): 0/3 non-ablated runs.

Slow Region recall2 effect: seed 1 +5.13 pp, seed 2 -6.84 pp; mean -0.85 pp, conditional paired-stream 95% CI [-3.66, +2.01] pp. Consistent improvement is not demonstrated. The interval conditions on these two search seeds.

A binary seed-1 random search: 73.75%, versus trained 78.85% and initial 71.46%, using the same 961-theta budget.

85 pytest tests and 9 CLI/config smoke combinations passed; four untrained probes and four original ProMax native-task cards replay exactly. Readout controls, train/validation/test splits, every regularizer, numerical ranges and timing are in the research report.

Actual objective scope: validation MSE/noise, train-fitted weight complexity, combined train+validation kinetic-energy drift. Finite measurements cover 64-event streams.

The interrupted cloud attempts and the full-budget Actions recovery runs are recorded separately. Cross-runner readout statistics are checked at rtol=atol=1e-12, with actual numerical errors recorded; held-out aggregate and per-operation accuracy must still replay exactly. The original bitwise audit failure is preserved.

New cards: models/endophalon3/20261031_*.json. Logs, argv, predictions, audits and actual SHA-256 manifests: docs/endophalon3_runs/20261031-training/. The recovered executor later synchronized the generated FileTree index/ledger and passed lint at baseline closeout commit `24ebc6bb4133634380d7122070465e8445de014e`; original model and log bytes remain preserved.


## Opt-in routing-loss follow-up — 2026-10-31

A separate branch tested one predeclared change: `0.5 × pooled validation MSE + 0.5 × worst-operation validation MSE`. The existing default objective, physics, 144 nodes, 2 substeps, 16 zones, binary input and 40×24 budget remain unchanged. Both outer search seeds 1/2 completed, with 961 evaluations each.

| Seed | Frozen original test | Modified test | Frozen original on new confirmation | Modified on new confirmation |
| --- | --- | --- | --- | --- |
| 1 | 78.85% | 75.31% | 77.29% | 73.33% |
| 2 | 88.33% | 81.77% | 86.56% | 78.54% |

The modification reduced worst-operation validation MSE but lowered overall held-out accuracy. It is not accepted as an improvement. Neither seed passed the overall/per-operation gates on the original test and predeclared unused confirmation streams 10000–10015, so this follow-up did not expand to B. The original B study above remains complete.

Validation-only diagnostic readouts suggest usable information remains in the states, with losses associated with projection/shared operation selection. These higher-capacity controls do not establish a single causal mechanism. Full per-operation results, limitations, regularizers, cost and raw evidence: [routing report](endophalon3_routing_report.ko.md), [run artifacts](endophalon3_routing_runs/). Two new cards are isolated in `models/endophalon3_routing/`. All 89 pytest tests passed; default-objective replay differences on the two frozen A cards were exactly zero.
