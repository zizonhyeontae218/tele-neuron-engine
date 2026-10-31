# Teleneuron-Endophalon-3 (base, untrained)

**Status:** the base is built and tested. **Nothing has been trained.** Start
with [endophalon3_handoff.md](endophalon3_handoff.md).

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

| Config | Single zone readout | Single readout on all node displacements | Separate per-op readout on nodes (≈ what the state holds) |
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
