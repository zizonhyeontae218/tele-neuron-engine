# Teleneuron-Endophalon-2-Pro

The fifth, separate version. It keeps Endophalon-2's protocol: the physics
keeps running, each input pushes the physics when it arrives, and the readout
reads a fixed number of steps later. Two things change: **the physics itself**
and **input timing**. All earlier versions are unchanged.

| | Endophalon-2 | **Endophalon-2-Pro** |
|---|---|---|
| Physics | 144 colliding balls | **144 nodes joined by nonlinear springs** (506 springs, no collisions) |
| Restoring force | added anchor spring | the springs themselves (plus a weak anchor so the network stays in place) |
| Input | 1/r² push from a point | Gaussian push near a learned point (learned width) |
| Input timing | every 16 steps | every 16 steps **or irregular: each gap drawn from 12–24 steps** |
| `R_drift` | pooled across streams (flawed) | **per stream**: median \|energy trend\| across streams |

## Physics

- **Network:** nodes start on a jittered 6×6×4 lattice in a 5×5×3 box. Each
  node is joined to its 6 nearest neighbors (degree 6–12, 506 springs).
- **Spring force:** `k_linear·x + k_cubic·x³`, where x is the extension along
  the spring.
- **Integration:** two substeps per step.
- **Blow-ups:** a θ whose integration diverges is flagged and scored
  `J = 10`. No run produced one.
- **θ (27 dimensions with 4 zones):** damping, `k_linear`, `k_cubic`, mass
  spread, anchor, heat bath, pulse strength, stimulus width, readout width,
  2 stimulus points, 4 readout centers.
- **Readout:** 4 (or 8) soft readout zones, read 12 steps after each input.
- **Training:** same bilevel loop, splits and tasks as Endophalon-2.
  - Exact ridge readout per task.
  - 16 train / 8 validation / 16 test streams of 64 inputs each.
  - 30 generations × 16 candidates.
  - About 13 minutes per run on 4 cores.

## Results (held-out test streams)

Each cell shows test accuracy over the whole stream, and in parentheses the
first half → second half of each stream.

### Trained on xor + recall1 + xor_mem, 4 readout zones

| Task | Balls (E2) | **Springs, regular, seed 1** | **Springs, regular, seed 2** | Springs, irregular, seed 1 | Springs, irregular, seed 2 | Random search (regular) | Untrained springs |
|---|---|---|---|---|---|---|---|
| xor | 99% | 98% (98→98) | **100%** (99→100) | 90% | 95% | 57% | 50% |
| recall1 | 86% | 95% (96→94) | **100%** (100→100) | 86% | 94% | 72% | 72% |
| xor_mem | 84% | 96% (97→95) | **100%** (100→100) | 83% | 84% | 67% | 50% |

- With every input reset (no history), xor is 100% and the memory tasks are
  at chance (≈50%).
- The physics-free readout on the raw bits gets xor 48% and xor_mem 56%.

### Trained on xor + recall2 + parity3w (deeper memory)

| Task | 4 zones | **8 zones** |
|---|---|---|
| xor | 82% | **99%** (99→99) |
| recall2 | 71% | **97%** (96→97) |
| parity3w (3-input parity) | 57% | **80%** (77→82) |
| recall1 (not trained) | 72% | 95% |

### Ablation: physics or readout zones? (`docs/endophalon2_pro_ablation.json`)

Soft zones are Gaussian, i.e. nonlinear in node positions. So the trained θ
was rescored with a **purely linear readout of raw node displacements** (432
numbers, linear in the physical state), and again with the cubic spring term
set to 0. Continuous mode, test streams:

| Model | Observation | xor | recall2 | parity3w |
|---|---|---|---|---|
| regular, seed 1 | 4 soft zones | 98% | 62% | 53% |
| | **linear node displacements** | **100%** | **100%** | **100%** |
| | linear, no cubic term | 99% | 100% | 99% |
| regular, seed 2 | linear node displacements | 100% | 100% | 98% |
| irregular, seed 1 | 4 soft zones | 90% | 58% | 57% |
| | linear node displacements | 93% | 87% | 70% |

## What this shows

- **Changing the physics paid off.** With the same protocol, readout and
  budget, the spring network beats the balls on every trained task: xor_mem
  96–100% vs 84%, recall1 95–100% vs 86%. Both regular-interval seeds
  converge to similar results, and the second seed reaches 100/100/100.
- **The nonlinearity is in the physics.** A linear readout of the network
  state does XOR at 100%, and no linear map of the input bits can do that. It
  survives removing the cubic term, so it comes from the network's geometry
  (springs pull along their changing directions) and from the
  position-dependent input push. Training lowered `k_cubic` in five of six
  runs (0.5 → 0.006–0.22; one irregular run kept 0.42).
- **The state holds at least 3 inputs of history.** A linear readout of node
  displacements reaches recall2 100% and parity3w 98–100%. So, after
  training, the physics is no longer the bottleneck; the small readout is.
  With 4 zones, deep-memory tasks stay at 57–71%. With 8 zones, they reach
  recall2 97% and parity3w 80%.
- **Learned physics varies between seeds.** Damping ended at 0.84–0.92 in
  every run. It was lightest (0.92) in the deep-memory, 8-zone run, which fits
  needing a longer echo, but one run cannot establish a trend. Pulse strength
  differs across seeds of the *same* setup (0.28 vs 0.05), so several quite
  different physics systems solve the same tasks.
- **Irregular timing works, but costs accuracy.** xor 90–95%, xor_mem 83–84%.
  Even a linear readout of the full state gets only 93% XOR. When the gap to
  the previous input varies, the leftover state at readout varies too, so this
  limit is in the physics, not only the readout.
- **Continuous operation holds.** In every run, accuracy in the second half of
  each stream equals the first half within ±2 points (parity3w with 8 zones:
  77 → 82).

## Caveats

- **Seeds.** Two seeds for the main regular and irregular runs, one for the
  deep-memory runs.
- **`R_drift` still reads higher on test streams than during training**
  (e.g. 0.18 → 0.28, 0.22 → 0.42). The gap is much smaller than in
  Endophalon-2 (0.01 → 0.5). The values are small: 0.4 means the median
  stream's energy changes by about ×1.5 across 60 inputs. Accuracy does not
  degrade over the stream.
- **The 432-feature linear readout is a diagnostic, not a model.** It shows
  what the physics state holds. The deployed readout is still 4 or 8 zones.
- **Not a matched comparison with the balls.** Same box, node count, protocol
  and budget, but the input kernel differs (Gaussian vs 1/r²).

## Model cards (`models/endophalon2_pro/`)

| Card | Trained on | Timing | Zones |
|---|---|---|---|
| `teleneuron_endophalon2_pro_regular_multi.json` (+ `_seed2`) | xor, recall1, xor_mem | regular | 4 |
| `teleneuron_endophalon2_pro_irregular_multi.json` (+ `_seed2`) | xor, recall1, xor_mem | irregular 12–24 | 4 |
| `teleneuron_endophalon2_pro_regular_deep.json` | xor, recall2, parity3w | regular | 4 |
| `teleneuron_endophalon2_pro_regular_deep_k8.json` | xor, recall2, parity3w | regular | 8 |

Run logs: `docs/endophalon2_pro_runs/`.

## Run

```bash
python -m tele_neuron.endophalon2_pro --config configs/endophalon2_pro/regular.json \
  --tasks xor recall1 xor_mem --random-baseline --output models/endophalon2_pro/regular.json
python -m tele_neuron.endophalon2_pro --config configs/endophalon2_pro/irregular.json --tasks xor recall1 xor_mem
python -m tele_neuron.endophalon2_pro --config configs/endophalon2_pro/regular_zones8.json --tasks xor recall2 parity3w
python -m tele_neuron.endophalon2_pro.ablation --config configs/endophalon2_pro/regular.json \
  --model models/endophalon2_pro/teleneuron_endophalon2_pro_regular_multi.json
```

## Next

1. **Readout zones are the bottleneck now:** sweep 4 / 8 / 16 zones on one
   trained physics, and train for all six tasks at once.
2. **Irregular timing:** let the readout see elapsed time, or give the physics
   a fast "reset-free settling" mode (e.g. damping that depends on activity).
3. **More seeds for the deep-memory runs.**
