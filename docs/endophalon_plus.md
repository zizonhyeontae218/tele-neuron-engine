# Plus physics base and Teleneuron-Endophalon-Plus

Two pieces:

1. **Plus base** (`src/tele_neuron/plus/`) — a larger, batched ball-physics
   engine built for running many simulations in parallel.
2. **Teleneuron-Endophalon-Plus** (`src/tele_neuron/endophalon_plus/`) — the
   Endophalon idea (fixed physics, linear readout) moved onto the Plus base,
   with many more observation regions and **half-learning**: three fixed
   physics presets and a learned gate that picks one preset per input.

## Plus base

The base splits a simulation into three ideas:

| Concept | What it holds | Changes between… |
|---|---|---|
| `WorldSpec` | space, ball count and radius, input points, steps, dt, seed, noise | never within an experiment |
| `Preset` | damping, restitution, input strength, initial speed, mass distribution | presets: "different physical laws" on the same world |
| `Batch` | B simulations as `(B, N, 3)` arrays, each row with its own preset, bits, realization | rows |

`NumpyBackend.step` advances every row at once. Forces, integration and wall
reflection are array math. Collisions use one spatial hash whose key includes
the batch row, so rows can never touch each other. Candidate pairs come from a
bincount/cumsum cell table over the own cell plus 13 neighbor cells.

`run_features` splits jobs into chunks and runs chunks in worker processes.
Each row depends only on its own job, so results are identical for any chunk
size or worker count (tested).

**Deliberate physics difference.** Phase 1 resolves colliding pairs one after
another; the Plus backend computes every contact from the same pre-collision
state and sums the impulses (Jacobi style). With collisions turned off, the
Plus engine reproduces Endophalon's `BallReservoir` to ~1e-13 (tested).

**Throughput** (4-core container, 768 balls, 32 steps):

| Engine | ball-steps / s |
|---|---|
| Endophalon `BallReservoir` (Python collision loop) | ~56k |
| Plus, 1 process | ~530k (≈9.5×) |
| Plus, 4 processes | ~2M (≈37×) |

## Endophalon-Plus

### World and observation

`8 × 8 × 6` space, 768 balls, 32 steps, taps `[8, 16, 24, 32]`. Each tap is
observed with two voxel grids, `[4, 4, 2]` and `[8, 8, 4]`, plus mean speed:
289 values per tap and **1156 features per preset**. Endophalon used 48.

### Presets

| Preset | damping | restitution | input strength | masses |
|---|---|---|---|---|
| `elastic` | 0.995 | 0.95 | 0.3 | all 1.0 |
| `viscous` | 0.9 | 0.5 | 0.6 | all 1.0 |
| `heavy_mix` | 0.98 | 0.8 | 0.6 | 30% at 3.0, 70% at 0.3 |

Presets are hand-designed and **never trained**.

### Half-learning: the gate

`LinearGate` picks a preset with `argmax(W @ [bits, 1])`, with `W` shaped
`3 × (inputs + 1)`.

- **Why linear.** A gate that sees all the input bits could simply encode the
  answer (route every "1" case to one preset), and then the physics would do
  nothing. A linear gate can only cut the input space with flat boundaries, so
  it cannot compute parity by itself.
- **Readout.** Features are placed in a block for the chosen preset, and the
  other blocks are zero. One ridge fit is then the same as one readout per
  preset. With ~3.5k features and fewer samples, ridge is solved in dual form.
- **Search.** Every (preset, realization, case) is simulated **once** and
  cached. The gate search (random restarts plus hill climbing over `W`) only
  re-fits readouts on the cache. Candidates are scored on training
  realizations held out from the fit. Constant gates (one preset for
  everything) are always candidates.
- **Leakage guard.** Candidates are ranked by validation accuracy, then
  per-output accuracy, then *less* leakage, then squared error. Leakage is how
  well the targets can be predicted from the chosen preset alone. Without this
  term, the first matrix run picked a parity3 gate whose choice alone was
  87.5% correct. After the fix, gate-only accuracy is at the trivial floor in
  every cell below.

### Controls reported on every run

- **Bits linear** — the same ridge readout on the raw input bits.
- **Gate only** — a readout that sees only which preset was chosen, which
  measures leakage.
- **Fixed preset ×3** — no gate; every input runs on one preset.

## Results

Test accuracy: every output bit correct, on 16 held-out noise realizations
(128 test samples for 3-input tasks, 256 for 4-input tasks). Both models use 48
training realizations. Raw data: `docs/endophalon_plus_matrix.json`.
Reproduce with `python -m tele_neuron.endophalon_plus.matrix`.

| Task | Jitter | Endophalon | Plus best single preset | Plus routed | Gate only | Bits linear |
|---|---|---|---|---|---|---|
| 3-bit parity | 0.1 | 96.1% | 100.0% (viscous) | **100.0%** | 50.0% | 50.0% |
| 3-bit parity | 0.4 | 85.9% | 100.0% (viscous) | **100.0%** | 50.0% | 50.0% |
| 3-bit parity | 1.0 | 70.3% | 100.0% (viscous) | **100.0%** | 50.0% | 50.0% |
| Full adder | 0.1 | 96.1% | 100.0% (viscous) | **100.0%** | 12.5% | 50.0% |
| Full adder | 0.4 | 84.4% | 100.0% (viscous) | **100.0%** | 12.5% | 50.0% |
| Full adder | 1.0 | 68.0% | 100.0% (viscous) | **100.0%** | 12.5% | 50.0% |
| 4-bit parity | 0.1 | 82.0% | 100.0% (viscous) | **100.0%** | 50.0% | 50.0% |
| 4-bit parity | 0.4 | 63.7% | 99.6% (viscous) | **100.0%** | 50.0% | 50.0% |
| 4-bit parity | 1.0 | 55.1% | 96.5% (viscous) | **100.0%** | 50.0% | 50.0% |
| 2-bit adder | 0.1 | 77.7% | 100.0% (viscous) | **100.0%** | 25.0% | 25.0% |
| 2-bit adder | 0.4 | 64.5% | 99.6% (viscous) | **100.0%** | 25.0% | 25.0% |
| 2-bit adder | 1.0 | 47.7% | 97.3% (viscous) | **99.6%** | 25.0% | 25.0% |
| 2-bit multiplier | 0.1 | 91.8% | 100.0% (viscous) | **100.0%** | 43.8% | 81.2% |
| 2-bit multiplier | 0.4 | 81.2% | 100.0% (viscous) | **100.0%** | 43.8% | 81.2% |
| 2-bit multiplier | 1.0 | 70.7% | 98.8% (viscous) | **99.6%** | 31.2% | 81.2% |

Model cards:

- `models/endophalon_plus/teleneuron_endophalon_plus_full_adder.json`
  (jitter 0.1)
- `models/endophalon_plus/teleneuron_endophalon_plus_parity4_stress.json`
  (jitter 1.0): routed 100%. Single presets: elastic 76.2%, viscous 96.5%,
  heavy_mix 83.2%. Gate only: 50%.

## Reading the results honestly

- **Most of the gain over Endophalon comes from scale, not routing.** A single
  Plus preset already beats Endophalon in every row. Plus has 8× more balls
  and 24× more features. Its box is also 2× wider, so the same absolute jitter
  is *smaller relative noise* than in Endophalon's `4 × 4 × 3` box. The
  Endophalon column is a reference point, not a matched comparison.
- **Routing helps only where a single preset is not already saturated.** That
  means 4-bit parity, the 2-bit adder and the 2-bit multiplier at jitter
  0.4–1.0. The gain is +0.4 to +3.5 points. The clearest case is 4-bit parity
  at jitter 1.0: 96.5% → 100%.
- **Small test sets.** One sample is 0.4% (4-input tasks) or 0.8% (3-input
  tasks). Gaps below ~1 point are within noise.
- **"Best single preset" favors the controls.** It is the max over three
  presets *on the test set*. The routed gate was chosen without seeing the
  test set.
- **The gate still uses several presets when one would do.** At low noise,
  validation saturates at 100% and leakage is already at the floor, so ties go
  to squared error. The test score is unaffected, but these assignments are
  not "necessary" routing.
- **No leakage left in these runs.** Gate-only accuracy equals the trivial
  majority level, or falls below it. This holds for 2-bit multiplier at
  jitter 1.0 (31.2% vs 43.8%), and in every row the routed model beats both
  physics-free controls by a wide margin.

## Run

```bash
python -m tele_neuron.endophalon_plus --config configs/endophalon_plus/full_adder.json
python -m tele_neuron.endophalon_plus --config configs/endophalon_plus/parity4_stress.json --workers 4
python -m tele_neuron.endophalon_plus --config configs/endophalon_plus/adder2.json --jitter 1.0 0.1 \
  --output models/endophalon_plus/adder2_stress.json
python -m tele_neuron.endophalon_plus.matrix   # full table, ~25 minutes on 4 cores
```

## Next steps worth trying

- Learn the presets themselves, for example with an evolution strategy over
  damping, restitution and input strength, scored through the cached-gate
  search.
- A noise level that is matched to box size, so the Endophalon and Plus
  columns become directly comparable.
- Harder tasks (5–6 inputs, multi-output arithmetic) where single presets fall
  well below 100% even at low noise.
- A GPU backend behind the same `Backend` protocol.
