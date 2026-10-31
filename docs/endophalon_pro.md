# Teleneuron-Endophalon-Pro

A third, separate version next to Endophalon and Endophalon-Plus. It is a
first test of a next-generation idea: **train the physics itself** so that a
*small* linear readout is enough.

| Version | Physics | What is learned |
|---|---|---|
| Endophalon | one fixed system, 96 balls | linear readout only |
| Endophalon-Plus | three fixed presets, 768 balls | readout + a gate that picks a preset |
| **Endophalon-Pro** | **one trainable system, 144 balls** | **physics parameters θ + readout** |

Endophalon and Plus are unchanged. Pro reuses the Plus engine through its
`Backend` protocol and adds per-row stimulus points.

## Training loop

```txt
repeat (outer loop, cross-entropy search over θ):
  1. propose physics parameters θ
  2. run the physics for every (noise realization, input case)
  3. inner loop: solve the readout exactly, W*(θ) = ridge fit on train realizations
  4. score θ on validation realizations:
       J(θ) = L_val(θ, W*(θ)) + λ1·R_noise + λ2·R_stability + λ3·R_complexity
  5. move the search distribution toward the lowest-J candidates
```

The fitness of a physics system is the best readout it allows. Test
realizations are never seen by either loop, and they are used only for the
final numbers below.

### θ: what the outer loop can change

| Parameter | Range |
|---|---|
| damping | 0.80 – 1.00 |
| restitution | 0 – 1 |
| input strength | 0.01 – 2.0 (log scale) |
| initial speed | 0 – 0.15 |
| mass spread σ | 0 – 1.5 (masses `exp(σ·z)` with mean 1, `z` fixed per world) |
| stimulus points | x, y, z of each input's force source, inside the box |

θ has 17 dimensions for 3 inputs and 20 for 4 inputs. Every θ maps through
sigmoids to a valid system, so the search needs no constraints.

### J: the objective terms

| Term | Definition |
|---|---|
| `L_val` | mean squared error of `W*(θ)` scores on validation realizations |
| `R_noise` | within-case spread of the scores across realizations ÷ spread between case means (noise-to-signal ratio) |
| `R_stability` | share of balls pinned against a wall at readout time (strong inputs can jam the physics) |
| `R_complexity` | mean squared readout weight on standardized features |

Defaults: λ1 = 0.05, λ2 = 0.1, λ3 = 0.01.

### Setup

- World: `5 × 5 × 3` box, 144 balls, readout at step 32.
- Readout: voxel occupancy on a coarse grid, with **4** features (`2×2×1`)
  or **9** features (`3×3×1`). Endophalon used 48 and Plus used 1156.
- Noise: position jitter 0.1, velocity jitter 0.02. 24 train, 12 validation
  and 24 test realizations.
- Outer loop: 30 generations × 16 candidates, 481 θ evaluations in total.
  Takes 5–9 minutes on 4 cores.

## Results

Test accuracy counts a sample correct only if every output bit is right.
Raw data: `docs/endophalon_pro_matrix.json`.

### Trained vs untrained physics, same readout size

| Task | Readout | Initial θ | Random search (same budget) | Trained θ | Trained θ, λ = 0 | Bits linear |
|---|---|---|---|---|---|---|
| 3-bit parity | 4 | 75.0% | 76.0% | **96.9%** | 92.2% | 50.0% |
| 3-bit parity | 9 | 85.4% | 98.4% | 98.4% | 96.9% | 50.0% |
| Full adder | 4 | 37.0% | 68.8% | **83.9%** | 82.3% | 50.0% |
| Full adder | 9 | 75.5% | 97.4% | 93.8% | **99.5%** | 50.0% |
| 4-bit parity | 9 | 78.9% | 78.4% | 84.6% | **85.7%** | 50.0% |
| 2-bit adder | 9 | 35.2% | 63.3% | 68.0% | **71.1%** | 25.0% |

### Readout dependence: accuracy vs readout size

The physics is trained once at the size in the first column. The readout is
then refit at every other size.

| Task (trained at) | Physics | 2 | 4 | 9 | 16 | 36 | 144 |
|---|---|---|---|---|---|---|---|
| 3-bit parity (4) | initial | 41.7% | 75.0% | 85.4% | 90.1% | 88.5% | 72.4% |
| | **trained** | 63.0% | **96.9%** | 99.5% | 100% | 99.5% | 95.3% |
| Full adder (4) | initial | 34.4% | 37.0% | 75.5% | 86.5% | 87.5% | 68.8% |
| | **trained** | 25.0% | **83.9%** | 93.8% | 96.9% | 98.4% | 93.2% |
| 2-bit adder (9) | initial | 25% | 21% | 35% | 50% | 58% | 61% |
| | trained | 15% | 9% | **68%** | 66% | 76% | 79% |

Model cards (3-bit parity and full adder, trained with a 4-feature readout):

- `models/endophalon_pro/teleneuron_endophalon_pro_parity3_k4.json`
- `models/endophalon_pro/teleneuron_endophalon_pro_full_adder_k4.json`

## What this shows

- **Training the physics reduces how big the readout must be.** On 3-bit
  parity, trained physics with 4 features beats untrained physics at *every*
  readout size up to 144. Untrained physics never reaches 90% on the full
  adder, at any size. On the 2-bit adder, trained physics with 9 features
  matches untrained physics with 144.
- **The physics specializes to its readout.** θ trained for a 4-feature
  readout is strong at 4 features. θ trained for 9 features can be *worse*
  than untrained physics at 4 features (3-bit parity: 57% vs 75%; full
  adder: 50%). Physics and readout are tuned as a pair.
- **The bilevel search matters most when the readout is small.** At 4
  features it clearly beats random search with the same budget (96.9% vs
  76.0%, 83.9% vs 68.8%). At 9 features, random search around θ₀ is about as
  good, and once better (full adder: 97.4% vs 93.8%). With a bigger readout
  there are many good θ near the start, so plain sampling finds one.
- **Consistent choices across runs:** mass spread always goes to ≈0 (uniform
  masses), and restitution rises to 0.91–0.98. Damping stays at 0.986–0.997,
  except on the 2-bit adder, where it drops to 0.95 and 0.86 (more friction).
  With a 4-feature readout the initial speed drops (0.05 → 0.005–0.03), so
  the input drives the motion instead of the initial thermal motion.
- **"Readout dependence" fell in size, not in weight.** `R_complexity` (the
  readout weight size) *rose* after training in every run. The physics lets
  fewer features work, but those features carry larger weights. λ3 = 0.01 is
  too small to push back, and that is a natural next knob.

## Caveats

- **One seed per cell.** Differences under ~3 points (e.g. regularized vs
  λ = 0 at 9 features) are not established.
- **Regularizers help only with the tiny readout.** With 4 features they
  helped (3-bit parity 96.9% vs 92.2%). With 9 features, λ = 0 was as good
  or better in 3 of 4 cells.
- **The objective changed once, and the change is on record.** The first run
  (v0, also in the JSON) used raw score spread as `R_noise` with λ1 = 0.5.
  The readout could lower that term by shrinking every score toward a
  constant, and those regularized runs scored *below* λ = 0 (full adder:
  65.6% vs 82.3% at 4 features). `R_noise` was then made scale-free and
  λ1 lowered to 0.05. All tables above are v1. The change was motivated by
  the definition, but it was made after seeing results.
- **Test sets are small.** 24 realizations × 8 or 16 cases, so one sample is
  0.3–0.5%.
- **Stimulus points can move.** The trained systems are not directly
  comparable to Endophalon's fixed geometry. That is part of θ by design.

## Run

```bash
python -m tele_neuron.endophalon_pro --config configs/endophalon_pro/parity3.json --grid 2x2x1
python -m tele_neuron.endophalon_pro --config configs/endophalon_pro/full_adder.json --random-baseline \
  --output models/endophalon_pro/full_adder.json
python -m tele_neuron.endophalon_pro.matrix --tasks parity3 full_adder --ablation
python -m tele_neuron.endophalon_pro.matrix --tasks parity4 adder2 --grids 3x3x1 --ablation
```

## Toward Endophalon 2

The target architecture is a nonlinear physics system: inputs hit chosen
stimulus points, and a linear readout reads the state at a set time. Pro
already has that shape: learned stimulus points, a fixed readout step and a
linear readout. Natural next steps:

- **Make the readout time part of θ** ("output after a specific time").
- **A stronger λ3, or a hard cap on readout features,** to push the remaining
  work from the readout into the physics.
- **Multiple seeds per cell,** and a gradient-free optimizer with step-size
  adaptation (e.g. full CMA-ES) for the 4-input tasks, where 30 generations
  were not enough to saturate.
