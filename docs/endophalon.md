# Teleneuron-Endophalon

Endophalon is a **separate model family** from TeleNeuron-001, not its next
version. The two share the same ball physics but put the learning in different
places.

| | TeleNeuron-001 | Teleneuron-Endophalon |
|---|---|---|
| What is learned | The physics itself (seed, particle masses) | Only a linear readout |
| Physics | Searched until one zone answers | Fixed, untrained reservoir |
| Observation | Particle count in one zone at step K | Voxel occupancy at several tap steps |
| Readout | `count >= threshold` | Ridge regression, closed form |
| Outputs | One bit per hand-placed zone | Any number of bits from the same features |
| Evaluation | Exact replay of one seed | Held-out jittered initial states |

No backpropagation is used in either family.

## Pipeline

```txt
task truth table
  -> BallReservoir.run(bits, realization)   # fixed physics, jittered start
  -> VoxelObserver.features(snapshots)      # occupancy per voxel per tap
  -> RidgeReadout.fit / predict             # W = (X^T X + aI)^-1 X^T Y
  -> model card JSON
```

Code lives in `src/tele_neuron/endophalon/`:

- `tasks.py` — truth tables: `and`, `or`, `xor`, `half_adder`, `majority3`,
  `parity3`, `full_adder`.
- `reservoir.py` — `Reservoir` protocol and `BallReservoir`, which reuses the
  Phase 1 primitives (`input_forces`, `reflect_bounds`, `resolve_collisions`).
- `observer.py` — `VoxelObserver`, fraction of balls per voxel at each tap.
- `readout.py` — `RidgeReadout` with feature standardization and an
  unpenalized bias.
- `train.py` — dataset collection, train/test split by realization, baseline.
- `model.py` — model card save/load and prediction.

## Noise and honest evaluation

A truth table has only `2^n` inputs, so fitting it is not evidence by itself.
Each *realization* starts from the seeded base state plus Gaussian jitter on
positions and velocities. The readout is fitted on `train_realizations` and
scored on separate `test_realizations` it never saw, so it has to rely on
input-driven structure that survives chaotic collisions.

Every run also reports a **baseline**: the same ridge readout fitted directly
on the raw input bits. A linear map of the bits cannot express XOR or parity,
so the baseline stays at 50% on those outputs. The gap between baseline and
reservoir is the nonlinearity the physics contributes.

## Run

```bash
python -m tele_neuron.endophalon --config configs/endophalon/xor.json
python -m tele_neuron.endophalon --config configs/endophalon/parity3.json
python -m tele_neuron.endophalon --config configs/endophalon/full_adder.json \
  --model-id teleneuron_endophalon_full_adder \
  --output models/endophalon/teleneuron_endophalon_full_adder.json
```

## Current results

96 balls, 24 steps, taps `[8, 16, 24]`, grid `[4, 4, 1]` (48 features),
ridge alpha 10, position jitter 0.1, velocity jitter 0.02,
48 train / 16 test realizations.

| Task | Train | Test (held out) | Baseline test |
|---|---|---|---|
| XOR | 100.00% | 100.00% | 50.00% |
| 3-bit parity | 98.70% | 96.09% | 50.00% |
| Full adder (sum + carry both right) | 98.70% | 96.09% (carry 100%) | 50.00% |

The remaining test errors concentrate on inputs `011` and `111`.

## Tuning notes

- With jitter 0.02 every task reaches 100% test; that noise level is too easy
  to say much, so the bundled configs use 0.1 (about one ball diameter).
- At jitter 0.1, a `[4, 4, 2]` grid with only 16 train realizations overfits
  (train 100%, full adder test ~81%). More realizations and a flatter grid fix
  most of it.
- Above jitter ~0.4 test accuracy falls toward ~70%.
