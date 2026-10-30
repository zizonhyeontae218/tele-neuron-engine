# Tele-Neuron Engine

> **한국어:** [README.ko.md](README.ko.md)

Phase 1 single-PC prototype for a physical reservoir computing experiment.
Input bits inject force into virtual balls in a 3D space, the balls move and
collide for K steps, and readout zones convert the final distribution into
output bits.

This is a physical reservoir computing prototype, not a practical LLM
replacement.

## What This Is

- A deterministic Python + NumPy simulation scaffold.
- A tiny experiment target for AND/XOR style binary classification.
- A base for spatial hashing, collision response, readout, visualization, and
  seed search.

## What This Is Not

- It is not a distributed phone-node system yet.
- It is not GPU-only code.
- It is not a claim that physical collision simulation replaces modern LLMs.

## Setup

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

## Current Status

Phase 1 is implemented as a single-PC prototype. It includes deterministic
ball initialization, input force injection, K-step physics, reflective
boundaries, spatial hashing collision candidates, simple collision response,
count-based readout, seed search, score logging, and Matplotlib visualization.

## Run

```bash
pytest
python -m tele_neuron.experiment --config configs/and_tiny.json
python -m tele_neuron.experiment --config configs/or_tiny.json
python -m tele_neuron.experiment --config configs/xor_tiny.json
```

## Watch The Simulation

Open a live Matplotlib display where x/y are position and z is shown as color:

```bash
python -m tele_neuron.experiment --config configs/and_tiny.json --display --bits 11
```

Save the same view as a GIF:

```bash
python -m tele_neuron.experiment --config configs/xor_tiny.json --display --bits 01 --save outputs/xor_01.gif
```

Save final K-step positions as a PNG:

```bash
python -m tele_neuron.experiment --config configs/xor_tiny.json --bits 01 --snapshot outputs/xor_01_final.png
```

## Config Lab

Open the lightweight browser editor for coordinates, emitters, readouts, cases,
and preview snapshots:

```bash
python -m tele_neuron.config_lab
```

The lab saves edited configs to `configs/lab/` and preview PNGs to
`outputs/lab/`. Original configs are not overwritten.

## Learner Lab

Open the browser trainer for config-based model creation and continued model
training:

```bash
python -m tele_neuron.learner_lab
```

Algorithms are loaded from `Vanila_learner/Algorithm/*.py`. The lab can create
a new model with random per-ball masses or continue from an existing model JSON.
Trained models are saved to `models/lab/`.

## First Model

`TeleNeuron-001` is the first reproducible tiny XOR model found by seed search:

```bash
python -m tele_neuron.experiment --config configs/xor_first_model.json
python -m tele_neuron.experiment --config configs/xor_first_model.json --display --bits 01
```

Model card:

```txt
models/tele_neuron_001.json
```

Example output:

```txt
tele_neuron_001_xor seed=1 threshold=3 score=4/4 accuracy=100.00%
bits=00 target=0 output=0 count=2 collisions=8
bits=01 target=1 output=1 count=7 collisions=33
bits=10 target=1 output=1 count=8 collisions=34
bits=11 target=0 output=0 count=2 collisions=28
```

## Teleneuron-Endophalon

A separate model family, not a successor to 001. The physics is kept fixed as a
reservoir, voxel occupancy at several time steps becomes a feature vector, and
only a linear readout is fitted with closed-form ridge regression. It solves
3-bit parity and the full adder (sum + carry) at ~96% on held-out jittered
initial states, where a linear readout on the raw bits stays at 50%.

```bash
python -m tele_neuron.endophalon --config configs/endophalon/full_adder.json
```

Details: [docs/endophalon.md](docs/endophalon.md). Model cards:
`models/endophalon/`.

## Teleneuron-Endophalon-Plus

A separate version that sits next to Endophalon. It is not a replacement:
both are kept and both are run. Endophalon asks whether a small, fully fixed
reservoir can compute at all. Plus asks what scale and partial learning add.
Plus runs on its own **Plus physics base**, a batched engine that runs many
simulations as one `(B, N, 3)` array and splits them across worker processes:
about 9.5× faster per core, about 37× faster on 4 cores. The world
has 768 balls, and each tap is observed with 1156 features per preset.
Half-learning: three fixed physics presets, and a learned *linear* gate that
picks one preset per input. On 4-bit parity at jitter 1.0, routing reaches
100% versus 96.5% for the best single preset. A readout that sees only the
gate's choice stays at 50%.

```bash
python -m tele_neuron.endophalon_plus --config configs/endophalon_plus/parity4_stress.json --workers 4
```

Details and the full results table: [docs/endophalon_plus.md](docs/endophalon_plus.md).

## Teleneuron-Endophalon-Pro

A third, separate version, and a first test of a next-generation idea: train
the physics itself so that a *small* linear readout is enough. There is one
trainable physics system with 144 balls. An outer loop searches over its
parameters θ (damping, restitution, input strength, initial speed, mass
spread, stimulus point positions). For each candidate, an inner loop solves
the linear readout exactly. θ is scored by
`J = L_val + λ1·R_noise + λ2·R_stability + λ3·R_complexity` on validation
noise realizations.

On 3-bit parity, trained physics with a **4-feature** readout scores 96.9% on
held-out noise. Untrained physics stays at or below 90.1% at every readout
size up to 144 features.

```bash
python -m tele_neuron.endophalon_pro --config configs/endophalon_pro/parity3.json --grid 2x2x1
```

Details, the full results and caveats: [docs/endophalon_pro.md](docs/endophalon_pro.md).

## Teleneuron-Endophalon-2

The fourth, separate version, and the first step toward the target
architecture: **continuous nonlinear computation**. The 144-ball physics is
never reset. Input pulses hit learned stimulus points every 16 steps on top
of the running state, and a linear readout reads 4 learned soft readout zones
12 steps after each pulse. An anchor spring and a heat bath keep the long run
stable.

- **Untrained:** the physics does XOR at 99% when reset per pulse, but only
  55–70% while running.
- **Trained** with the bilevel loop: 99% while running, steady from the first
  half of each stream to the second. Tasks a reset system cannot do at all
  also work: current pulse XOR previous pulse at 84%, recall of the previous
  pulse at 86%.

```bash
python -m tele_neuron.endophalon2 --config configs/endophalon2/base.json --tasks xor recall1 xor_mem
```

Details, results and caveats: [docs/endophalon2.md](docs/endophalon2.md).

## Teleneuron-Endophalon-2-Pro

The fifth, separate version. It keeps Endophalon-2's protocol (the physics
keeps running, each input pushes it, the readout reads 12 steps later), with
two changes:

- **New physics:** a **nonlinear spring network**. 144 nodes, 506 springs,
  no collision detection.
- **Irregular input timing:** inputs can arrive at uneven gaps of 12–24
  steps.

Results with 4 readout zones:

| Timing | xor | recall1 | xor_mem |
|---|---|---|---|
| Regular, best seed | **100%** | **100%** | **100%** |
| Irregular | 90–95% | 86–94% | 83–84% |

- The ball version (Endophalon-2) reached 99 / 86 / 84 on the same tasks.
- A purely linear readout of the trained network's state does XOR at 100% and
  3-input parity at about 100%. So the nonlinearity and memory are in the
  physics, and the small readout is now the limit. With 8 zones, deep-memory
  training reaches recall2 97% and 3-input parity 80%.

```bash
python -m tele_neuron.endophalon2_pro --config configs/endophalon2_pro/regular.json --tasks xor recall1 xor_mem
```

Details, results and caveats: [docs/endophalon2_pro.md](docs/endophalon2_pro.md).

## Teleneuron-Endophalon-2-ProMax8 / ProMax16

The sixth, separate version: **one physics system and one readout**, with the
operation chosen by input. Each event carries two bits and an opcode (`xor`,
`recall1` or `xor_mem`), which can change on every event. The opcode pushes
the spring network at its own stimulus point, and a single linear readout
over 8 (ProMax8) or 16 (ProMax16) zones must give that operation's answer.

- **Results:** after training, the single readout is right on 91–97% of
  events, against 67–71% untrained and 78–81% for random search.
- **The routing happens in the physics:** giving the opcode to the readout
  as well changes almost nothing.

```bash
python -m tele_neuron.endophalon2_promax --config configs/endophalon2_promax/promax16.json
```

Details and caveats: [docs/endophalon2_promax.md](docs/endophalon2_promax.md).

## Known Limitations

- The physics model is intentionally simple and tuned only for tiny binary
  tasks.
- Collision response is a stable approximation, not a full rigid-body engine.
- Seed search and threshold selection are enough for Phase 1, but not a general
  learning algorithm.
- Distributed phone nodes, networking, GPU-only paths, and LLM replacement
  claims are out of scope.

## License

MIT. See [LICENSE](LICENSE).

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and
keep Phase 1 scope clear: this is a physical reservoir computing prototype, not
a practical LLM replacement.
