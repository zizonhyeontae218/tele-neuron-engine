# Tele-Neuron Engine

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
