# Contributing

Thanks for your interest in Tele-Neuron Engine.

This repository is currently a Phase 1 single-PC physical reservoir computing
prototype. Contributions should keep that scope clear and avoid claims that the
project replaces LLMs or practical production AI systems.

## Setup

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

## Before Opening A PR

Run:

```bash
python -m pytest
python -m tele_neuron.experiment --config configs/and_tiny.json
python -m tele_neuron.experiment --config configs/or_tiny.json
python -m tele_neuron.experiment --config configs/xor_tiny.json
```

## Good First Areas

- Improve learner algorithms in `Vanila_learner/Algorithm/`.
- Add larger configs under `configs/`.
- Improve visualization and Lab usability.
- Add benchmark reports for repeated experiment runs.

## Scope Guard

Please keep distributed phone-node execution, networking, and GPU-only paths out
of Phase 1 changes unless they are explicitly documented as future work.
