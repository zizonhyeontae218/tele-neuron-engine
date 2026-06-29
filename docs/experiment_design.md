# Phase 1 Experiment Design

This document records the first Tele-Neuron Engine experiment design used to
complete `GOAL.md`.

## Scope

Phase 1 runs on one PC only. It does not implement distributed phone nodes,
networking, GPU-only code, or any claim that this replaces LLMs.

## Reservoir

- Space: 3D box, currently `4 x 4 x 3` for tiny tasks.
- State: ball position, velocity, mass, radius.
- Determinism: every run is controlled by an explicit config seed.
- Numeric type: NumPy float64.

## Input Encoding

Each binary task uses two input points. For each bit set to `1`, nearby balls
receive an outward force:

```txt
direction = normalize(ball_position - input_point)
force = direction * input_strength / max(distance_squared, epsilon)
```

Bits set to `0` inject no force.

## Physics

Each case runs for `K = simulation.steps` iterations:

```txt
velocity = velocity * damping + acceleration * dt
position = position + velocity * dt
```

Balls reflect at the simulation boundary by clamping position and reversing the
velocity component for that axis.

## Collision Detection And Response

Collision candidates use spatial hashing:

```txt
cell = floor(position / cell_size)
```

Only the current cell and its 26 neighbors are checked. A pair collides when:

```txt
dx^2 + dy^2 + dz^2 <= (rA + rB)^2
```

The Phase 1 response separates overlapping balls and applies a simple impulse
with configurable restitution.

## Readout

The readout is count-based:

```txt
output = 1 if count_in_zone >= threshold else 0
```

The output zone is configured per experiment. The current implementation logs
readout counts and collision counts for each input case.

## Search

No backpropagation is used. The initial trainer searches deterministic seeds
and candidate thresholds, then stores the best score in `outputs/*.json`.

## First Model

`TeleNeuron-001 XOR` is the first fixed model card:

```txt
models/tele_neuron_001.json
configs/xor_first_model.json
```

It solves the XOR tiny task with `4/4` correct cases under the current Phase 1
physics and readout.
