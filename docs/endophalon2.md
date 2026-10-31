# Teleneuron-Endophalon-2

The fourth, separate version, and the first step toward the target
architecture: **continuous nonlinear computation in a physics system that
never stops**. Endophalon, Plus and Pro are unchanged.

Every earlier version reset the physics for each input. Endophalon-2 keeps
one physics system running. Input pulses hit stimulus points every
`interval` steps, on top of whatever the previous pulses left behind. A linear
readout reads soft readout zones `n` steps after each pulse.

```txt
warmup ── pulse 0 ──(n)── read ── pulse 1 ──(n)── read ── pulse 2 ── … (64 pulses, no reset)
          a0,b0                   a1,b1                   a2,b2
```

## Questions and tasks

Two input channels per pulse (`a_k`, `b_k`):

| | Question | Tasks |
|---|---|---|
| Q1 | Can it compute a nonlinear function of the current pulse while carrying the previous ones? | `xor`: a_k ⊕ b_k (`and` as a sanity check) |
| Q2 | Does it remember earlier pulses? | `recall1/2/3`: a_{k−1}, a_{k−2}, a_{k−3} |
| Q3 | Can it use memory and nonlinearity together? | `xor_mem`: a_k ⊕ a_{k−1}; `parity3w`: a_k ⊕ a_{k−1} ⊕ a_{k−2} |

**Persistence:** each test score is also split into the first and second
half of the stream. A system that degrades over time shows a lower second
half.

**Success criterion (agreed up front):** Q1 in continuous mode within 5
points of the reset baseline, held through the end of the stream.

## Design

### Engine (stage 0)

`src/tele_neuron/endophalon2/engine.py` adds two pieces to the Pro backend:

- **Rest dynamics.**
  - *Anchor spring:* each ball is pulled weakly toward its home position.
    Without it, pulses push balls into a configuration that never recovers.
  - *Heat bath:* a small random velocity kick each step. Without it, damping
    freezes the system.

  Both cost O(N) per step, which is small next to collision handling. Their
  strengths are part of θ.
- **Soft readout zones.** K zones with learnable centers and a shared width.
  Each zone reports the Gaussian-weighted share of balls near its center,
  i.e. it reads "around the readout zone".

The state is recorded at every delay 1…interval after each pulse onset, so
the readout delay can be chosen afterwards. With rest dynamics off, the engine
reproduces the Pro engine exactly. Results do not depend on chunking or worker
count, the heat bath keeps a damped system moving, and the anchor pulls pushed
balls back (all tested).

### Training (stage 2)

The bilevel loop from Pro, adapted to streams:

| | Pro | Endophalon-2 |
|---|---|---|
| sample | one reset run | one pulse inside a 64-pulse continuous stream |
| split | noise realizations | **whole streams**: 16 train / 8 validation / 16 test |
| transient | none | first 4 pulses of each stream are dropped (washout) |
| inner loop | one ridge readout | one exact ridge readout **per task**, at delay n = 12 |
| θ | physics + stimulus points | physics + anchor + heat bath + pulse strength + stimulus points + **readout zone centers and width** (25 dims with 4 zones) |
| J | L_val + noise + stability + complexity | mean-task L_val + λ1·R_noise + λ2·R_drift + λ3·R_complexity |

- `R_noise`: within-class spread of the readout scores ÷ the gap between class
  means.
- `R_drift`: |log(late energy ÷ early energy)| plus the share of balls pinned
  to a wall late in the stream.
- Weights: λ1 = 0.05, λ2 = 0.1, λ3 = 0.01.
- Outer loop: 30 generations × 16 candidates, 481 θ in total. Each run takes
  about 15 minutes on 4 cores.
- Physics: 144 balls in a `5 × 5 × 3` box, interval 16 steps, pulses 4 steps
  long.

## Stage 1: untrained physics (`docs/endophalon2_feasibility.json`)

Test accuracy on held-out streams, with the delay n picked on validation:

| Variant | Readout | XOR, continuous | XOR, reset per pulse | recall1, continuous | Energy, early → late |
|---|---|---|---|---|---|
| no rest dynamics | 9-cell grid | 54% | 98% | 71% | 0.027 → 0.001 (freezes) |
| anchor | 9-cell grid | 71% | 100% | 87% | 0.015 → 0.074 (heats up) |
| anchor + bath | 9-cell grid | 69% | 99% | 89% | 0.017 → 0.017 (stable) |
| anchor + bath | 36-cell grid | 70% | 100% | 87% | stable |
| anchor + bath | 4 soft zones | 56% | 90% | 71% | stable |

- **Physics-free baseline (last 3 bit pairs, linear readout):** XOR 48%,
  recall 100%, `xor_mem` 56%.
- **Takeaway.** The same physics that computes XOR almost perfectly from a
  reset collapses toward chance once it keeps running. What the previous
  pulses left behind swamps the current one. Rest dynamics make the run
  stable, so energy is flat and early-stream and late-stream accuracy match.
  They do not close the XOR gap, and a bigger readout grid barely helps (69%
  → 70%). The physics, not the readout size, was the bottleneck.

## Stage 2: trained physics

Test accuracy on 16 held-out streams × 60 scored pulses, readout at n = 12.
Each cell shows the whole stream, with the first half → second half in
parentheses.

| Task | Initial θ (continuous) | **Trained θ, multi-task, 4 zones** | Trained θ, XOR only, 4 zones | Trained θ, multi-task, 8 zones | Random search (multi-task, 4 zones) | Trained θ, reset per pulse |
|---|---|---|---|---|---|---|
| and | 76% | **100%** (100 → 100) | 93% | 87% | 78% | 100% |
| **xor** (Q1) | 55% | **99%** (99 → 99) | 90% | 74% | 60% | 100% |
| recall1 (Q2) | 72% | **86%** (87 → 86) | 71% | 79% | 67% | 51% |
| recall2 | 59% | 48% | 63% | 67% | 56% | 51% |
| **xor_mem** (Q3) | 57% | **84%** (84 → 85) | 58% | 73% | 55% | 49% |
| parity3w | 48% | 49% | 46% | 57% | 54% | 47% |

- "Multi-task" means θ was trained for `xor`, `recall1` and `xor_mem`
  together, each with its own readout.
- Physics-free baseline: XOR 48%, `xor_mem` 56%.

Model cards (θ, one readout per task, all metrics):

- `models/endophalon2/teleneuron_endophalon2_multi_k4.json`
- `models/endophalon2/teleneuron_endophalon2_xor_k4.json`
- `models/endophalon2/teleneuron_endophalon2_multi_k8.json`

## What this shows

- **The success criterion is met.** After training, XOR on the current pulse
  in a never-reset physics system is 99%, against 100% for the reset baseline.
  It holds from the first half of the stream to the second (99% → 99%), with
  only 4 soft readout zones.
- **The running physics computes things a reset system cannot.** `xor_mem`
  (current pulse ⊕ previous pulse) reaches 84% and `recall1` 86%. Reset runs
  are at chance on both (49–51%), because they have no history. The
  physics-free linear readout on raw bits gets 56% on `xor_mem`, so that 84%
  needs the physics' nonlinearity *and* its memory. That is the core of
  "continuous nonlinear computation".
- **The memory is short: about one pulse.** `recall2` and 3-pulse parity
  stayed at chance in the 4-zone multi-task run.
- **How the physics changed.**
  - Damping dropped from 0.99 to 0.81 (heavy friction), and the anchor spring
    got 10× stiffer (0.002 → 0.023): the system now forgets quickly, settling
    within about one interval.
  - Pulse strength rose 4× (0.08 → 0.31), so each new pulse dominates what is
    left of the last one.
  - The heat bath stayed small (0.002).
  - Mass spread went to 0 again, as in Pro.
  - In short, training traded memory depth for a clean current-pulse signal,
    plus a one-pulse echo.
- **Training on the harder mix helped XOR itself.** The XOR-only run reached
  90%, the multi-task run 99%. Same seed and budget, single runs.
- **8 zones was worse than 4** (XOR 74%). θ grows from 25 to 37 dimensions,
  and 30 generations were not enough: the search was still moving (σ 0.07 at
  the end, J still falling slowly).
- **Search beats random sampling clearly.** With the same 481-θ budget,
  random sampling around θ₀ reached XOR 60% and `xor_mem` 55%.

## Caveats

- **One seed per run.** In particular, "multi-task beats XOR-only" and "4
  zones beat 8" are single observations.
- **`R_drift` is a noisy metric, and the search exploited it.** Per-stream
  energy varies a lot (late/early ratios from 0.04 to 6.8 across test
  streams), because each stream's random bits inject different energy.
  `R_drift` pools streams, so on the train+val streams a rising half and a
  falling half can cancel. The search picked θ with `R_drift` ≈ 0.01 on those
  streams, but the same θ scores 0.15–0.85 on test streams, which is the same
  range as the untrained θ (0.53). The persistence that matters is not
  affected: early-vs-late test accuracy is flat in every run. Fix for next
  time: a per-stream drift (median per-stream log ratio, or an energy trend
  slope).
- **Fixed timing.** Interval 16, pulse length 4, readout at n = 12. Irregular
  intervals are deferred, as agreed.
- **The readout delay was fixed during training.** Stage 1 picked n per task
  on validation, while stage 2 trains and scores at n = 12 only.

## Run

```bash
# stage 1: untrained physics, continuous vs reset
python -m tele_neuron.endophalon2.feasibility --config configs/endophalon2/base.json
# stage 2: bilevel training
python -m tele_neuron.endophalon2 --config configs/endophalon2/base.json --tasks xor recall1 xor_mem \
  --random-baseline --output models/endophalon2/multi_k4.json
```

## Next

1. **Fix `R_drift`** (per-stream), and run 2–3 seeds per configuration.
2. **Deeper memory.** Train with `recall2` / `parity3w` in the mix, to see
   whether the physics can trade some of the current-pulse signal for a longer
   echo.
3. **Irregular pulse intervals** (deferred from the plan).
4. **Learn the readout delay n**, now that the physics settles within about
   one interval.
