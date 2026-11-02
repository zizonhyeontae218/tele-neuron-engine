# Model names after the 2026-11-02 owner decision

| Current name | Architecture | Commands / files |
| --- | --- | --- |
| Endophalon 3T | Previous separate-level and joint single-head Endophalon 3 experiments | Explicit CLI `python -m tele_neuron.endophalon3t`; historical compatibility path `tele_neuron.endophalon3`; existing `configs/endophalon3/`, `models/endophalon3/`, `models/endophalon3_joint/`, routing cards and reports |
| Endophalon 3 shared-state AB alpha experiment | Two independent A/B parameter/readout banks, one continuous position/velocity/clock/RNG state; externally commanded A/B selection | `python -m tele_neuron.endophalon3_alpha`; new `configs/endophalon3_alpha/`, `models/endophalon3_alpha/`, `docs/endophalon3_alpha_runs/` |

The rename is a current catalog/CLI name. Historical model JSON family strings, scientific source, configs, logs and dated reports are not rewritten. The original `tele_neuron.endophalon3` command retains its historical labels for reproduction. The explicit `endophalon3t` CLI calls the same implementation with the 3T name for newly generated CLI outputs. This does not turn a historical model into a newly trained model.

The new 3 format stores both bank configs and separate heads under `endophalon3-alpha/v1`; a session checkpoint is `endophalon3-session/v1`. Both are distinct from the preserved 3T format. A compatible saved bank can be imported; the new runtime revalidates the shared geometry, clock, seed and input encoding. Switches do not reset positions or velocities. Full parameter changes are externally controlled and may change stored energy; the runtime reports that change.

## Current software-linked 3

The subsequent owner direction keeps A/B physical sessions independent and links values in software. Current entry point: `python -m tele_neuron.endophalon3_software`. Model schema `endophalon3-software/v1`; router checkpoint schema `endophalon3-software-session/v1`. Paused independent local clocks and histories replace implicit physical memory sharing. Explicit registers pass actual output bits. The shared-state alpha implementation/cards above remain intact as previous experiments. See [software design and measurements](endophalon3_software.md).
