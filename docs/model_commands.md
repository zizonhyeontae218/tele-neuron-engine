# Model entry points

Run from the repository root after `python -m pip install -e . -r requirements.txt`.
Each model has its own module, configs, tests, docs and model-card directory.

| Generation | Model | Python module | Example config |
|---|---|---|---|
| 1 | Endophalon | tele_neuron.endophalon | configs/endophalon/xor.json |
| 1 | Plus | tele_neuron.endophalon_plus | configs/endophalon_plus/full_adder.json |
| 1 | Pro | tele_neuron.endophalon_pro | configs/endophalon_pro/parity3.json |
| 2 | Endophalon 2 | tele_neuron.endophalon2 | configs/endophalon2/base.json |
| 2 | 2 Pro | tele_neuron.endophalon2_pro | configs/endophalon2_pro/regular.json |
| 2 | 2 ProMax8 | tele_neuron.endophalon2_promax | configs/endophalon2_promax/promax8.json |
| 2 | 2 ProMax16 | tele_neuron.endophalon2_promax | configs/endophalon2_promax/promax16.json |
| 3 | 3-A (base, untrained) | tele_neuron.endophalon3 | configs/endophalon3/level_a.json |
| 3 | 3-B (base, untrained) | tele_neuron.endophalon3 | configs/endophalon3/level_b.json |

Train/evaluate with `python -m <module> --config <config>` using any row above.
Add `--output <new-path.json>` to save a new card. Do not overwrite original cards.
Training defaults are full experiments, not smoke workloads. `--help` lists each
module's actual options. ProMax8 and ProMax16 are configurations of the same
module, not separate Python packages.

`python -m pytest -q` runs all tests present in the current branch.
`python scripts/smoke_models.py` imports each available family and executes all
seven CLI/config combinations on reduced temporary configurations (three on v1).
This proves execution and serialization, not historical benchmark reproduction.

On master and archive/endophalon-v2, replay the original four ProMax cards with
`python scripts/replay_promax.py`. This uses saved theta/readout and the original
held-out streams, without retraining or overwriting any card.

Endophalon-3 is a base without trained models. Add `--probe` for the untrained
capacity probe. Training method, run order and know-how:
[endophalon3_handoff.md](endophalon3_handoff.md).
