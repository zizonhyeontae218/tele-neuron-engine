# Model entry points

Run from the repository root after `python -m pip install -e . -r requirements.txt`.
Each model has its own module, configs, tests, docs and model-card directory.

| Generation | Model | Python module | Example config |
|---|---|---|---|
| 1 | Endophalon | tele_neuron.endophalon | configs/endophalon/xor.json |
| 1 | Plus | tele_neuron.endophalon_plus | configs/endophalon_plus/full_adder.json |
| 1 | Pro | tele_neuron.endophalon_pro | configs/endophalon_pro/parity3.json |

Train/evaluate with `python -m <module> --config <config>` using any row above.
Add `--output <new-path.json>` to save a new card. Do not overwrite original cards.
Training defaults are full experiments, not smoke workloads. `--help` lists each
module's actual options. Generation-2 commands and modules are available on master and archive/endophalon-v2.

`python -m pytest -q` runs all tests present in the current branch.
`python scripts/smoke_models.py` imports each available family and executes all
three generation-1 CLI/config combinations on reduced temporary configurations.
This proves execution and serialization, not historical benchmark reproduction.
