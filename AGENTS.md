# Tele-Neuron Engine agent instructions

DynaMath 1 exception (2026-11-12): the owner explicitly authorized the isolated
feature/dynamath-1 branch and its new experiments. This overrides the historical
new-branch prohibition below only for that work; never merge it into master or
endophalon3. See docs/dynamath/GOAL.md and src/tele_neuron/dynamath/AGENTS.md.

Read GOAL.md, FileTree.md, docs/model_commands.md and docs/endophalon3_software.md.
Current 3 is ONLY independent A/B, connected by explicit predicted-output registers.
A/B each own physical state, clock, RNG and local history. Pause the inactive model.
144 nodes, 2 substeps, gap16, delay12, 16 zones per active model. No oracle input.

The owner authorized deletion of obsolete 3T/shared-state/old model artifacts;
experimental history is consolidated in docs/endophalon3_development.md.
Do not restore discarded versions or launch new searches without a new task.
Current A/B training is 120 generations each; report failures and accuracy gates plainly.
Preserve current models/configs/seeds, current reproducibility evidence and all v1/v2.

master and endophalon3 must stay aligned. No new remote work branch, force push,
history rewriting or unrelated branch deletion. Leave Claude, generation archives
and iPad Lab branches intact; retain ipad_lab.html and v1/v2 backup consolidation.

Commands: python -m pytest -q; python scripts/smoke_models.py;
python scripts/verify_backup_consolidation.py; python scripts/filetree_index.py update;
python scripts/filetree_index.py lint. Install: python -m pip install -e . -r requirements.txt.
Model training/audit commands are in docs/model_commands.md.
