# DynaMath 003b

Canonical contract: docs/dynamath/experiment-003b/GOAL.md. Seed 43, fresh
Model1, word vocabulary <=8192 including controls 0..4, 50,000 Adam updates.
Preserve all earlier experiments. Nine baseline ASTs, one S and scalar m,
one formula/tick, three ticks/slot, actual FIFO feedback at inference.
No embedding/head/BPE, no test-driven selection. Restart lane clocks with S/m
at dialogue boundaries. Never claim partial training is complete.
Run pytest, scripts/filetree_index.py update and lint. Large checkpoints live
in ignored models/dynamath/003b and Actions artifacts, never repeated Git blobs.
