# DynaMath 002 checkpoints

Five Korean checkpoints and five independently trained synthetic recall checkpoints.
`model9` owns one runtime scalar per law; `model3-local` and `model3-cross` own one
per block (local-only versus all-block scalar reads); `model1` shares one scalar.
All have nine laws and one information state. `model9-frozen` is the matched
coefficient-count control with scalar writes disabled.

Load via `dynamath.v002.Model.load`. Korean checkpoints include optimizer, torch
RNG, training stream state/position/tick/hash. Session snapshots are separate JSON;
fresh inference starts from S=0 and canonical 0/1/e scalar constants. Recall models
have tiny fixed bit/query vocabularies, not Korean weights. No answer table or
trained encoder/output head is embedded. These are experimental, incoherent
language models; learned scalar recall was not demonstrated. See report and
artifact-integrity.json under docs/dynamath/experiment-002.
