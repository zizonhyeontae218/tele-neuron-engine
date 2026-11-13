# DynaMath 003: structural law search

Run from the repository root on Python 3.12 with CPU PyTorch and pytest:

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install pytest
PYTHONPATH=src python scripts/run_dynamath_003.py
PYTHONPATH=src python scripts/check_dynamath_003.py
PYTHONPATH=src python -m pytest -q tests/test_dynamath_003.py tests/test_dynamath_002.py
PYTHONPATH=src python -m dynamath.v003 models/dynamath/003/law1-final.pt --question '안녕하세요' --ticks 180 --snapshot /tmp/003-snapshot.json
PYTHONPATH=src python -m dynamath.v003 models/dynamath/003/law1-final.pt --restore /tmp/003-snapshot.json --ticks 10
```

The runner downloads the pinned Song Ys CSV when missing and verifies its frozen SHA-256 using the existing loader. Inference and snapshot restore require only the checkpoint; no corpus is loaded. Running the experiment overwrites its 003 results/checkpoints. `evaluation-plan.json` freezes the IDs; `results-003.json` contains raw candidates, training history, validation and final test data, and FIFO snapshots; `supplemental.json` contains ablation and synthetic checks. The scripts use first sorted IDs and fixed seeds. Existing source/data/checkpoints remain untouched.

## Design and scope

We reuse the 002 model and its Session protocol, and replace only the candidate nonlinearity in each of nine individually parameterized laws with a validated scalar/vector elementwise AST. The low-rank affine computation, recurrent blend, one writable runtime scalar, address layout, input queue, two inspections, and feedback all retain 002 semantics. ASTs accept only exact, shape-preserving `input`, `tanh`, `sin`, `scale`, `add`, and `clip` nodes, with depth <=6; no eval or arbitrary code. Output of the offered mutations is bounded. The deterministic outer search tries the same identity-initialized mutation at F1, F2, and F3 and selects lowest validation response CE, including an unedited control. Adam adapts all continuous parameters inside each candidate; discrete edits are not differentiated. The grammar is intentionally narrow and this is not a general-purpose genetic programming system. Additional search across the other six laws, richer operators, seeds and budgets is untested.

F1 before: `tanh(z)`. F1 after: `clip(tanh(z) + tanh(α)·sin(z))`, where `z = 1.5 tanh(d₁)⊙S + tanh(SV₁+b₁+mQ₁)U₁+c₁+mR₁`; clipping is to [-1,1]. The full state update is `(1-a₁)⊙S + a₁⊙expression(z)`, `a₁=.05+.9 sigmoid(leak₁)`. This adds clip, add, scale, sin and a second input occurrence (7 nodes/6 edges versus 2 nodes/1 edge). α starts at zero and the insertion initially exactly matches the parent, including the clip because tanh is bounded. All arms allocate the same nine α parameters (78,003 total), though only an edited AST uses its own α.

## Findings and limitations

The frozen 20-update search used the first 512 training and 32 validation rows, batch 8, eight slots/update, and one thread pair. Validation response CE was control 1.6354311, F1 1.6354165, F2 1.6354505, F3 1.6354285. F1 was selected solely on validation; the difference is tiny. Both F1 and control then received 20 additional updates with optimizer and recurrent training state resumed. On the 32 frozen test rows, control/F1 response CE was 1.6162475/1.6162532, PPL 5.034164/5.034193, and accuracy 0.49414/0.49414. Thus the selected structural edit did **not** improve final held-out performance. The untouched 002a/002b checkpoints scored CE 1.638349/1.502529, PPL 5.146666/4.493038 and accuracy 0.47571/0.51424 on this small subset; this is not a compute-matched comparison with their prior training.

Three fixed test prompts were generated with 180 ticks and unconstrained FIFO feedback. F1 and control generated the same texts, one incomplete with an unattached Jamo and two EOS; no question-time emissions. The 002b outputs included unattached Jamo and question-time emissions on all three despite its better CE. Raw text, controls, EOS, omissions, peak/final queues and snapshots are retained. This is too small to establish conversational quality. Second-check SILENCE supervision and the old teacher-forced schedule remain unchanged; their mismatch with closed-loop behavior remains unresolved.

A restore-parent AST ablation on validation changes CE by only 4.5e-7; learned F1 α is -0.0000332 and the observed one-tick state difference is 2.75e-5. This confirms a real but negligible causal effect, not a meaningful discovered language law. In an independently specified bounded scalar recurrence whose target contains the offered sine term, the same search family reduced held-out one-step MSE from 0.02449 to 5.89e-7 and nine-step error from 0.715 to 0.0. This favorable target was deliberately chosen to exercise the grammar; it is not independent evidence of open-ended discovery. A separately trained tiny bit recall task reached 100% at delays 0, 5, 20 for both arms; resetting shared state reduced accuracy to the test class proportions, while resetting the scalar did not. It demonstrates this tiny task uses state, not that the scalar stores the bit or that language recall works. See supplemental JSON for exact trials and seed.

Each arm ran exactly 20 updates × 8 slots × 3 ticks × 8 lanes = 3,840 lane-state transitions in search; two arms continued for another 3,840 each. Total training was 23,040 lane transitions plus evaluation/generation and synthetic checks. Per-law low-rank matrix products cost approximately 4×288×12 = 13,824 floating-point multiply/add operations per lane/tick, excluding bias, scalar gates, elementwise nonlinearities, optimizer and backprop. The mutation adds one 288-element sine, multiply, add and clip on one in nine ticks; nonlinear operations are not comparable to a fixed FLOP count. Recorded search training elapsed 0.275–0.398 seconds per arm; these CPU timings exclude validation, test, download and synthetic trials. The limited single-seed budget, short continuation, small evaluation set and limited candidate grammar are major restrictions; no significance or efficiency win is claimed.

## Verification

The first whole-suite run had 36 passes and two CLI failures because the package was not installed and the tests launch from an external working directory. Install the project with `python -m pip install -e .` before `python -m pytest -q`; the subsequent result is recorded in `verification.txt`. Snapshot replay was exercised across 30 + 10 ticks, and the focused tests compare restored sessions exactly. Previous 001–002b tracked files and checkpoints were not modified.
