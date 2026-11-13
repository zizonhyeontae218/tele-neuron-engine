# DynaMath 002 execution plan

Implement issue #12 and CODEX_HANDOFF.md; preserve every existing 001a asset.
The newer user request authorizes scalar-sharing controls: Model 9 (one/law),
Model 3 local and cross-read (one/block), Model 1 (one shared across all laws).
All have nine independent laws. Only current owner scalar may be written;
Model 3 cross-read reads all three, local masks other blocks. No external bank.

Core/runtime: `src/dynamath/v002/__init__.py`; experiment/CLI:
`src/dynamath/v002/__main__.py`; tests: `tests/test_dynamath_002.py`.
Use fixed address input/output, old scalar read before state update then bounded
hard assignment after state update; straight-through gate gradients in training.
Freeze seed, budgets, sample IDs and metrics in experiment-002.json before runs.
Train first readout to next target and second to SILENCE; both checked naturally
in runtime. Teacher forcing substitutes observed prior output at next input slot;
missed/double outputs at inference produce real FIFO delay, never target fallback.
No mode masks. Keep state and scalars across pair boundaries during stream training.

Validate invariants, exact restart including RNG/optimizer, legacy tests and artifact
audit. Run bounded Korean training and distinct delayed-bit training with all four
reset arms retaining absolute tick. Report failure and queue saturation honestly.
Primary results are a bounded subset/compute experiment, not a full convergence claim.
