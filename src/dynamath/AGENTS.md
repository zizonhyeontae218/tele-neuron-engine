# DynaMath implementation

Follow root AGENTS.md and GOAL.md. DynaMath 001a is the current frozen baseline;
its original experiment goal is docs/dynamath/experiment-001a/GOAL.md. The latest
user-authorized cleanup supersedes old instructions to retain unrelated models.

Run python -m dynamath --help. Korean CLI/training: korean.py; deterministic
corpus/tokenizer: korean_data.py; shared setup/loss: training.py; laws: core.py;
state/events/checkpoints: runtime.py. The package has no other model dependency.

Preserve one shared persistent state, five cyclic independently learned laws,
additive injection/feedback, immediate per-tick detection, and no output mutation.
Only law coefficients learn. SILENCE is an ordinary mapped address; expose question
emissions and failed outputs. Keep the 5-tick and one-law experimental controls.
Checkpoint schema dynamath-1/v1 is retained for saved-weight compatibility; it is
not the current experiment designation. No unbudgeted training or target lookup.

002 is isolated in v002; its nine laws and runtime writable scalars follow the
new handoff and authorized Model 9/3/1 topology controls. Legacy code stays frozen.
