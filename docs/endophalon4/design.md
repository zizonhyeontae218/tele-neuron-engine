# Endophalon 4 preregistered protocol

## Boundaries and intended files
New package: runtime.py (fixed physics plus external force), experiment.py
(streams/cache/train-only heads), training.py (CEM and atomic state), evaluation.py
(held-out controls and uncertainty). Three scripts expose train/evaluate/audit.
configs/endophalon4 holds immutable protocol; models/endophalon4 holds both
independent final bundles and generation checkpoints; docs/endophalon4 holds
history, environment, audits, report. Existing packages/models are read-only.
FileTree.md will be regenerated with the repository index tool.

## Coupling and stability
One direction at a time. Sender and receiver use the exact saved version3
physics/theta, separate positions/velocities and independent local RNG streams.
Both advance every tick, including warmup32; 2 substeps/tick, interval16, delay12.
At tick start z=soft_zones(sender actual positions), z0=soft_zones(home).
q=tanh( ((z-z0)/0.02) @ tanh(u[0:16]) / 4 ).
center=world_size*(1+tanh(u[16:19]))/2;
width=0.3+1.7*(1+tanh(u[19]))/2;
gain=0.08*(1+tanh(u[20]))/2.
F_i=gain*q*exp(-||x_i-center||²/(2 width²))*(x_i-center)/(||x_i-center||+1e-9).
This external force is held for both existing integration substeps of that tick.
The spring, anchor, damping, mass, bath and integration equations are unchanged.
Per-node added force norm <=0.08; this bound is not a proof of global stability.
Check finite values every tick; long-horizon test after selection. Receiver head
reads ONLY its own 16 zones. No target/label or sender head in physical coupling.

Sender is unaffected by receiver, so exact sender states may be cached per tick
for candidate evaluation. The cache is an exact trajectory, not an approximation
or software target. Cache keys include protocol/model/source hashes and split.
A live simultaneous implementation must agree with cached execution in tests.

## Data and selection (fixed before running)
48 events per stream; score all four lags from event16 onward (32 events/lag).
Sender random instructions contain independently random a,b; target at event t
for lag L is sender a[t-L]. Receiver input bits/opcodes have separate seed and
stream namespace. Every split/direction has disjoint fixed seeds and IDs, outside
version3 confirmation IDs 60056–60119 and 61064–61127.
Train4, validation4, test16, confirmation32 streams per direction. Training is
intentionally small to fit the local CPU budget; uncertainty/overfitting are limits.
One train-only Ridge head (16 input, four outputs); score mean validation MSE.
CEM21 dimensions, elite6, sigma0.5/floor0.05, seeds84001/84002. Generation0 u=0;
then 40x24 using the inherited stable weighted-elite update. No early stopping.
Original operation heads stay frozen; report their per-op accuracy on the SAME
sender and receiver programs, uncoupled and coupled. No retraining to hide loss.

## Controls, inference and acceptance
All modes use identical clock, input streams, physics noise, and sample window.
Uncoupled and explicit-zero controls fit their own train-only memory heads.
Also report selected-head zero ablation. Shuffled control independently permutes
sender trajectory ticks within each stream with a fixed RNG, preserving marginal
states and breaking alignment; refit its memory head on training only.
Software-register reference uses the saved sender's actual predicted XOR output
on independent a,b=0 probe instructions; hold its prediction for each lag, feed
that register as receiver a on an XOR(a,0) probe and measure saved receiver output.
It is an explicit software memory mechanism, evaluated separately from physical
memory, with both systems advancing all 16 ticks/event. No oracle register.

Primary gate is pooled four-lag accuracy per direction >=0.75, improvement over
uncoupled >=0.10, pooled frozen operation accuracy drop <=0.03, finite trajectories
and deterministic replay. Report lag gates separately to avoid pooled concealment.
Use paired stream-cluster bootstrap (fixed seed, 2000 draws) for improvement CI;
streams, not correlated events, are independent units. No held-out model selection.
Confirmation runs only after both complete, writes an exclusive-use marker first;
a crash requires reporting incomplete confirmation, never silently reusing it.

## Resume and provenance
Atomic JSON via temporary file, fsync and os.replace; a SHA-256 envelope detects
corruption. Keep all compact generation states (no physical intermediate logs).
Newest valid checkpoint fallback allowed only if it cannot repeat a completed
generation (corrupt newer files cause refusal). Check exact model/config/source
hashes, environment/NumPy and optimizer RNG before continuing. Each command
advances at most five generations independently for both directions. Segment
logs record before/after generations/evaluations and checkpoint hashes.
Test continuous versus serialized/resumed optimizer and real physics evaluation.
No GitHub write until completion, evaluation, audit and tests. Commit uses the
specified message plus GitHub's skip-checks trailer to avoid triggering Actions.
