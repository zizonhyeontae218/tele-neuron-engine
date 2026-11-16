# DynaMath 003b architecture

003a Model1's equations, AST implementation and Session/FIFO semantics remain the
reference. The baseline `{'op':'z'}` means tanh(z); all nine ASTs retain this leaf.
The AST forge/search/mutation engine remains unchanged for future work, and no
formula search is run in 003b. We initialize from seed 43, not character weights.

For law i: h=tanh(S V_i+b_i+m Q_i),
z=1.5 tanh(d_i) S+h U_i+c_i+m R_i;
a=.05+.9 sigmoid(leak_i); S'=(1-a) S+a tanh(z).
A thresholded gate retains or replaces m with e*tanh(S' writer_i+wb_i).
Training uses the inherited straight-through gate estimator; inference uses
the same hard decision. Nine laws have independent coefficients, one common S
per session and exactly one mutable scalar m. The AST alpha coefficient is
retained for compatibility, although the baseline leaf does not use it.

## Addresses and cost

For W=8192, inputs occupy S[0:8192], output scores occupy S[8192:16384],
and 64 latent addresses occupy S[16384:16448]. S has 16,448 FP32 addresses;
with m, one session uses 65,796 bytes; batch 8 uses 526,368 bytes for S/m only.
There are 4,441,203 stored parameters (17,764,812 FP32 bytes), including nine
inactive baseline alpha scalars. Adam moments and gradients add storage;
autograd through 16 slots adds temporary memory. Reports record peak process
RSS and CUDA allocator bytes separately from this analytical state footprint.
The low-rank matrix paths cost 4*N*rank = 789,504 multiply/add FLOPs per lane
tick, excluding all other arithmetic/backward/optimizer costs. This analytical
quantity is not a profiler measurement.

Parameter storage is split into nine modules to avoid constructing full-nine-law
slice gradients for every tick. A focused test verifies bitwise agreement with
003a equations for all nine laws across 18 ticks at identical initial RNG.
No token embedding or output projection is introduced; logits are gain 6 times
the fixed output slice. Input adds one at its fixed token address.

## Data and lane clocks

NFC exact full-dialogue duplicates are removed within and across splits. Every
adjacent utterance pair is checked for cross-split leakage, a stricter policy
than checking only training pairs (0,1), (2,3), etc. Test has highest priority,
then validation, then train; conflicts remove whole dialogues. The actual
training pairs preserve 003a's nonoverlapping speaker-order pairs. Vocabulary
frequency uses every utterance in the filtered training split (including odd
final utterances), and never validation/test. Limits and OOV are in JSON.

The data order is shuffled once using seed 43, assigned round-robin to eight
lanes, then repeated in that fixed order. Each lane wraps independently only
after consuming all its slots. Unequal lanes therefore have different epoch
counts, explicitly recorded; no lane data are discarded. Full finite evaluation
visits each slot once and masks exhausted lanes. No evaluation subset cap exists.

Each dialogue starts S=0, m=0, clock=0. Clocks, S and m persist across its
pairs and TBPTT windows. Rows are grouped by their own law phase at execution;
only F_(clock mod 9) acts at a tick. All slots are three ticks: input on the
first, output checks on the next two. Dialogue reset always resumes with F1.

## Training versus generation

The teacher-forced stream injects correct answer tokens; it does not inject
predictions. Response and SILENCE CE are class-averaged over the full TBPTT
window and weighted .75/.25. This is deliberately different from 003a's .5/.5
per-check average. Both first-check question SILENCE and second-check SILENCE
are supervised. Response omission/false emissions apply the actual .1 threshold.

Closed-loop generation uses the unchanged Session: output is enqueued as FIFO
feedback, state is retained, and question-time emission is allowed and logged.
Input has its own budget proportional to all pending FIFO and prompt tokens;
the response budget starts only when that turn's external BOT is admitted.
Neither pending feedback nor unexpected controls are cleaned between turns.
Generation trace/IDs expose all failures; the display decoder alone hides
control spellings except UNK, while raw_tokens preserves every control event.

Memory probes clone the same-session context snapshot, reset S only, m only or
both, and retain clock/FIFO. They also swap the stated fact. Pending FIFO is an
additional information channel, so these probes cannot isolate semantic memory
in S/m. Five exact-word probes alone do not establish natural-language memory.

## Failure and continuation

Checkpoint writes use atomic rename. Every 500 completed updates and on a clean
time cap, save latest; save all four milestones. SIGINT/SIGTERM request a stop
after the current complete update. Hash mismatches reject resumes. A crashed
partial optimizer update is discarded; restore the last complete checkpoint.
Exact restart was tested on one CPU/software environment, not across devices.
Actions uploads binaries even after failed training and chains bounded segments.
Large binaries never enter repeated Git commits; small evidence is branch-only.
