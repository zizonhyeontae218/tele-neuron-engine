# DynaMath 1 architecture

## Contract and mathematical laws

This is coefficient learning in a fixed function family, not symbolic formula discovery.
There is one row-vector information state `s` of length N per running instance.
Five laws have entirely independent coefficients; they do not have private states.
Batching during training merely runs independent instances in parallel.

At tick t, drain the pending input queue exactly once into a sparse additive vector:

```text
x = s_t + I_t
k = t mod 5
alpha_k = 0.05 + 0.90 sigmoid(a_k)
d_k = 1.5 tanh(q_k)
h = tanh(x V_k + c_k)
F_k(x) = (1-alpha_k) * x + alpha_k * tanh(d_k * x + h U_k + b_k)
s_(t+1) = F_k(x)
```

Products involving alpha/d are elementwise. Exactly one law executes per tick.
The next law is F0,F1,F2,F3,F4,F0,... . Input is applied before that law;
output addresses are inspected afterwards. An idle tick executes the same law with
I=0. No positions, springs, collisions, mass, forces or numerical integration exist.

| Tensor | Character model | Logic model | Learned? |
|---|---|---|---|
| s | [batch,160] | [batch,64] | Runtime only; zero initialization |
| U | [5,24,160] | [5,16,64] | Yes |
| V | [5,160,24] | [5,64,16] | Yes |
| c | [5,24] | [5,16] | Yes |
| b,q,a | each [5,160] | each [5,64] | Yes |
| input/output maps, gain, threshold | fixed | fixed | No |

Character vocabulary (27 symbols, training-only sorted Unicode codepoints):
`\n .:?abcdefghiklmnoprstuvwxy`. Addresses [0,27) receive +1 for characters;
addresses [27,54) emit the corresponding characters; [54,160) are additional
state addresses. All 160 addresses interact through the laws. No embedding or
trainable output projection exists. Core injection also supports signed amplitudes
and duplicate addresses using scatter-add. Unknown characters raise before queue mutation.

The logic vocabulary is `aAbBxr01`: 0/1 for the first operand, 2/3 for the second,
4/5 for XOR/recall query, and output addresses 14/15 for 0/1. Input amplitude is
random in [0.9,1.1] per episode (query amplitude +1). The 1-law comparison uses the
same state/rank, five ticks per input and identical data/optimizer settings, with
only one coefficient set. It is a deliberately smaller baseline, not a parameter-matched ablation.

## Events, persistence and feedback

Every tick runs an event eligibility check. Emission is permitted only when
`t_next mod 5 == 0`, making five ticks one I/O cycle. Choose the largest output
address; ties choose the lowest address. Emit if its raw state value >=0.10 and
at least five ticks have elapsed since the previous event. This is a level gate
with global refractory time, not a real-number equality or mandatory rising-edge
test. It permits consecutive equal letters. Simultaneous candidates yield only one
winner. Polling a boundary twice cannot duplicate an event. The detector never
writes to s; only its last-event clock changes. Gain=6 affects training logits,
not address values or the fixed threshold.

Prompt conditioning masks outward events while applying one character per five
ticks. At the final prompt boundary the same detector is polled once. Each emitted
character, when generation continues, is queued as +1 at its input address for
the next tick. No template or truth token is consulted at inference. On an empty
prompt generation starts with idle ticks. A silent cycle continues idle ticks,
without repeatedly feeding an old token. A generated newline ends the default
turn; the final token is not fed back unless generation continues. Raw `tick` and
realtime mode emit events without automatic feedback; generation explicitly adds it.

The CLI retains s and clocks across chat turns. A new Session, an independent
training episode or explicit evaluation split starts at zero; output itself never
resets anything. If resuming in the middle of a cycle, generate first uses up to
four genuine idle ticks to align the next input. Prompt ticks and alignment count
against max_ticks. An insufficient budget can condition only a prefix and return
an empty string. Default limits are 120 output characters and 2000 ticks, including
conditioning. No-output runs terminate at the tick limit. Ctrl-C/EOF/`/quit` stop
chat safely; optional state-out saves in finally. In tick realtime mode seconds/hz
control scheduling. A stopped process does no work; no background-time claim is made.

## Training

Only U,V,c,b,q,a learn, using Adam, gradient clipping at norm 1, and backpropagation
through all five laws. With continuous output values z and correct address y:

```text
L = cross_entropy(6*z, y) + 0.1 * mean(relu(0.25-z_y))
```

The auxiliary positive margin connects differentiable training to the fixed
0.10 event threshold. It is not a differentiable simulation of refractory logic;
it does not teach silence at every inappropriate/idle time. Held-out text reporting
uses plain cross entropy, excluding the hinge. Reported training loss includes it.

Logic: 600 full-episode BPTT updates, batch64, lr0.003, seed41, data generator42.
Each batch has independently sampled bits/operation/amplitudes, common random
0..2 full idle cycles after each operand, then the query. The next episode starts
at phase `step mod 5`, deliberately training all starting phases. No history is
reset as a consequence of an output; independent supervised episodes reset by design.

Characters: 800 TBPTT updates, 24 characters x batch16, lr0.002, seed43. Sixteen
seeded stream offsets proceed deterministically through the training text. The
state is detached at TBPTT boundaries, never zeroed there. Stream wrap preserves
state; its endpoint jumps to character zero after the penultimate input/last
character target, an artificial boundary documented as a limitation. Phase is
always 0 at character boundaries. Teacher forcing adds only the current input
character to predict the next. Validation uses only its preceding true characters,
with independent zero initialization; generated samples use predictions as feedback.

## Stability, memory and phase

Bounded outer tanh and alpha in (0.05,0.95) prevent idle explosion:
`||s_after_m||inf <= 1 + 0.95^m max(||s_start||inf-1,0)` without inputs.
For per-address input increments bounded by B each tick, a conservative bound is
`max(initial_bound, 1+19B)`. Arbitrarily large/unbounded input queues are not
covered by a bounded-input guarantee. The nonlinear Jacobian is not constrained
below one; contraction, faithful indefinite memory and semantic idle activity are
not guaranteed. Saturation and memory loss remain possible. The 10,000-tick idle
probe checks numerical stability, not meaningful long-term recollection.

Per-address learnable leaks and diagonal recurrence provide multiple time scales;
low-rank nonlinear interaction supplies cross-address computation. XOR and delayed
recall are evidence of short-term capacity, not arbitrary long-term memory.
Logic phase probes cover all five starts. Character training uses aligned cycles;
arbitrary-offset language quality is unproven. Saved runtime clocks and alignment
make phase behavior deterministic instead of claiming mathematical phase invariance.

## Size and complexity

Each law has `2Nr + r + 3N` coefficients. Five-law character model: 40,920 total,
8,184 active per tick; logic: 11,280 total, 2,256 active. The 1-law logic reference
has 2,256 total. Complexity is O(Nr) time, O(N+r) working activation per instance,
O(5Nr) model storage. Batched training additionally stores the unroll for BPTT.

The two matrix products use approximately `4Nr` floating-point operations per tick
(count a multiply and an add separately). A deliberately approximate accounting
including diagonal/leak arithmetic is `4Nr+12N+2r`: 17,328 for N160/r24, excluding
184 tanh evaluations, sigmoid/parameter transforms, indexing, event logic and Python
allocation. Therefore this is not an exact hardware instruction count.
FP32 formula weights occupy 163,680 bytes; persistent state 640 bytes per instance.
Optimizer states, Python/PyTorch, serialization and allocator overhead are separate.

## Checkpoints

Schema `dynamath-1/v1`, loaded using torch.load(weights_only=True) on CPU. All saves
include Config and six coefficient tensors. Character training additionally stores
Adam, torch RNG, the sixteen stream states/offsets, total step/tick/phase, corpus
SHA256, before-metrics and history. Resume restores exact numerical continuation
on the tested environment (a split-vs-uninterrupted test verifies this). `--steps`
is the total desired step count, not extra steps; changed corpus hashes and
backwards step counts are rejected.

Session snapshots store state, absolute ticks (phase=ticks mod5), pending queue,
last-event tick, threshold and refractory. They contain weights too, so are
self-contained. No stochastic sampling is used in runtime. The logic checkpoint
contains final weights, Adam, optimizer/data RNG, step and history; it is an
inference/reproducibility artifact, not exposed by the character-resume CLI.
