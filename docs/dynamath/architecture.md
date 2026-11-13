# DynaMath architecture

DynaMath applies learned coefficients in a fixed low-rank law family to one
persistent numeric information state. The current baseline is DynaMath 001a;
its source data, budget and measured limitations are in the versioned report.

## One tick

For row-vector state s, fixed addressed additive input I and absolute tick t:

```text
x = s + I
k = t mod laws                 # five normally; one for the control
alpha_k = 0.05 + 0.90 sigmoid(leak_k)
d_k = 1.5 tanh(diagonal_k)
h = tanh(x V_k + inner_bias_k)
s_next = (1-alpha_k) * x + alpha_k * tanh(d_k*x + h U_k + bias_k)
```

One tick applies exactly one law, including idle ticks without input. The five
coefficient sets are independent but act on the same state; they have no private
hidden states. Only U,V,inner_bias,bias,diagonal,leak learn. State starts at zero;
there is no learned tokenizer, embedding, output head or initial state.

## Fixed addresses and events

DynaMath 001a uses vocabulary112, N352/rank24. Inputs occupy [0,112), outputs
[112,224), and additional latent addresses [224,352). Training logits are gain6
multiplied by output addresses only. Neither softmax nor emission argmax selects
latent positions. Each tick adds pending input before applying the law, then
inspects output immediately. The highest output address wins (lowest index on
ties), with fixed threshold0.1 and refractory1. Repeated polling cannot emit twice
at the same tick. Readout changes only the event clock, never numeric state.

The Korean session treats mapped SILENCE as no outward event, competing with all
other output addresses. USER/BOT/EOS/SILENCE/UNK are fixed reserved symbols0..4.
NFC input is decomposed to NFD; vocabulary comes from training only; unseen
codepoints map to UNK. Decoding composes NFC and preserves malformed unattached
Jamo/replacement characters instead of repairing predictions.

USER/question ticks target SILENCE. The BOT injection tick predicts the first
answer token. Its emitted token is additive input at the following tick and
predicts its successor, ending in EOS. Teacher forcing uses true input tokens;
free generation uses predictions and idles on unexpected silence. Question-time
emissions are exposed in diagnostics. No inference answer lookup exists.

## Persistence and controls

State and absolute phase persist across tokens, turns, TBPTT detach boundaries,
and snapshots. Checkpoints retain pending input and event clock as appropriate.
Raw tick mode advances idle dynamics without automatic emitted-token feedback;
generation queues feedback explicitly. Process downtime performs no hidden work.

The explicit io_ticks=5 option retains original 5-tick eligibility boundaries for
the 001a timing control. One-law controls share the fixed mappings/protocol with
one coefficient set; the approximately parameter-matched variant changes rank.
Main baseline:89,880 coefficients,17,976 active/tick; 1,408 bytes FP32 state.
Approximate arithmetic/tick:4Nr+12N+2r, excluding nonlinearities and runtime costs.

Saved coefficients retain serialization schema dynamath-1/v1 for compatibility;
this is a file format identifier, not the experiment's name. The independent
src/dynamath package loads the unchanged 001a checkpoints. Cleanups never retrain
or rewrite saved weights. Future research must predeclare a new bounded run and
preserve the frozen baseline. Coherent conversation and semantic memory remain
unachieved in the current evidence.
