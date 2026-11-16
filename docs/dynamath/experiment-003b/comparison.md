# DynaMath 003a / 003b comparison boundaries

| Item | 003a English pilot | 003b contract |
|---|---|---|
| Model | nine-law Model1, fixed control selected | same nine baseline tanh(z) ASTs |
| Vocabulary | 105 characters | 8192 word/punctuation/control tokens |
| Parameters | 74,223 | 4,441,203 including nine unused baseline alpha scalars |
| S size | 274 | 16,448 |
| Dataset | 512 training dialogues; 16/16 held-out pilot | all filtered splits, 9880/974/996 dialogues |
| Leakage | whole-dialogue filter; full-split Q/A overlap remained | zero exact adjacent Q/A overlap after whole-dialogue removal |
| Lane reset | S/m reset; shared law phase kept running | independent S/m/clock reset per dialogue |
| Lane coverage | shortest-lane truncation | every lane consumed in full |
| Objective | equal per-check CE | response .75 / SILENCE .25, class averages |
| Update target | 5000 pretrain plus pilot search | 50,000 fresh Adam updates; actual status in training-report.md |
| FIFO prompt budget | 180 ticks, sometimes exhausted before BOT | separate complete input admission and 96-slot response budget |

Comparable observations: runtime rule adherence, parameter/state/analytical cost,
raw repetition, nontermination, silence, question emissions, and bounded fixed
prompt behavior (with caveats for different budgets). The 003a example `o e e`
failed to show useful conversation. Inspect 003b raw samples before any claim.

Not directly comparable: character PPL versus word PPL, CE or token accuracy
across tokenizations; loss across changed weights; end-to-end efficiency across
different hardware/data coverage; causal benefits of word tokenization alone.
This phase changes multiple factors and is not a matched ablation. Fixed ASTs
cannot establish an advantage from formula evolution. Implementation and lower
training loss cannot establish understanding or semantic memory.
