# DynaMath 003b — actual training status

Status: **50,000 updates completed**.
Actual completed updates: **50,000 / 50,000**;
remaining: **0**. Last preserved segment training time:
13550.503 seconds (optimizer work, excludes evaluation).

Vocabulary: 8,192; parameters:
4,441,203; S addresses: 16,448;
one runtime scalar m. All nine ASTs are the 003a baseline tanh(z).

## Milestones

| Step | Validation response CE | Word PPL | Token accuracy | FIFO EOS rate |
|---:|---:|---:|---:|---:|
| 5000 | 6.033361 | 417.114574 | 0.000000 | 0.000000 |
| 20000 | 5.997793 | 402.539334 | 0.000000 | 0.000000 |
| 35000 | 5.990084 | 399.448239 | 0.000000 | 0.000000 |
| 50000 | 5.984509 | 397.227288 | 0.000000 | 0.000000 |

`results.json` retains every measured rate, timing, memory and provenance field.
`training-progress.png` shows logged training CE/LR and available validation CE.
No milestone is filled from estimates. Test status: **full frozen test evaluated after final selection**.
The fixed final step 50,000 was selected before looking at test metrics.
Before corpus finalization, FIFO columns describe the fixed 22-prompt set.
After finalization they cover every paired turn in the complete validation
split; the fixed-prompt summary is retained separately as fixed_prompt_FIFO.

## Objective and execution

Response-token CE is weighted .75; all SILENCE checks together are weighted .25,
including question-time first checks and all second checks. Each class is averaged
over its own count across the entire 16-slot TBPTT window. If a window contains
no response tokens, that term contributes zero. This differs from 003a's equal
per-check objective. Teacher-forced metrics do not measure actual FIFO output.

Every lane consumes its full assigned data and wraps independently; full finite
evaluation masks exhausted lanes rather than truncating to the shortest one.
At each dialogue start, S, m and that lane's clock all restart; detach alone does
not reset a session. The order is shuffled once with seed 43 and repeated.

Model/runtime/optimizer/scheduler/RNG/data cursors are atomically preserved in
`models/dynamath/003b/latest.pt`; milestone files are stored at the four declared
steps. Binaries are excluded from Git and uploaded as Actions artifacts (90-day
retention); download them for longer archival. Small reports are committed only
on feature/dynamath-003b. Failed/time-limited execution never marks completion.

## Limits

No claim of useful conversation, semantic memory, general understanding or
superiority of formula evolution follows from implementation or loss alone.
Character PPL from 003a and word PPL here have different units and cannot be
ranked directly. The narrower de-duplicated corpus and loss also differ.
Raw generation retains repetition, silence, missing EOS, premature emissions,
unexpected control IDs and pending feedback; no answer repair is applied.
