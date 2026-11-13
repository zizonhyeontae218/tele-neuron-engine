# DynaMath 002b — final measured continuation

Original 002a 18,000-update checkpoints were resumed, not reinitialized. Same nine laws,
same formula parameterization and supervision; only the update budget changed.

| Model | Cumulative updates | Validation CE | Full-test CE | Full-test PPL | EOS / 17 |
|---|---:|---:|---:|---:|---:|
| model1 | 72000 | 1.5116 | 1.5083 | 4.519 | 17 |
| model3-cross | 72000 | 1.5192 | 1.5054 | 4.506 | 13 |

## Comparison to 002a on the same full test split
| Model | 002a CE | 002b CE | CE change |
|---|---:|---:|---:|
| model1 | 1.6648 | 1.5083 | -0.1565 |
| model3-cross | 1.6572 | 1.5054 | -0.1519 |

## Interpretation limits
- This trains coefficients of a fixed family of formulas, NOT symbolic formula discovery.
- Model1 has 9 laws and 1 shared scalar; Model3-cross has 9 laws and 3 cross-readable scalars.
- Original min-lane stream truncation retained to preserve exact optimizer continuation.
- 002a and 002b final test use the same full split, objective and readout schedule.
- Single seed; no held-out test selection; initial and new model files preserved.
- Teacher-forced PPL is not proof of coherent closed-loop dialogue.
- EOS and scalar writes alone are not proof of useful memory or response meaning.

Detailed stage records, checkpoint hashes and unedited generated text are in per-model progress JSON.
