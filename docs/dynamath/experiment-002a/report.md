# DynaMath 002a — measured continuation

Original 002 checkpoints were resumed, not reinitialized. Same nine laws,
same formula parameterization and supervision; only the update budget changed.

| Model | Cumulative updates | Validation CE | Full-test CE | Full-test PPL | EOS / 17 |
|---|---:|---:|---:|---:|---:|
| model1 | 18000 | 1.6658 | 1.6648 | 5.285 | 17 |
| model3-cross | 18000 | 1.6694 | 1.6572 | 5.245 | 17 |
| model9-frozen | 18000 | 1.6942 | 1.6793 | 5.362 | 16 |

## Interpretation limits
- This trains coefficients of a fixed family of formulas, NOT symbolic formula discovery.
- Model1 has 9 laws and 1 shared scalar; Model3-cross has 9 laws and 3 cross-readable scalars.
- Original min-lane stream truncation retained to preserve exact optimizer continuation.
- Previous 002 test covered 96 pairs; 002a final test covers the full split. Comparisons have different denominators.
- Single seed; no held-out test selection; initial and new model files preserved.
- Teacher-forced PPL is not proof of coherent closed-loop dialogue.
- EOS and scalar writes alone are not proof of useful memory or response meaning.

Detailed stage records, checkpoint hashes and unedited generated text are in per-model progress JSON.
