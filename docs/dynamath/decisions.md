# DynaMath 1 decisions

| Question | Decision and evidence | Alternative deferred/rejected |
|---|---|---|
| What learns? | Only coefficients in five independent low-rank nonlinear laws. All inputs/output positions and zero initialization are fixed. | Learned embeddings, a large output head or an external LM would obscure whether the laws learn. |
| Formula family? | Leaky diagonal plus U*tanh(V*x+c), bounded by outer tanh. Inner nonlinearity allows nonlinear cross-address interactions at low rank. | A diagonal-only law cannot combine addresses. A dense N*N law raises per-tick cost. |
| Why five? | Required cyclic design, not an empirical claim of necessity. The single-law reference also reached 100% on both small tasks. | No search over law counts; no superiority claim. |
| Coefficients shared? | No. All six parameter groups are independent across phases; only information state is shared. | Shared weights would collapse the specified five-law distinction. |
| Training? | Full BPTT for short logic episodes and TBPTT for text; Adam, clipped gradients. This directly optimizes the formulas. | Evolutionary or symbolic search would require a larger budget and less direct gradient signal. |
| Stable memory? | Per-address leak and diagonal recurrence, bounded tanh. Short delayed recall demonstrated; idle boundedness proved. | No claim of perfect indefinite memory, stable semantics or universal dynamical computation. |
| Phase? | Logic sees all five start phases; text uses five-tick symbol cycles. Persist absolute ticks and align resumed prompts by actual idle ticks. | Exact mathematical phase invariance is not assumed or proved. |
| Event rule? | Fixed positive level threshold, deterministic winner, five-tick refractory, no state reset. Correct-address hinge connects training to eligibility. | Exact float equality and mandatory edge crossing would fail on repeated letters; a learned separate event model is unnecessary for the first prototype. |
| Text encoding? | 27 train-derived character symbols with fixed one-hot address addition. An intentionally tiny authored English corpus makes every data item inspectable. | Byte256 would leave little room for useful state at this size; multilingual/general dialogue is out of scope. |
| Output selection? | Greedy argmax among actual output addresses. | Sampling could conceal deterministic failure cases and would require event-compatible selection policy. |
| Training state? | Zero only for independent episodes/evaluation splits. Continuous character streams carry and detach state at TBPTT boundaries. | A trained initial state would violate the brief; resetting after emissions would destroy the core contract. |
| Validation? | Fixed disjoint dialogue pairs, training-only vocabulary, no holdout-based checkpoint selection. Logic uses fresh seed episodes and a longer-gap probe. | Held-out finite XOR combinations are not a novel truth table; this is episode generalization only. |
| Resource limit? | Exactly 600 five-law logic updates, 600 reference updates, 800 character updates; no search or production-model retraining. | Increasing data/model/training until attractive samples appear would hide first-experiment limitations. |
| Existing branch restriction? | Owner's attached brief explicitly authorizes the new feature branch. Root exception and local instructions record its scope. | Never align or merge this new family into master/endophalon3 automatically. |

Implementation refinement after the single training run: factored the event check into
`poll()` so tick and final-prompt inspection use one policy, validated inputs before
mutation, made sample generation compatible with arbitrary training alphabets, and
improved resume/benchmark metadata. No formula, learned coefficient, training schedule
or event decision changed. The published samples were rechecked with the final runtime;
the final-runtime benchmark is separate from the retained initial-runtime benchmark.

A full smoke run exercised the existing eight model/config combinations using temporary
small fixtures, as required by the repository. It did not alter saved Endophalon artifacts
or rerun their production training searches.

Final committed preservation audit found that the original `pyproject.toml` is
itself a pinned reproduction file. The initial optional-dependency edit therefore
failed that audit. Restored its exact original bytes and moved torch to the
separate `docs/dynamath/requirements.txt`. The correction changes packaging/docs,
not model coefficients, runtime behavior or the original experiment measurements.
