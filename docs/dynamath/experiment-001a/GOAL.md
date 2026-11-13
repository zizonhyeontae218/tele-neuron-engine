# DynaMath 001a — frozen goal and experiment budget

Data credit: Song Ys (`songys`), https://github.com/songys/Chatbot_data, MIT.

Continue the existing feature/dynamath-001a branch at 15d897b. Preserve all
Experiment 001 and Endophalon checkpoints and evidence. No merge to master.

Stage A: isolate event scheduling without changing law coefficients/geometry.
Check after every tick, default refractory=1, no duplicate event on repeated
poll, no readout mutation. Test every phase, crossing, ties, silence, repeated
symbols, idle, additive queue, snapshot and first-tick emission. Explicit legacy
io_ticks=5/refractory=5 remains available for the original trained protocol.

Stage B: NFC ingest/NFD Unicode vocabulary built from training only; reserved
USER/BOT/EOS/SILENCE/UNK addresses. Unseen codepoints map to UNK, measured by split.
Five independent original low-rank laws, disjoint fixed input/output maps, latent
addresses; only law coefficients train. Question ticks target SILENCE; BOT tick
predicts first answer token; answer feedback predicts next, EOS feeds silence.
Absolute phase and shared state persist across tokens and conversations.

Pin original upstream CSV and MIT license. Normalize labels by stripping spaces.
Group questions by NFC/casefold/alphanumeric-only key (punctuation and whitespace
variants together); this reduces exact-normalized leakage but is not semantic
deduplication. Deterministic seed=43 grouped 80/10/10 split, before streams.
Freeze split IDs/hashes and eval plan before learning; never tune on final test.

Budget fixed before pilot: CPU one thread, one seed=43, batch16/TBPTT32,
Adam lr=.002, gradient clip=1, N=2*vocab+128, rank24, gain6. Pilot at most 4
updates on fixture data for bugs. One bounded run per configuration: five laws
1-tick; five laws 5-tick; one law 1-tick rank24; one law 1-tick rank chosen to
approximately match five-law parameters. Each at most 2400 updates or 600 seconds
of training, whichever comes first. No quality-driven reruns or checkpoint
selection; save last. All train pairs are eligible in sequential partitioned lanes.
Report actual unique coverage/tokens and any cap truncation. Evaluation may take
additional bounded CPU time; it never trains on validation/test.

Acceptance: runtime invariants and deterministic resume pass; actual Korean
training/checkpoint and reproducible decoded Korean free generation; report CE,
perplexity, policy errors, frozen prompt output, EOS/timeouts and semantic scores
against frequent answer baseline. Improvement is measured, not assumed. Memory
probes use delayed bit recall and Korean two-turn questions whose answer requires
the previous turn; zero-shot untrained probes measure limits, not trained recall.
Quality failure does not justify retraining or a useful-chatbot claim.

Implementation map: existing runtime.py gains explicit scheduling compatibility;
korean_data.py acquisition/grouping/tokenizer; korean.py train/eval/runtime/CLI;
tests/test_dynamath_001a.py; configs/dynamath/experiment-001a.json;
data/dynamath/001a/ manifests/license (downloaded CSV excluded); models/dynamath/001a/;
versioned evidence/report here. Update indexes with scripts/filetree_index.py.
