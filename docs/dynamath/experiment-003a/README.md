# DynaMath 003a — executable CPU pilot

This is a partial experiment, not evidence of successful formula evolution or useful conversation. The full requested 96/16/4 competition, 20k+ English training, multidimensional final selection, synthetic ground-truth task, S/m language ablations and multi-turn memory tests remain outstanding. See `architecture.md`, `data-audit.json`, `formula-library.json`, both `*-search.json` files, `english-results.json`, and `ablation.json` for exact methods and measured evidence. The previous 002/003 experiments and their checkpoints remain unchanged.

## Reproduce

From the repository root, install CPU PyTorch and `pip install -e '.[dev]'`. Network access is needed for pinned DailyDialog and Song Ys data; the data are cached in ignored `work/` and `data/dynamath/001a/ChatbotData.csv`, not committed. DailyDialog is CC BY-NC-SA 4.0, for noncommercial research; its revision, archive SHA-256, filtering and full overlap audit are recorded in `data-audit.json`.

```sh
python scripts/check_dynamath_003a_catalog.py
python scripts/train_dynamath_003a_english.py --steps 5000 --seconds 600
python scripts/run_dynamath_003a.py --language korean --steps 20
python scripts/run_dynamath_003a.py --language english --steps 20
python scripts/evaluate_dynamath_003a.py
python -m pytest -q
python -m dynamath.v003a models/dynamath/003a/english/control-final.pt --question 'Hello, how are you?' --ticks 180
```

These commands overwrite 003a results and checkpoints. To resume the English control to a cumulative target, use `python scripts/train_dynamath_003a_english.py --resume models/dynamath/003a/english/control.pt --steps 20000 --seconds 1200`; it checks the training stream hash and restores optimizer, RNG, S, m, clock and position. `--steps` is a cumulative target. The checkpoint stores the source pin and data path, but acquisition remains necessary on a fresh machine. Training and evaluation time limits may stop runs early: inspect completed steps before comparing arms.

The catalog has 20,000 canonical JSON ASTs, ten balanced construction labels of 2,000 each, SHA-256 `158b1c0bd75a08d6516c1addef235233f6c4190b001aaad46ecf13f9550fa1af`. Labels 0–9 cover unary forms, nested forms, sums, products, residuals, S reference, m reference, polynomial/periodic mixing, gating, and composite graphs. Every graph passed a nine-point finite probe; one is constant on that particular probe because a nonnegative m makes `poly(abs(m))−poly(m)` vanish. This is not proof of degeneracy for negative m. More general algebraic redundancy is not exhaustively removed, and formal shape, safety, diversity and performance coverage is incomplete.

## Results and interpretation

The fixed baseline was selected for both languages after the second pilot stage. Therefore all nine selected laws are `Fᵢ(S,m) = (1−aᵢ)⊙S+aᵢ⊙tanh(zᵢ)` for i=1,…,9, with individually learned parameters and zᵢ as defined in `architecture.md`. No evolved law won. Formula graph edits were real, but this pilot does not demonstrate that evolution helps.

Korean, starting from 002a step 18,000: after 40 matched updates, validation CE was 1.61845967 for fixed and 1.61847500 for its mutated offspring; fixed test CE/PPL/accuracy were 1.61625/5.03416/0.49414. Three generated answers were `좋은 건 같ᄋ`, `잘 거예요.`, `잘 거예요.`; 2/3 terminated with EOS, and the first has an uncomposed Jamo. The 002b model has a different prior budget and is only a historical reference, not a matched control; see 003's report.

English: independent vocabulary (105 symbols), latent 64, rank 12, 74,223 parameters; 5,000 pretraining updates, 960,000 lane-transitions and 179.07 s training wall time on this shared CPU. Validation CE/PPL were 3.4160/30.4478 before search. At 20 search updates candidate2 scored 3.16458 versus fixed 3.47820; at 40, fixed scored 3.23700 versus candidate2 continuation 3.29741 and offspring 3.29740. The fixed winner's test CE/PPL/accuracy were 3.17921/24.02772/0.13099; teacher-forced response omission was 0.52274 and false emissions in silence slots 0.02122. Three 180-tick test outputs were ` o e e e e e e e `, `  e e e e e ` and empty; none emitted EOS. The last prompt exceeded the input-admission budget (peak/final queue 106/47), so its empty response is not a clean response-quality measurement. Repeated letters and nontermination preclude a useful dialogue claim.

The English candidate2 formula restoration increased validation CE by 0.23214 without retraining; this indicates a local structural effect on that trained candidate, not superiority to the matched fixed control. Korean offspring restoration changed CE by −0.00001589. Restoration and generation evidence appear in the JSON files. No test score is presented for unselected candidates.

Each language's retained pilot used 140 total updates over seven arms/continuations, or 26,880 lane-transitions (batch 8, eight slots and three ticks per update). Along with the retained English pretraining this is 5,280 updates and 1,013,760 lane-transitions, excluding evaluation/generation. Approximate low-rank forward matrix arithmetic is 13.35 billion multiply/add FLOPs across these retained runs (English 13,152 and Korean 13,824 per lane/tick), excluding nonlinearities, gates, backward, optimizer and evaluation; actual total FLOPs were not measured. Search training wall times were approximately 3.47 s English and 12.07 s Korean, under differing shared-CPU contention, so they do not establish relative efficiency. An additional 5,000-update English run was discarded after discovering source split duplicates, and preliminary Korean pilot reruns were also performed; they are excluded from retained-run totals.

The original DailyDialog split contains 81/108/4 exact dialogue overlaps across train:validation/train:test/validation:test. Test has priority, then validation; 213 training and four validation dialogues were removed without reassigning them. Remaining whole-dataset exact Q/A overlaps are 349/309/26; the actual 512/16/16 pilot subsets had zero exact Q/A overlap. This is exact matching, not a guarantee against paraphrase leakage. English vocabulary is derived only from filtered training; observed UNK and normalized round-trip error rates are zero on the measured full splits. The train/validation/test counts after filtering are 10,905/996/1,000.

The earlier 003 small bit-recall experiment remains untouched and is not natural-language memory evidence. Free-generation relevance, repetition and Unicode were inspected only on the three shown examples; no comprehensive repetition rate or semantic evaluation was calculated. No FLOP profiler, session S/m reset study or synthetic ground-truth experiment has been executed in 003a. The scope and batching limitations are recorded in `architecture.md`.
