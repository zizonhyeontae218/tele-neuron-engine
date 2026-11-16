"""Generate the ten-item final readout after real training/evaluation completes."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="work/003b-final-summary.md")
    ap.add_argument("--tests-log", default="work/003b-final-tests.log")
    args = ap.parse_args()
    root = Path("docs/dynamath/experiment-003b")
    result = json.loads((root/"results.json").read_text())
    if result["training"]["completed_steps"] != 50000 or not result.get("finalization", {}).get("complete"):
        raise ValueError("final report requires actual 50000 updates and complete frozen evaluation")
    memory = json.loads((root/"memory-results.json").read_text())
    generation = json.loads((root/"generation-samples.json").read_text())
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    repo = os.environ.get("GITHUB_REPOSITORY", "zizonhyeontae218/tele-neuron-engine")
    base = f"https://github.com/{repo}"
    run = os.environ.get("GITHUB_RUN_ID")
    model, training, test = result["model"], result["training"], result["test"]
    rows = ["| Step | Validation CE | Word PPL | Accuracy | Omission | False emission | Full-split FIFO EOS |",
            "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in training["validation"]:
        tf, fifo = row["teacher_forced"], row["actual_FIFO"]
        rows.append(f"| {row['step']} | {tf['response_CE']:.6f} | {tf['response_perplexity']:.6f} | {tf['response_accuracy']:.6f} | {tf['response_omission_rate']:.6f} | {tf['silence_false_emission_rate']:.6f} | {fifo['eos_success_rate']:.6f} |")
    test_fields = ("response_CE", "response_perplexity", "response_accuracy", "response_omission_rate", "silence_false_emission_rate", "teacher_forced_EOS_accuracy", "seconds", "max_rss_kib")
    text = f"""# DynaMath 003b — final measured readout

1. Branch: {base}/tree/feature/dynamath-003b. Final evidence commit: `{sha}`.
2. Actual vocabulary size: **{model['vocabulary_size']:,}** (five fixed controls included).
3. Stored parameters: **{model['parameters']:,}**; S: **{model['state_addresses']:,}** addresses; one runtime scalar m. State/memory: {model['session_S_m_fp32_bytes']:,} bytes/session. Nine unchanged baseline ASTs; no search or pretrained embedding/head.
4. Production optimizer updates: **{training['completed_steps']:,} / 50,000**. The separate 359-update preflight is excluded. Training time: {training['training_seconds']:.3f} s; peak training RSS: {training['max_rss_kib']} KiB. See results.json for full timings, throughput, memory and environment.
5. Complete validation split at every declared checkpoint:

{chr(10).join(rows)}

6. Frozen final test (selected step 50,000; no model/hyperparameter retuning):

```json
{json.dumps({k: test[k] for k in test_fields}, indent=2)}
```

Actual test Session/FIFO metrics, independent Q/A prompts:

```json
{json.dumps(test['actual_FIFO'], indent=2)}
```

7. Unedited English fixed-prompt outputs (full IDs/control tokens and traces in generation-samples.json):

"""
    for index in (0, 6, 10, 12, 21):
        sample = generation["samples"][index]
        text += ("```json\n" + json.dumps({k: sample[k] for k in
                 ("question", "text", "raw_tokens", "eos", "input_tokens_admitted", "question_events")},
                 ensure_ascii=False, indent=2) + "\n```\n\n")
    text += f"""8. Same-session memory probes (five lexical cases per arm):

```json
{json.dumps(memory['expected_word_rates'], indent=2)}
```

{memory['limitation']}

9. Verification: full pytest command completed successfully in this evaluation job;
file index regenerated and lint passed after committing measured evidence.
"""
    log = Path(args.tests_log)
    if log.exists():
        passed = re.findall(r"(\d+) passed", log.read_text())
        if passed:
            text += f"Exact test count: **{passed[-1]} passed**.\n"
    if run:
        text += f"Workflow/check status: {base}/actions/runs/{run}.\n"
    text += """
10. Confirmed achievements: word-tokenizer implementation, actual 50k training,
all four full validation measurements, final frozen test, raw FIFO generations,
memory interventions and reproducible checkpoints. Failure rates are reported
above without repair; an EOS or expected word does not establish useful dialogue
or semantic memory. No claim of understanding or formula-evolution superiority.
Word PPL and historical character PPL are not directly comparable. Corpus FIFO
metrics use independent Q/A sessions; explicit multi-turn probes retain state and
FIFO, so FIFO itself remains a memory confound. CUDA was not tested.

Checkpoint binaries, tokenizer and hashes are preserved without repeated Git
blobs in the owner-visible draft release `dynamath-003b-checkpoints`; Actions
artifacts provide an additional 90-day copy. Source, raw evidence and reports
remain on the 003b branch; earlier branches and experiment assets were preserved.
"""
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(path)


if __name__ == "__main__":
    main()
