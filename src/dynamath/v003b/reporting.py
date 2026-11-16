"""Inspectable status, milestone tables and standalone progress plots."""
import json
from pathlib import Path


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    tmp.replace(path)


def write_reports(root, result):
    root = Path(root)
    dump(root/"results.json", result)
    train = result["training"]
    rows = ["| Step | Validation response CE | Word PPL | Token accuracy | FIFO EOS rate |",
            "|---:|---:|---:|---:|---:|"]
    for entry in train["validation"]:
        teacher, actual = entry["teacher_forced"], entry["actual_FIFO"]
        rows.append(f"| {entry['step']} | {teacher['response_CE']:.6f} | {teacher['response_perplexity']:.6f} | {teacher['response_accuracy']:.6f} | {actual['eos_success_rate']:.6f} |")
    if len(rows) == 2:
        rows.append("| No milestone evaluated yet | — | — | — | — |")
    text = f"""# DynaMath 003b — actual training status

Status: **{'50,000 updates completed' if train['complete'] else 'INCOMPLETE'}**.
Actual completed updates: **{train['completed_steps']:,} / {train['target_steps']:,}**;
remaining: **{train['remaining_steps']:,}**. Last preserved segment training time:
{train['training_seconds']:.3f} seconds (optimizer work, excludes evaluation).

Vocabulary: {result['model']['vocabulary_size']:,}; parameters:
{result['model']['parameters']:,}; S addresses: {result['model']['state_addresses']:,};
one runtime scalar m. All nine ASTs are the 003a baseline tanh(z).

## Milestones

{chr(10).join(rows)}

`results.json` retains every measured rate, timing, memory and provenance field.
`training-progress.png` shows logged training CE/LR and available validation CE.
No milestone is filled from estimates. Test status: **{result['test_status']}**.
The fixed final step 50,000 was selected before looking at test metrics.

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
"""
    (root/"training-report.md").write_text(text)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    history = train["history"]
    for key, label in (("response_CE", "Response CE"), ("silence_CE", "SILENCE CE")):
        axes[0].plot([r["step"] for r in history], [r[key] for r in history], label=label)
    axes[0].set(ylabel="Training cross entropy", xlabel="Actual optimizer update")
    if history:
        axes[0].legend()
    axes[1].plot([r["step"] for r in history], [r["lr"] for r in history])
    axes[1].set(ylabel="Learning rate", xlabel="Actual optimizer update")
    val = train["validation"]
    if val:
        axes[2].plot([r["step"] for r in val], [r["teacher_forced"]["response_CE"] for r in val], marker="o")
    else:
        axes[2].text(.5, .5, "No validation milestone reached", ha="center", va="center", transform=axes[2].transAxes)
    axes[2].set(ylabel="Full validation response CE", xlabel="Actual optimizer update")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(root/"training-progress.png", dpi=160)
    plt.close(fig)
