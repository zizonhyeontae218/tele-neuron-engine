"""Complete full-corpus closed-loop metrics from the four frozen checkpoints.

This never trains, changes hyperparameters or reselects a checkpoint. The main
runner already measures full teacher-forced validation/test and fixed prompts.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import torch
from dynamath.v003b import Model
from dynamath.v003b.data import prepare
from dynamath.v003b.evaluation import corpus_generation
from dynamath.v003b.reporting import dump, write_reports


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="models/dynamath/003b")
    ap.add_argument("--output", default="docs/dynamath/experiment-003b")
    args = ap.parse_args()
    torch.set_num_threads(2)
    root, models = Path(args.output), Path(args.models)
    result = json.loads((root/"results.json").read_text())
    if result["training"]["completed_steps"] != 50000 or not result["training"]["complete"]:
        raise ValueError("cannot finalize an incomplete production run")
    splits, tok, audit = prepare()
    selection = json.loads((root/"selected-checkpoint.json").read_text())
    if hashlib.sha256((models/"latest.pt").read_bytes()).hexdigest() != selection["checkpoint_sha256"]:
        raise ValueError("final checkpoint selection hash mismatch")
    if result["data_sha256"] != audit["data_sha256"] or result["tokenizer_sha256"] != tok.sha256:
        raise ValueError("frozen evaluation source mismatch")

    def measure(path, split, step):
        saved_path = root/f"closed-loop-{split}-{step:05d}.json.gz"
        identity = {"checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "split_sha256": audit["splits"][split]["sha256"], "tokenizer_sha256": tok.sha256,
                    "response_slots": 96, "step": step, "protocol": "independent-QA-Session/v1"}
        if saved_path.exists():
            with gzip.open(saved_path, "rt") as f:
                saved = json.load(f)
            if saved["identity"] != identity:
                raise ValueError("immutable closed-loop evaluation mismatch")
            return saved["summary"]
        model, payload = Model.load(path)
        if payload["restart"]["completed_steps"] != step:
            raise ValueError("checkpoint step mismatch")
        if payload["restart"]["tokenizer_sha256"] != tok.sha256 or payload["restart"]["data_sha256"] != audit["data_sha256"]:
            raise ValueError("checkpoint/data mismatch")
        actual = corpus_generation(model, tok, splits[split], slots=96)
        tmp = saved_path.with_suffix(".gz.tmp")
        # Deterministic gzip header; preserve raw tokens and every failure.
        with tmp.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as compressed:
                compressed.write(json.dumps({"identity": identity, **actual}, ensure_ascii=False).encode())
        tmp.replace(saved_path)
        print(json.dumps({"split": split, "step": step, **actual["summary"]}), flush=True)
        return actual["summary"]

    validation = {row["step"]: row for row in result["training"]["validation"]}
    for step in (5000, 20000, 35000, 50000):
        if step not in validation:
            raise ValueError(f"missing full teacher-forced validation milestone {step}")
        entry = validation[step]
        entry.setdefault("fixed_prompt_FIFO", entry["actual_FIFO"])
        entry["actual_FIFO"] = measure(models/f"step-{step:05d}.pt", "validation", step)
        write_reports(root, result)
    # Selection is immutable and already fixed to 50k; this is closed-loop test
    # scoring only. Do not repeat or tune the saved teacher-forced test result.
    result["test"]["actual_FIFO"] = measure(models/"latest.pt", "test", 50000)
    result["finalization"] = {"complete": True,
        "validation_generation_pairs_per_checkpoint": audit["splits"]["validation"]["pairs"],
        "test_generation_pairs": audit["splits"]["test"]["pairs"],
        "method": "entire filtered splits, fresh actual Session/FIFO per Q/A pair; multi-turn memory assessed separately; teacher-forced test not rerun"}
    write_reports(root, result)
    goal = root/"GOAL.md"
    goal.write_text(goal.read_text().replace("- [ ]", "- [x]"))


if __name__ == "__main__":
    main()
