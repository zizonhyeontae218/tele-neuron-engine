"""Inspect a trusted 003b checkpoint without using test for interim selection."""
import argparse
import json
from pathlib import Path
import torch
from dynamath.v003b import Model
from dynamath.v003b.data import prepare
from dynamath.v003b.evaluation import teacher_forced, free_generation, memory_experiment
from dynamath.v003b.reporting import dump


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default="work/003b-evaluation")
    ap.add_argument("--mode", choices=["validation", "generate", "memory"], default="validation")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(2)
    splits, tok, audit = prepare()
    model, p = Model.load(args.checkpoint, args.device)
    if p["restart"]["tokenizer_sha256"] != tok.sha256 or p["restart"]["data_sha256"] != audit["data_sha256"]:
        raise ValueError("checkpoint/data/tokenizer mismatch")
    if args.mode == "validation":
        result = teacher_forced(model, splits["validation"], tok)
    elif args.mode == "generate":
        result = free_generation(model, tok)
    else:
        result = memory_experiment(model, tok)
    dump(Path(args.output)/(args.mode+".json"), {"checkpoint_step": p["restart"]["completed_steps"], **result})
    print(json.dumps({"mode": args.mode, "checkpoint_step": p["restart"]["completed_steps"], "output": args.output}))


if __name__ == "__main__":
    main()
