"""Fixed 50k-update 003b contract, bounded wall-time segments and full milestones."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from dynamath.v003b import Config, Model
from dynamath.v003b.data import LaneStream, prepare
from dynamath.v003b.evaluation import teacher_forced, free_generation, memory_experiment
from dynamath.v003b.reporting import dump, write_reports
from dynamath.v003b.training import seed_all, run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dynamath/experiment-003b.json")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume")
    parser.add_argument("--models", default="models/dynamath/003b")
    parser.add_argument("--output", default="docs/dynamath/experiment-003b")
    parser.add_argument("--cache", default="work/dailydialog/data.zip")
    args = parser.parse_args()
    if args.seconds is not None and args.seconds <= 0:
        parser.error("seconds must be positive")
    cfg = json.loads(Path(args.config).read_text())
    if cfg["steps"] != 50000:
        raise ValueError("003b production target must remain 50000 updates")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    torch.set_num_threads(cfg["threads"])
    seed_all(cfg["seed"])
    splits, tokenizer, audit = prepare(args.cache, cfg["vocabulary_limit"])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out/"tokenizer.json")
    dump(out/"tokenizer-report.json", audit)
    # Commit before any generation/test; prompts are fixed in evaluation.py.
    dump(out/"evaluation-plan.json", {"config": cfg, "selection": cfg["selection"],
        "test_policy": "full test once after fixed final 50000; no test-driven changes",
        "data_sha256": audit["data_sha256"], "tokenizer_sha256": tokenizer.sha256})
    resume = None
    if args.resume:
        model, payload = Model.load(args.resume, device)
        resume = payload["restart"]
        if model.config.vocabulary != tokenizer.vocabulary:
            raise ValueError("checkpoint vocabulary mismatch")
    else:
        model = Model(Config(tokenizer.vocabulary, memories=1, latent=cfg["latent"], rank=cfg["rank"])).to(device)
    stream = LaneStream(splits["train"], tokenizer, cfg["batch"], cfg["seed"], shuffle=True, device=device)

    def milestone(model, step):
        metrics = teacher_forced(model, splits["validation"], tokenizer, cfg["batch"])
        generation = free_generation(model, tokenizer, cfg["response_slots"])
        dump(out/f"validation-{step:05d}.json", metrics)
        dump(out/f"generation-{step:05d}.json", generation)
        # Keep the required raw-sample filename honest even before completion.
        dump(out/"generation-samples.json", {"checkpoint_step": step, **generation})
        return {"teacher_forced": metrics, "actual_FIFO": generation["summary"]}

    training, restart = run_training(model, stream, cfg, tokenizer.sha256,
        audit["data_sha256"], args.models, args.seconds, resume, milestone)
    nparams = sum(p.numel() for p in model.parameters())
    result = {"schema": "dynamath-003b-results/v1", "training": training,
        "data_sha256": audit["data_sha256"], "tokenizer_sha256": tokenizer.sha256,
        "model": {"vocabulary_size": len(tokenizer.vocabulary), "parameters": nparams,
            "state_addresses": model.n, "memory_scalars": 1, "parameter_fp32_bytes": 4*nparams,
            "session_S_m_fp32_bytes": 4*(model.n+1),
            "batch_S_m_fp32_bytes": 4*cfg["batch"]*(model.n+1),
            "low_rank_matrix_forward_flops_per_lane_tick": 4*model.n*cfg["rank"],
            "cost_note": "Analytical low-rank matrix multiply/add only; excludes nonlinearities, backward, Adam and grouping overhead.",
            "asts": model.asts}, "test_status": "not run: final checkpoint not reached", "test": None}
    if training["complete"]:
        # Persist selection before loading or evaluating the test split.
        checkpoint = Path(args.models)/"latest.pt"
        selection = {"step": 50000, "criterion": cfg["selection"],
                     "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     "tokenizer_sha256": tokenizer.sha256, "data_sha256": audit["data_sha256"]}
        selected_path = out/"selected-checkpoint.json"
        if selected_path.exists() and json.loads(selected_path.read_text()) != selection:
            raise ValueError("immutable final selection mismatch")
        dump(selected_path, selection)
        test_path = out/"test-final.json"
        if test_path.exists():
            saved_test = json.loads(test_path.read_text())
            if saved_test["selection"] != selection:
                raise ValueError("test selection mismatch")
            result["test"] = saved_test["metrics"]
        else:
            result["test"] = teacher_forced(model, splits["test"], tokenizer, cfg["batch"])
            dump(test_path, {"selection": selection, "metrics": result["test"]})
        result["test_status"] = "full frozen test evaluated after final selection"
        memory = memory_experiment(model, tokenizer, cfg["response_slots"])
        dump(out/"memory-results.json", {"checkpoint_step": 50000, **memory})
    else:
        dump(out/"generation-samples.json", {"status": "pending: first milestone not reached", "samples": []}) if not (out/"generation-samples.json").exists() else None
        dump(out/"memory-results.json", {"status": "pending: final checkpoint not reached", "trials": []})
    checkpoints = []
    for path in sorted(Path(args.models).glob("*.pt")):
        checkpoints.append({"path": path.name, "bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    dump(out/"checkpoint-manifest.json", {"completed_steps": training["completed_steps"],
         "tokenizer_sha256": tokenizer.sha256, "data_sha256": audit["data_sha256"], "files": checkpoints})
    write_reports(out, result)
    print(json.dumps({"completed_steps": training["completed_steps"], "remaining_steps": training["remaining_steps"],
                      "complete": training["complete"], "training_seconds": training["training_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
