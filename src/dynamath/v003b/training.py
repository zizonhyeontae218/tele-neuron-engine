"""Atomic checkpoints including optimizer, schedule, RNG and per-lane sessions."""
import json
import math
import platform
import random
import resource
import signal
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import LaneStream
from .model import block
from .tokenizer import digest


def environment(device):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "platform": platform.platform(),
            "device": str(device), "threads": torch.get_num_threads(),
            "cuda": torch.version.cuda, "source_commit": commit}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_at_update(update, cfg):
    if update <= cfg["warmup"]:
        return cfg["lr"] * update / cfg["warmup"]
    fraction = min(1., (update-cfg["warmup"]) / (cfg["steps"]-cfg["warmup"]))
    return cfg["minimum_lr"] + .5*(cfg["lr"]-cfg["minimum_lr"])*(1+math.cos(math.pi*fraction))


def contract_hash(cfg):
    return digest({k: v for k, v in cfg.items() if k not in
                   ("threads", "seconds_per_segment", "checkpoint_interval")})


def run_training(model, stream, cfg, tokenizer_sha256, data_sha256, output,
                 seconds=None, resume=None, on_milestone=None, stop_at=None):
    """stop_at is for focused tests; production always targets cfg['steps']."""
    device = next(model.parameters()).device
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch: lr_at_update(epoch+1, cfg)/cfg["lr"])
    s, m = model.zero(cfg["batch"])
    clocks = torch.zeros(cfg["batch"], device=device, dtype=torch.long)
    step, history, elapsed_before, slots_before, measured_steps_before = 0, [], 0., 0, 0
    last_validation = []
    if resume is not None:
        for key, expected in (("contract_sha256", contract_hash(cfg)),
                              ("tokenizer_sha256", tokenizer_sha256), ("data_sha256", data_sha256)):
            if resume[key] != expected:
                raise ValueError(f"resume {key} mismatch")
        opt.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        stream.load_state_dict(resume["stream"])
        s, m = resume["s"].to(device), resume["m"].to(device)
        clocks = resume["clocks"].to(device)
        step = resume["completed_steps"]
        history = list(resume["history"])
        last_validation = list(resume["validation"])
        elapsed_before = resume["training_seconds"]
        slots_before = resume["trained_lane_slots"]
        measured_steps_before = step
        random.setstate(resume["python_rng"])
        np.random.set_state(resume["numpy_rng"])
        torch.set_rng_state(resume["torch_rng"])
        if resume["cuda_rng"] is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all(resume["cuda_rng"])
    stop = {"requested": False}
    previous_handlers = {}
    def request_stop(signum, frame):
        stop["requested"] = True
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, request_stop)
    start = time.monotonic()
    train_seconds = 0.
    tokens_seen = slots_before
    target = cfg["steps"] if stop_at is None else min(stop_at, cfg["steps"])
    cap = cfg["seconds_per_segment"] if seconds is None else seconds
    env = environment(device)
    latest = None

    def checkpoint(path):
        nonlocal latest
        latest = {"completed_steps": step, "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(), "stream": stream.state_dict(),
            "s": s.detach(), "m": m.detach(), "clocks": clocks.clone(),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "contract_sha256": contract_hash(cfg), "tokenizer_sha256": tokenizer_sha256,
            "data_sha256": data_sha256, "history": history, "validation": last_validation,
            "training_seconds": elapsed_before+train_seconds, "trained_lane_slots": tokens_seen,
            "environment": env, "config": cfg,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cuda_peak_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0}
        model.save(path, restart=latest)

    model.train()
    try:
        while step < target and not stop["requested"]:
            # Always complete an update before preserving state; never checkpoint
            # detached S from half an optimizer update.
            if time.monotonic()-start >= cap:
                break
            tick_start = time.monotonic()
            opt.zero_grad(set_to_none=True)
            response_sum = s.new_zeros(())
            silence_sum = s.new_zeros(())
            responses = silences = 0
            for _ in range(cfg["tbptt_slots"]):
                x, y, reset, active = stream.next()
                s, m, clocks, out = block(model, s, m, clocks, x, reset, active)
                is_response = y != 3
                ce = F.cross_entropy(out[0], y, reduction="none")
                response_sum = response_sum + ce[is_response].sum()
                silence_sum = silence_sum + ce[~is_response].sum()
                silence_sum = silence_sum + F.cross_entropy(out[1], torch.full_like(y, 3), reduction="sum")
                responses += int(is_response.sum())
                silences += int((~is_response).sum()) + len(y)
                tokens_seen += int(active.sum())
            # Class-balanced objective across ALL response and SILENCE checks in
            # this TBPTT window, including first-check question SILENCE slots.
            response_ce = response_sum / max(responses, 1)
            silence_ce = silence_sum / max(silences, 1)
            loss = cfg["response_weight"]*response_ce + cfg["silence_weight"]*silence_ce
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite loss; latest complete update is preserved")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
            used_lr = opt.param_groups[0]["lr"]
            opt.step()
            scheduler.step()
            s, m = s.detach(), m.detach()
            step += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            train_seconds += time.monotonic()-tick_start
            if step == 1 or step % 100 == 0 or step in cfg["milestones"]:
                row = {"step": step, "loss": float(loss.detach()),
                       "response_CE": float(response_ce.detach()), "silence_CE": float(silence_ce.detach()),
                       "response_checks": responses, "silence_checks": silences,
                       "lr": used_lr, "gradient_norm_before_clip": float(grad),
                       "training_seconds": elapsed_before+train_seconds}
                history.append(row)
                print(json.dumps(row), flush=True)
            if step in cfg["milestones"]:
                checkpoint(output/f"step-{step:05d}.pt")
                if on_milestone is not None:
                    result = on_milestone(model, step)
                    last_validation.append({"step": step, **result})
                    model.train()
                    checkpoint(output/f"step-{step:05d}.pt")
            if step % cfg["checkpoint_interval"] == 0:
                checkpoint(output/"latest.pt")
        checkpoint(output/"latest.pt")
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    model.eval()
    return {"completed_steps": step, "target_steps": cfg["steps"],
            "remaining_steps": cfg["steps"]-step, "complete": step == cfg["steps"],
            "training_seconds": elapsed_before+train_seconds,
            "segment_seconds": time.monotonic()-start, "segment_updates": step-measured_steps_before,
            "trained_lane_slots": tokens_seen, "history": history, "validation": last_validation,
            "positions": stream.positions.tolist(), "lane_epochs": stream.epochs.tolist(),
            "lane_lengths": stream.lengths.tolist(), "clocks": clocks.tolist(),
            "max_rss_kib": latest["max_rss_kib"], "cuda_peak_bytes": latest["cuda_peak_bytes"],
            "environment": env, "training_input_slots_per_second": tokens_seen/max(elapsed_before+train_seconds, 1e-9)}, latest
